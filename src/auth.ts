/**
 * Authentication + user-data routes.
 *
 * Email/password sign-up (collecting the full student profile up front) and
 * login, both returning a JWT. Protected routes verify the JWT and attach the
 * user id to the request. Also serves /me (profile get/update) and /messages
 * (chat history) for the signed-in student.
 */
import { Router } from "express";
import type { Request, Response, NextFunction } from "express";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import {
  pool,
  createUser,
  getUserByEmail,
  getUserById,
  updateUser,
  getMessages,
  addMessage,
  listConversations,
  createConversation,
  renameConversation,
  deleteConversation,
  conversationOwnedBy,
  ensureDefaultConversation,
  rowToUser,
  DEFAULT_CHAPTERS,
} from "./db.js";

// Compact unique id for new conversations (no external deps needed).
function newId(prefix = "conv"): string {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

const JWT_SECRET = process.env.JWT_SECRET || "dev-insecure-secret-change-me";
const TOKEN_TTL = "30d";

if (!process.env.JWT_SECRET) {
  console.warn("[Auth] JWT_SECRET not set — using an insecure dev secret. Set JWT_SECRET before deploying.");
}

export function signToken(userId: number, email: string): string {
  return jwt.sign({ userId, email }, JWT_SECRET, { expiresIn: TOKEN_TTL });
}

/** Verify a raw token string (used by the WebSocket upgrade). */
export function userIdFromToken(token: string | null): number | null {
  if (!token) return null;
  try {
    const payload = jwt.verify(token, JWT_SECRET) as { userId: number };
    return payload.userId ?? null;
  } catch {
    return null;
  }
}

/** Express middleware: require a valid Bearer JWT. */
export function requireAuth(req: Request, res: Response, next: NextFunction) {
  const header = req.headers.authorization || "";
  const token = header.startsWith("Bearer ") ? header.slice(7) : null;
  const userId = userIdFromToken(token);
  if (!userId) return res.status(401).json({ error: "Please sign in to continue." });
  (req as any).userId = userId;
  next();
}

export const authRouter = Router();

// --- Sign up: collect the full student profile in one go ---
authRouter.post("/auth/signup", async (req: Request, res: Response) => {
  try {
    const {
      email,
      password,
      name,
      board,
      grade,
      language,
      preferredAnalogy,
      examGoals,
      confidenceLevel,
    } = req.body || {};

    if (!email || !/^\S+@\S+\.\S+$/.test(email)) {
      return res.status(400).json({ error: "Please enter a valid email address." });
    }
    if (!password || String(password).length < 6) {
      return res.status(400).json({ error: "Password must be at least 6 characters." });
    }

    const existing = await getUserByEmail(pool, email);
    if (existing) {
      return res.status(409).json({ error: "An account with this email already exists. Try signing in." });
    }

    const passwordHash = await bcrypt.hash(String(password), 10);
    const row = await createUser(pool, {
      email,
      passwordHash,
      name: (name || "Student").trim(),
      board: board || "CBSE",
      grade: grade || "11th Grade",
      language: language || "Hinglish",
      preferredAnalogy: preferredAnalogy || "Daily Life",
      examGoals: examGoals || "",
      confidenceLevel: Number(confidenceLevel) || 3,
      chapters: DEFAULT_CHAPTERS,
    });

    const token = signToken(row.id, row.email);
    res.status(201).json({ token, user: rowToUser(row) });
  } catch (err: any) {
    // Unique-violation safety net (race with the existence check).
    if (err?.code === "23505") {
      return res.status(409).json({ error: "An account with this email already exists." });
    }
    console.error("Signup error:", err);
    res.status(500).json({ error: "Could not create your account. Please try again." });
  }
});

// --- Log in ---
authRouter.post("/auth/login", async (req: Request, res: Response) => {
  try {
    const { email, password } = req.body || {};
    if (!email || !password) {
      return res.status(400).json({ error: "Email and password are required." });
    }
    const row = await getUserByEmail(pool, email);
    if (!row || !(await bcrypt.compare(String(password), row.password_hash))) {
      return res.status(401).json({ error: "Incorrect email or password." });
    }
    const token = signToken(row.id, row.email);
    res.json({ token, user: rowToUser(row) });
  } catch (err) {
    console.error("Login error:", err);
    res.status(500).json({ error: "Could not sign you in. Please try again." });
  }
});

// --- Current user ---
authRouter.get("/me", requireAuth, async (req: Request, res: Response) => {
  const row = await getUserById(pool, (req as any).userId);
  if (!row) return res.status(404).json({ error: "User not found." });
  res.json({ user: rowToUser(row) });
});

// --- Update profile + study log ---
authRouter.put("/me", requireAuth, async (req: Request, res: Response) => {
  try {
    const b = req.body || {};
    const row = await updateUser(pool, (req as any).userId, {
      name: (b.name || "Student").trim(),
      board: b.board || "CBSE",
      grade: b.grade || "11th Grade",
      language: b.language || "Hinglish",
      preferredAnalogy: b.preferredAnalogy || "Daily Life",
      examGoals: b.examGoals || "",
      confidenceLevel: Number(b.confidenceLevel) || 3,
      chapters: Array.isArray(b.chapters) ? b.chapters : [],
    });
    if (!row) return res.status(404).json({ error: "User not found." });
    res.json({ user: rowToUser(row) });
  } catch (err) {
    console.error("Profile update error:", err);
    res.status(500).json({ error: "Could not save your changes." });
  }
});

// --- Conversations (separate chat windows) ---

// List the student's conversations, creating a default one on first visit
// (also adopts any legacy messages saved before conversations existed).
authRouter.get("/conversations", requireAuth, async (req: Request, res: Response) => {
  try {
    const uid = (req as any).userId;
    await ensureDefaultConversation(pool, uid, newId());
    const conversations = await listConversations(pool, uid);
    res.json({ conversations });
  } catch (err) {
    console.error("List conversations error:", err);
    res.status(500).json({ error: "Could not load your conversations." });
  }
});

authRouter.post("/conversations", requireAuth, async (req: Request, res: Response) => {
  try {
    const title = (req.body?.title || "New chat").toString().slice(0, 120);
    const conversation = await createConversation(pool, (req as any).userId, newId(), title);
    res.status(201).json({ conversation });
  } catch (err) {
    console.error("Create conversation error:", err);
    res.status(500).json({ error: "Could not start a new chat." });
  }
});

authRouter.patch("/conversations/:id", requireAuth, async (req: Request, res: Response) => {
  try {
    const title = (req.body?.title || "").toString().trim().slice(0, 120);
    if (!title) return res.status(400).json({ error: "A title is required." });
    await renameConversation(pool, (req as any).userId, String(req.params.id), title);
    res.json({ ok: true });
  } catch (err) {
    console.error("Rename conversation error:", err);
    res.status(500).json({ error: "Could not rename this chat." });
  }
});

authRouter.delete("/conversations/:id", requireAuth, async (req: Request, res: Response) => {
  try {
    await deleteConversation(pool, (req as any).userId, String(req.params.id));
    res.json({ ok: true });
  } catch (err) {
    console.error("Delete conversation error:", err);
    res.status(500).json({ error: "Could not delete this chat." });
  }
});

// --- Chat history (scoped to one conversation) ---
authRouter.get("/conversations/:id/messages", requireAuth, async (req: Request, res: Response) => {
  const uid = (req as any).userId;
  if (!(await conversationOwnedBy(pool, uid, String(req.params.id)))) {
    return res.status(404).json({ error: "Conversation not found." });
  }
  const messages = await getMessages(pool, uid, String(req.params.id));
  res.json({ messages });
});

authRouter.post("/conversations/:id/messages", requireAuth, async (req: Request, res: Response) => {
  try {
    const uid = (req as any).userId;
    const { id, role, text, mode, sources, attachments } = req.body || {};
    if (!id || !role || typeof text !== "string") {
      return res.status(400).json({ error: "Invalid message." });
    }
    if (!(await conversationOwnedBy(pool, uid, String(req.params.id)))) {
      return res.status(404).json({ error: "Conversation not found." });
    }
    await addMessage(pool, uid, { id, conversationId: String(req.params.id), role, text, mode, sources, attachments });
    res.status(201).json({ ok: true });
  } catch (err) {
    console.error("Add message error:", err);
    res.status(500).json({ error: "Could not save message." });
  }
});
