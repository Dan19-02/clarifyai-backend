/**
 * DB-layer smoke test using an in-memory Postgres (pg-mem) — no real database
 * needed. Validates the schema + the exact queries used by signup/login/profile/
 * messages/cache, plus the bcrypt + JWT flow. Run with: npm run test:db
 */
import { newDb } from "pg-mem";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import {
  initSchema,
  createUser,
  getUserByEmail,
  getUserById,
  updateUser,
  addMessage,
  getMessages,
  cacheGet,
  cacheSet,
  rowToUser,
  DEFAULT_CHAPTERS,
} from "./db.js";

function assert(cond: any, label: string) {
  if (!cond) throw new Error("FAILED: " + label);
  console.log("  ✓ " + label);
}

(async () => {
  const mem = newDb();
  const { Pool } = mem.adapters.createPg();
  const q = new Pool();

  await initSchema(q);
  assert(true, "schema created");

  const passwordHash = await bcrypt.hash("secret123", 10);
  const row = await createUser(q, {
    email: "Aarav@Example.com",
    passwordHash,
    name: "Aarav",
    board: "JEE",
    grade: "12th",
    language: "Hinglish",
    preferredAnalogy: "Cricket",
    examGoals: "Crack JEE Advanced",
    confidenceLevel: 3,
    chapters: DEFAULT_CHAPTERS,
  });
  assert(row.email === "aarav@example.com", "email is lowercased on insert");
  assert(row.chapters.length === DEFAULT_CHAPTERS.length, "chapters seeded as JSONB");

  const byEmail = await getUserByEmail(q, "AARAV@example.com");
  assert(byEmail && (await bcrypt.compare("secret123", byEmail.password_hash)), "login: password verifies");
  assert(!(await bcrypt.compare("wrongpass", byEmail.password_hash)), "login: wrong password rejected");

  const token = jwt.sign({ userId: row.id, email: row.email }, "testsecret", { expiresIn: "30d" });
  const decoded = jwt.verify(token, "testsecret") as any;
  assert(decoded.userId === row.id, "JWT signs + verifies");

  const updated = await updateUser(q, row.id, {
    name: "Aarav K",
    board: "NEET",
    grade: "Dropper",
    language: "English",
    preferredAnalogy: "Trains",
    examGoals: "AIIMS",
    confidenceLevel: 4,
    chapters: [{ id: "x", name: "Optics", mastery: "weak", confidenceScore: 20, lastStudied: "2026-06-27" }],
  });
  assert(updated.board === "NEET" && updated.chapters.length === 1, "profile + chapters update");

  const profileShape = rowToUser(updated);
  assert(profileShape.profile.preferredAnalogy === "Trains", "rowToUser maps camelCase profile");

  await addMessage(q, row.id, { id: "m1", role: "user", text: "Explain inertia", mode: "standard", sources: [] });
  await addMessage(q, row.id, { id: "m1", role: "user", text: "dup", sources: [] }); // ON CONFLICT DO NOTHING
  const msgs = await getMessages(q, row.id, 50);
  assert(msgs.length === 1, "messages append + dedupe by id");

  await cacheSet(q, "k1", { text: "cached answer", sources: [{ title: "T", uri: "u" }] });
  await cacheSet(q, "k1", { text: "updated answer", sources: [] }); // upsert
  const cached = await cacheGet(q, "k1");
  assert(cached && cached.text === "updated answer", "explanation cache upsert + read");

  console.log("\nDB SMOKE TEST PASSED ✓");
  process.exit(0);
})().catch((e) => {
  console.error("\nDB SMOKE TEST FAILED ✗\n", e);
  process.exit(1);
});
