/**
 * DB-layer smoke test using an in-memory Postgres (pg-mem). No real database
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
  createConversation,
  listConversations,
  deleteConversation,
  cacheGetByKey,
  cacheUpsertFull,
  knowledgeInsert,
  knowledgeCount,
  knowledgeAll,
  rowToUser,
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
  // Seed a non-empty literal (DEFAULT_CHAPTERS is intentionally empty now) so the
  // JSONB chapters round-trip is still genuinely exercised.
  const seedChapters = [
    { id: "sc1", name: "Test Chapter A", mastery: "weak", confidenceScore: 20, lastStudied: "2026-06-29" },
    { id: "sc2", name: "Test Chapter B", mastery: "strong", confidenceScore: 90, lastStudied: "2026-06-29" },
  ];
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
    chapters: seedChapters,
  });
  assert(row.email === "aarav@example.com", "email is lowercased on insert");
  assert(
    row.chapters.length === 2 && row.chapters[0].name === "Test Chapter A" && row.chapters[1].confidenceScore === 90,
    "chapters round-trip through the JSONB column"
  );

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

  const conv = await createConversation(q, row.id, "conv-1", "Physics doubts");
  assert(conv.id === "conv-1" && conv.title === "Physics doubts", "conversation create");

  await addMessage(q, row.id, { id: "m1", conversationId: conv.id, role: "user", text: "Explain inertia", mode: "standard", sources: [] });
  await addMessage(q, row.id, { id: "m1", conversationId: conv.id, role: "user", text: "dup", sources: [] }); // ON CONFLICT DO NOTHING
  const msgs = await getMessages(q, row.id, conv.id, 50);
  assert(msgs.length === 1, "messages append + dedupe by id (scoped to conversation)");

  const convList = await listConversations(q, row.id);
  assert(convList.length === 1 && convList[0].messageCount === 1, "conversation list reports message count");

  await deleteConversation(q, row.id, conv.id);
  assert((await listConversations(q, row.id)).length === 0, "conversation delete cascades messages");
  assert((await getMessages(q, row.id, conv.id, 50)).length === 0, "messages gone after conversation delete");

  const facets = { mode: "standard", board: "JEE", grade: "12th", language: "English", preferredAnalogy: "Daily Life" };
  await cacheUpsertFull(q, { cacheKey: "k1", ...facets, question: "what is inertia", embedding: [0.1, 0.2, 0.3], text: "cached answer", sources: [{ title: "T", uri: "u" }] });
  await cacheUpsertFull(q, { cacheKey: "k1", ...facets, question: "what is inertia", embedding: [0.1, 0.2, 0.3], text: "updated answer", sources: [] }); // upsert
  const cached = await cacheGetByKey(q, "k1");
  assert(cached && cached.text === "updated answer", "explanation cache upsert + read by key");

  await knowledgeInsert(q, { id: "kc1", subject: "Physics", topic: "Inertia", board: "CBSE", grade: "11", content: "Inertia note", embedding: [0.1, 0.2, 0.3] });
  assert((await knowledgeCount(q)) === 1, "knowledge insert + count");
  const chunks = await knowledgeAll(q);
  assert(chunks.length === 1 && Array.isArray(chunks[0].embedding), "knowledge embedding round-trips as a JS array");

  console.log("\nDB SMOKE TEST PASSED ✓");
  process.exit(0);
})().catch((e) => {
  console.error("\nDB SMOKE TEST FAILED ✗\n", e);
  process.exit(1);
});
