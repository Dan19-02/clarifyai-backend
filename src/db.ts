/**
 * PostgreSQL data layer.
 *
 * Every query function takes a `Queryable` (defaults to the shared pool) so the
 * exact same code can run against the real database in production and against an
 * in-memory Postgres (pg-mem) in tests.
 */
import pg from "pg";

const { Pool } = pg;

/** Anything with a node-postgres style `.query()` (real pool or pg-mem). */
export interface Queryable {
  query: (text: string, params?: any[]) => Promise<{ rows: any[] }>;
}

/**
 * The active database pool. Set by initDb() at startup — either a real
 * PostgreSQL pool, or an in-memory pg-mem pool as a zero-setup dev fallback.
 * Exported as a live binding so route modules always see the current pool.
 */
export let pool: any = null;

/** Connect to Postgres if DATABASE_URL works; otherwise fall back to in-memory. */
export async function initDb(): Promise<void> {
  const url = process.env.DATABASE_URL;
  if (url) {
    const isLocal = /localhost|127\.0\.0\.1/.test(url);
    const candidate = new Pool({
      connectionString: url,
      ssl: process.env.PGSSL === "false" || isLocal ? false : { rejectUnauthorized: false },
    });
    try {
      await candidate.query("SELECT 1");
      pool = candidate;
      await initSchema(pool);
      console.log("[DB] Connected to PostgreSQL.");
      return;
    } catch (e: any) {
      console.warn(`[DB] Could not reach Postgres (${e.message}). Falling back to in-memory dev DB.`);
      try {
        await candidate.end();
      } catch {
        /* ignore */
      }
    }
  } else {
    console.warn("[DB] DATABASE_URL not set.");
  }

  // Zero-setup in-memory fallback (data resets on restart).
  const { newDb } = await import("pg-mem");
  const PgMemPool = newDb().adapters.createPg().Pool;
  pool = new PgMemPool();
  await initSchema(pool);
  console.warn("[DB] ⚠ Using IN-MEMORY database — data resets on restart. Set a real DATABASE_URL to persist.");
}

export const DEFAULT_CHAPTERS = [
  { id: "ch-1", name: "Photosynthesis & Light Reactions", mastery: "developing", confidenceScore: 65, lastStudied: "2026-06-26" },
  { id: "ch-2", name: "Newton's Second Law of Motion", mastery: "weak", confidenceScore: 35, lastStudied: "2026-06-25" },
  { id: "ch-3", name: "Quadratic Equations & Roots", mastery: "strong", confidenceScore: 90, lastStudied: "2026-06-24" },
  { id: "ch-4", name: "Chemical Bonding & Valency", mastery: "developing", confidenceScore: 50, lastStudied: "2026-06-23" },
];

export const SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS users (
  id SERIAL PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  name TEXT NOT NULL DEFAULT 'Student',
  board TEXT NOT NULL DEFAULT 'CBSE',
  grade TEXT NOT NULL DEFAULT '11th Grade',
  language TEXT NOT NULL DEFAULT 'Hinglish',
  preferred_analogy TEXT NOT NULL DEFAULT 'Daily Life',
  exam_goals TEXT NOT NULL DEFAULT '',
  confidence_level INTEGER NOT NULL DEFAULT 3,
  chapters JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title TEXT NOT NULL DEFAULT 'New chat',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at);

CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  conversation_id TEXT,
  role TEXT NOT NULL,
  text TEXT NOT NULL,
  mode TEXT,
  sources JSONB NOT NULL DEFAULT '[]'::jsonb,
  attachments JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS explanation_cache (
  cache_key TEXT PRIMARY KEY,
  mode TEXT,
  board TEXT,
  grade TEXT,
  language TEXT,
  preferred_analogy TEXT,
  question TEXT,
  embedding JSONB,
  text TEXT NOT NULL,
  sources JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
  id TEXT PRIMARY KEY,
  subject TEXT,
  topic TEXT,
  board TEXT,
  grade TEXT,
  content TEXT NOT NULL,
  embedding JSONB NOT NULL
);
`;

export async function initSchema(q: Queryable = pool): Promise<void> {
  await q.query(SCHEMA_SQL);
  // Defensive migrations for databases created before conversations existed.
  for (const sql of [
    `ALTER TABLE messages ADD COLUMN IF NOT EXISTS conversation_id TEXT`,
    `ALTER TABLE messages ADD COLUMN IF NOT EXISTS attachments JSONB NOT NULL DEFAULT '[]'::jsonb`,
  ]) {
    try {
      await q.query(sql);
    } catch {
      /* column already exists / engine lacks IF NOT EXISTS — safe to ignore */
    }
  }
}

/** Shape returned to the client (camelCase profile + chapters). */
export function rowToUser(row: any) {
  return {
    id: row.id,
    email: row.email,
    profile: {
      name: row.name,
      board: row.board,
      grade: row.grade,
      language: row.language,
      preferredAnalogy: row.preferred_analogy,
      examGoals: row.exam_goals,
      confidenceLevel: row.confidence_level,
    },
    chapters: row.chapters || [],
  };
}

export interface NewUser {
  email: string;
  passwordHash: string;
  name: string;
  board: string;
  grade: string;
  language: string;
  preferredAnalogy: string;
  examGoals: string;
  confidenceLevel: number;
  chapters: any[];
}

export async function createUser(q: Queryable, u: NewUser) {
  const { rows } = await q.query(
    `INSERT INTO users (email, password_hash, name, board, grade, language, preferred_analogy, exam_goals, confidence_level, chapters)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
     RETURNING *`,
    [
      u.email.toLowerCase(),
      u.passwordHash,
      u.name,
      u.board,
      u.grade,
      u.language,
      u.preferredAnalogy,
      u.examGoals,
      u.confidenceLevel,
      JSON.stringify(u.chapters),
    ]
  );
  return rows[0];
}

export async function getUserByEmail(q: Queryable, email: string) {
  const { rows } = await q.query(`SELECT * FROM users WHERE email = $1`, [email.toLowerCase()]);
  return rows[0] || null;
}

export async function getUserById(q: Queryable, id: number) {
  const { rows } = await q.query(`SELECT * FROM users WHERE id = $1`, [id]);
  return rows[0] || null;
}

export interface ProfileUpdate {
  name: string;
  board: string;
  grade: string;
  language: string;
  preferredAnalogy: string;
  examGoals: string;
  confidenceLevel: number;
  chapters: any[];
}

export async function updateUser(q: Queryable, id: number, p: ProfileUpdate) {
  const { rows } = await q.query(
    `UPDATE users SET
       name=$2, board=$3, grade=$4, language=$5, preferred_analogy=$6,
       exam_goals=$7, confidence_level=$8, chapters=$9, updated_at=now()
     WHERE id=$1
     RETURNING *`,
    [
      id,
      p.name,
      p.board,
      p.grade,
      p.language,
      p.preferredAnalogy,
      p.examGoals,
      p.confidenceLevel,
      JSON.stringify(p.chapters),
    ]
  );
  return rows[0] || null;
}

export interface StoredMessage {
  id: string;
  conversationId: string;
  role: string;
  text: string;
  mode?: string;
  sources?: { title: string; uri: string }[];
  attachments?: any[];
}

export async function getMessages(q: Queryable, userId: number, conversationId: string, limit = 200) {
  const { rows } = await q.query(
    `SELECT id, role, text, mode, sources, attachments, created_at
     FROM messages WHERE user_id = $1 AND conversation_id = $2 ORDER BY created_at ASC LIMIT $3`,
    [userId, conversationId, limit]
  );
  return rows.map((r) => ({
    id: r.id,
    role: r.role,
    text: r.text,
    mode: r.mode || undefined,
    sources: r.sources || [],
    attachments: r.attachments || [],
    timestamp: new Date(r.created_at).toLocaleTimeString(),
  }));
}

export async function addMessage(q: Queryable, userId: number, m: StoredMessage) {
  await q.query(
    `INSERT INTO messages (id, user_id, conversation_id, role, text, mode, sources, attachments)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
     ON CONFLICT (id) DO NOTHING`,
    [m.id, userId, m.conversationId, m.role, m.text, m.mode || null, JSON.stringify(m.sources || []), JSON.stringify(m.attachments || [])]
  );
  // Bump the conversation so the most recently used one floats to the top.
  await q.query(`UPDATE conversations SET updated_at = now() WHERE id = $1 AND user_id = $2`, [m.conversationId, userId]);
}

// ---- Conversations (separate chat windows) ----
export async function listConversations(q: Queryable, userId: number) {
  const { rows } = await q.query(
    `SELECT c.id, c.title, c.created_at, c.updated_at, count(m.id) AS message_count
     FROM conversations c
     LEFT JOIN messages m ON m.conversation_id = c.id
     WHERE c.user_id = $1
     GROUP BY c.id, c.title, c.created_at, c.updated_at
     ORDER BY c.updated_at DESC`,
    [userId]
  );
  return rows.map((r) => ({
    id: r.id,
    title: r.title,
    messageCount: Number(r.message_count || 0),
    updatedAt: new Date(r.updated_at).toISOString(),
  }));
}

export async function createConversation(q: Queryable, userId: number, id: string, title = "New chat") {
  const { rows } = await q.query(
    `INSERT INTO conversations (id, user_id, title) VALUES ($1,$2,$3) RETURNING id, title, created_at, updated_at`,
    [id, userId, title]
  );
  const r = rows[0];
  return { id: r.id, title: r.title, messageCount: 0, updatedAt: new Date(r.updated_at).toISOString() };
}

export async function renameConversation(q: Queryable, userId: number, id: string, title: string) {
  await q.query(`UPDATE conversations SET title = $3 WHERE id = $1 AND user_id = $2`, [id, userId, title]);
}

export async function deleteConversation(q: Queryable, userId: number, id: string) {
  // Manual cascade (messages.conversation_id has no FK so it works on every engine).
  await q.query(`DELETE FROM messages WHERE user_id = $1 AND conversation_id = $2`, [userId, id]);
  await q.query(`DELETE FROM conversations WHERE id = $2 AND user_id = $1`, [userId, id]);
}

export async function conversationOwnedBy(q: Queryable, userId: number, id: string): Promise<boolean> {
  const { rows } = await q.query(`SELECT 1 FROM conversations WHERE id = $1 AND user_id = $2`, [id, userId]);
  return rows.length > 0;
}

/**
 * Make sure the user has at least one conversation, migrating any legacy
 * messages (saved before conversations existed) into a default one.
 */
export async function ensureDefaultConversation(q: Queryable, userId: number, newId: string): Promise<string> {
  const existing = await listConversations(q, userId);
  if (existing.length > 0) return existing[0].id;

  await createConversation(q, userId, newId, "My Study Log");
  // Adopt any orphaned messages from before this feature shipped.
  await q.query(`UPDATE messages SET conversation_id = $1 WHERE user_id = $2 AND conversation_id IS NULL`, [newId, userId]);
  return newId;
}

export interface CachedAnswer {
  text: string;
  sources: { title: string; uri: string }[];
}

export interface CacheFacets {
  mode: string;
  board?: string;
  grade?: string;
  language?: string;
  preferredAnalogy?: string;
}

/** Exact cache lookup by hashed key. */
export async function cacheGetByKey(q: Queryable, key: string): Promise<CachedAnswer | null> {
  const { rows } = await q.query(`SELECT text, sources FROM explanation_cache WHERE cache_key = $1`, [key]);
  if (!rows[0]) return null;
  return { text: rows[0].text, sources: rows[0].sources || [] };
}

/** Candidate cache rows (same facets) for semantic (embedding) matching. */
export async function cacheCandidates(
  q: Queryable,
  f: CacheFacets
): Promise<{ embedding: number[] | null; text: string; sources: any[]; question: string }[]> {
  const { rows } = await q.query(
    `SELECT embedding, text, sources, question FROM explanation_cache
     WHERE mode = $1 AND board = $2 AND grade = $3 AND language = $4 AND preferred_analogy = $5
       AND embedding IS NOT NULL
     LIMIT 300`,
    [f.mode, f.board ?? "", f.grade ?? "", f.language ?? "", f.preferredAnalogy ?? ""]
  );
  return rows.map((r) => ({ embedding: r.embedding, text: r.text, sources: r.sources || [], question: r.question || "" }));
}

export interface CacheUpsert extends CacheFacets {
  cacheKey: string;
  question: string;
  embedding: number[] | null;
  text: string;
  sources: any[];
}

export async function cacheUpsertFull(q: Queryable, r: CacheUpsert): Promise<void> {
  await q.query(
    `INSERT INTO explanation_cache (cache_key, mode, board, grade, language, preferred_analogy, question, embedding, text, sources)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
     ON CONFLICT (cache_key) DO UPDATE SET text = EXCLUDED.text, sources = EXCLUDED.sources, embedding = EXCLUDED.embedding`,
    [
      r.cacheKey,
      r.mode,
      r.board ?? "",
      r.grade ?? "",
      r.language ?? "",
      r.preferredAnalogy ?? "",
      r.question,
      r.embedding ? JSON.stringify(r.embedding) : null,
      r.text,
      JSON.stringify(r.sources || []),
    ]
  );
}

// ---- Knowledge base (RAG) ----
export interface KnowledgeChunk {
  id: string;
  subject: string;
  topic: string;
  board: string;
  grade: string;
  content: string;
  embedding: number[];
}

export async function knowledgeCount(q: Queryable): Promise<number> {
  const { rows } = await q.query(`SELECT count(*) AS n FROM knowledge_chunks`);
  return Number(rows[0]?.n || 0);
}

export async function knowledgeInsert(q: Queryable, c: KnowledgeChunk): Promise<void> {
  await q.query(
    `INSERT INTO knowledge_chunks (id, subject, topic, board, grade, content, embedding)
     VALUES ($1,$2,$3,$4,$5,$6,$7) ON CONFLICT (id) DO NOTHING`,
    [c.id, c.subject, c.topic, c.board, c.grade, c.content, JSON.stringify(c.embedding)]
  );
}

export async function knowledgeAll(
  q: Queryable
): Promise<{ subject: string; topic: string; board: string; content: string; embedding: number[] }[]> {
  const { rows } = await q.query(`SELECT subject, topic, board, content, embedding FROM knowledge_chunks`);
  return rows.map((r) => ({ subject: r.subject, topic: r.topic, board: r.board, content: r.content, embedding: r.embedding }));
}
