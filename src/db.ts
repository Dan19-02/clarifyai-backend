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

CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  role TEXT NOT NULL,
  text TEXT NOT NULL,
  mode TEXT,
  sources JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, created_at);

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
  role: string;
  text: string;
  mode?: string;
  sources?: { title: string; uri: string }[];
}

export async function getMessages(q: Queryable, userId: number, limit = 200) {
  const { rows } = await q.query(
    `SELECT id, role, text, mode, sources, created_at
     FROM messages WHERE user_id = $1 ORDER BY created_at ASC LIMIT $2`,
    [userId, limit]
  );
  return rows.map((r) => ({
    id: r.id,
    role: r.role,
    text: r.text,
    mode: r.mode || undefined,
    sources: r.sources || [],
    timestamp: new Date(r.created_at).toLocaleTimeString(),
  }));
}

export async function addMessage(q: Queryable, userId: number, m: StoredMessage) {
  await q.query(
    `INSERT INTO messages (id, user_id, role, text, mode, sources)
     VALUES ($1,$2,$3,$4,$5,$6)
     ON CONFLICT (id) DO NOTHING`,
    [m.id, userId, m.role, m.text, m.mode || null, JSON.stringify(m.sources || [])]
  );
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
): Promise<{ embedding: number[] | null; text: string; sources: any[] }[]> {
  const { rows } = await q.query(
    `SELECT embedding, text, sources FROM explanation_cache
     WHERE mode = $1 AND board = $2 AND grade = $3 AND language = $4 AND preferred_analogy = $5
       AND embedding IS NOT NULL
     LIMIT 300`,
    [f.mode, f.board ?? "", f.grade ?? "", f.language ?? "", f.preferredAnalogy ?? ""]
  );
  return rows.map((r) => ({ embedding: r.embedding, text: r.text, sources: r.sources || [] }));
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
): Promise<{ subject: string; topic: string; content: string; embedding: number[] }[]> {
  const { rows } = await q.query(`SELECT subject, topic, content, embedding FROM knowledge_chunks`);
  return rows.map((r) => ({ subject: r.subject, topic: r.topic, content: r.content, embedding: r.embedding }));
}
