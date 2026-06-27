/**
 * Retrieval (RAG) + embeddings + deep-verify.
 *
 * - embed(): Gemini text embeddings (with model fallback). Returns null on any
 *   failure so every caller degrades gracefully (RAG/semantic-cache just turn off).
 * - A small NCERT-aligned seed corpus is embedded into knowledge_chunks on boot;
 *   expand it later (or swap to pgvector + real textbook ingestion at scale).
 * - retrieveContext(): nearest concept notes for a question, injected into the prompt.
 * - verifyAnswer(): optional second-pass examiner that corrects factual/math errors.
 */
import { ai, apiKey } from "./gemini.js";
import { pool, knowledgeCount, knowledgeInsert, knowledgeAll } from "./db.js";

const EMBED_MODELS = ["text-embedding-004", "gemini-embedding-001"];
let chosenEmbedModel: string | null = null;

export async function embed(text: string): Promise<number[] | null> {
  if (!apiKey || !text?.trim()) return null;
  const models = chosenEmbedModel ? [chosenEmbedModel] : EMBED_MODELS;
  for (const model of models) {
    try {
      const r: any = await ai.models.embedContent({ model, contents: text });
      const values: number[] | undefined = r?.embeddings?.[0]?.values || r?.embedding?.values;
      if (Array.isArray(values) && values.length) {
        chosenEmbedModel = model;
        return values;
      }
    } catch {
      /* try next model */
    }
  }
  return null;
}

export function cosine(a: number[], b: number[]): number {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

// Small, factually-checked seed corpus (expand or replace with real ingestion).
const SEED_CORPUS = [
  { id: "k-newton2", subject: "Physics", topic: "Newton's Second Law", board: "CBSE", grade: "11",
    content: "Newton's Second Law: the net force on a body equals the rate of change of its momentum; for constant mass, F = m·a. Force is in newtons (N), mass in kg, acceleration in m/s². It explains why the same force gives a lighter object a larger acceleration." },
  { id: "k-newton1", subject: "Physics", topic: "Newton's First Law (Inertia)", board: "CBSE", grade: "11",
    content: "Newton's First Law (law of inertia): a body stays at rest or in uniform straight-line motion unless acted on by a net external force. Inertia is the tendency to resist changes in state of motion and increases with mass." },
  { id: "k-kinematics", subject: "Physics", topic: "Equations of Motion", board: "CBSE", grade: "11",
    content: "For constant acceleration: v = u + a·t; s = u·t + ½·a·t²; v² = u² + 2·a·s. For a body thrown up, at the highest point v = 0 and a = -g (g ≈ 9.8 m/s², often taken as 10 m/s² in problems)." },
  { id: "k-photosynthesis", subject: "Biology", topic: "Photosynthesis", board: "CBSE", grade: "11",
    content: "Photosynthesis: plants convert CO₂ and water into glucose and oxygen using light energy in chloroplasts. Overall: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂. The light reactions occur in the thylakoid membranes and produce ATP and NADPH; the Calvin cycle (dark reactions) fixes carbon in the stroma." },
  { id: "k-mitosis", subject: "Biology", topic: "Mitosis vs Meiosis", board: "NEET", grade: "11",
    content: "Mitosis produces two genetically identical diploid daughter cells (growth, repair). Meiosis produces four genetically varied haploid cells (gametes) via two divisions, with crossing over in prophase I creating variation." },
  { id: "k-quadratic", subject: "Mathematics", topic: "Quadratic Equations", board: "CBSE", grade: "10",
    content: "A quadratic ax² + bx + c = 0 (a ≠ 0) has roots x = (-b ± √(b²-4ac)) / (2a). The discriminant D = b²-4ac decides nature of roots: D>0 two real distinct, D=0 two real equal, D<0 no real roots. Sum of roots = -b/a, product = c/a." },
  { id: "k-bonding", subject: "Chemistry", topic: "Chemical Bonding & Valency", board: "CBSE", grade: "11",
    content: "Atoms bond to attain a stable (usually octet) electron configuration. Ionic bonds form by transfer of electrons (metal + non-metal); covalent bonds by sharing (non-metals). Valency is the combining capacity, related to electrons needed to complete the outer shell." },
  { id: "k-ohm", subject: "Physics", topic: "Ohm's Law", board: "CBSE", grade: "10",
    content: "Ohm's Law: at constant temperature, current through a conductor is directly proportional to the voltage across it, V = I·R, where R is resistance in ohms (Ω). It does not hold for non-ohmic devices like diodes." },
];

/** Embed and store the seed corpus once (idempotent). */
export async function ingestKnowledge(): Promise<void> {
  try {
    const existing = await knowledgeCount(pool);
    if (existing > 0) {
      console.log(`[RAG] Knowledge base ready (${existing} chunks).`);
      return;
    }
    if (!apiKey) {
      console.warn("[RAG] No GEMINI_API_KEY — skipping knowledge ingestion (RAG disabled).");
      return;
    }
    let ok = 0;
    for (const c of SEED_CORPUS) {
      const e = await embed(c.content);
      if (!e) continue;
      await knowledgeInsert(pool, { ...c, embedding: e });
      ok++;
    }
    console.log(`[RAG] Ingested ${ok}/${SEED_CORPUS.length} knowledge chunks${ok ? "" : " (embeddings unavailable)"}.`);
  } catch (e: any) {
    console.warn("[RAG] Ingestion skipped:", e.message);
  }
}

/** Top relevant notes for a question (given its embedding). Returns null if none. */
export async function retrieveContext(
  queryEmbedding: number[] | null,
  k = 2,
  threshold = 0.72
): Promise<string | null> {
  if (!queryEmbedding) return null;
  let rows;
  try {
    rows = await knowledgeAll(pool);
  } catch {
    return null;
  }
  const scored = rows
    .filter((r) => Array.isArray(r.embedding))
    .map((r) => ({ r, s: cosine(queryEmbedding, r.embedding) }))
    .filter((x) => x.s >= threshold)
    .sort((a, b) => b.s - a.s)
    .slice(0, k);
  if (!scored.length) return null;
  return scored.map((x) => `• [${x.r.subject} — ${x.r.topic}] ${x.r.content}`).join("\n\n");
}

/** Optional deep-verify: a meticulous examiner pass that corrects errors. */
export async function verifyAnswer(question: string, answer: string): Promise<string> {
  if (!apiKey) return answer;
  try {
    const r = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: [
        {
          role: "user",
          parts: [
            {
              text:
                `Student's question:\n${question}\n\nDraft answer to review:\n${answer}\n\n` +
                `Carefully check it for any factual error, wrong formula, or calculation mistake. ` +
                `If you find any, return a corrected version in the SAME format and tone. ` +
                `If it is fully correct, return it unchanged. Output ONLY the final answer.`,
            },
          ],
        },
      ],
      config: {
        temperature: 0.1,
        systemInstruction:
          "You are a meticulous subject examiner for Indian board/JEE/NEET material. You fix factual and mathematical errors in tutoring answers while preserving the warm tone and any section/notebook formatting. Output only the (corrected) answer.",
      },
    });
    return r.text || answer;
  } catch {
    return answer;
  }
}
