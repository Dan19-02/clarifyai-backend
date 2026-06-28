/**
 * Retrieval (RAG) + embeddings + topic gating + deep-verify.
 *
 * - embed(): Gemini embeddings with taskType=SEMANTIC_SIMILARITY (better
 *   calibrated for paraphrase/similarity than the default), with graceful
 *   fallback. The same taskType is used for the corpus, the query, and the
 *   semantic cache so all vectors are comparable.
 * - Multi-board seed corpus (CBSE / ICSE-ISC / JEE / NEET / General). Retrieval
 *   prefers the student's board (and universal "General" facts).
 * - topicTokens()/sharesTopic(): gate the semantic cache so it never reuses an
 *   answer across unrelated topics (e.g. osmosis vs diffusion).
 * - verifyAnswer(): optional deep-verify examiner pass.
 */
import { ai, apiKey } from "./gemini.js";
import { pool, knowledgeCount, knowledgeInsert, knowledgeAll } from "./db.js";

const EMBED_MODELS = ["text-embedding-004", "gemini-embedding-001"];
let chosenEmbedModel: string | null = null;

export async function embed(text: string, taskType = "SEMANTIC_SIMILARITY"): Promise<number[] | null> {
  if (!apiKey || !text?.trim()) return null;
  const models = chosenEmbedModel ? [chosenEmbedModel] : EMBED_MODELS;
  for (const model of models) {
    // Try with taskType first; fall back to no-config if the model rejects it.
    for (const cfg of [{ taskType }, undefined] as const) {
      try {
        const r: any = await ai.models.embedContent({ model, contents: text, ...(cfg ? { config: cfg } : {}) });
        const values: number[] | undefined = r?.embeddings?.[0]?.values || r?.embedding?.values;
        if (Array.isArray(values) && values.length) {
          chosenEmbedModel = model;
          return values;
        }
      } catch {
        /* try next config / model */
      }
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

const STOPWORDS = new Set(
  "a an the is are was were be of for to from in into on at with and or but what whats which who whom how why when where do does did explain define describe tell me about please give show find calculate solve simple simply detail short brief words word can you i my mean meaning concept topic kya hai hota hain ka ki ke ko ek mujhe batao samjhao".split(/\s+/)
);

/** Significant content tokens of a question (used to topic-gate the cache). */
export function topicTokens(message: string): Set<string> {
  return new Set(
    (message || "")
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length >= 3 && !STOPWORDS.has(w))
  );
}

/**
 * Whether two questions are about the SAME thing for cache-reuse purposes.
 * Rule: one question's content tokens must be a subset of the other's. This
 * allows paraphrases that only add filler ("process of photosynthesis" ⊇
 * "photosynthesis") but blocks pairs that differ by a meaningful token —
 * "newton FIRST law" vs "newton SECOND law", "KINETIC energy" vs "POTENTIAL
 * energy" — which would otherwise be wrongly reused.
 */
export function topicCompatible(a: Set<string>, b: Set<string>): boolean {
  if (a.size === 0 || b.size === 0) return a.size === b.size;
  const [small, big] = a.size <= b.size ? [a, b] : [b, a];
  for (const t of small) if (!big.has(t)) return false;
  return true;
}

// Board-aware seed corpus. "General" facts are served to every board; the rest
// carry board-appropriate framing. Expand this (or ingest real per-board
// textbook content + pgvector) for production coverage.
const SEED_CORPUS = [
  { id: "k-newton2", subject: "Physics", topic: "Newton's Second Law", board: "General", grade: "11",
    content: "Newton's Second Law: net force equals rate of change of momentum; for constant mass F = m·a. Force in newtons (N). The same force gives a lighter body a larger acceleration." },
  { id: "k-newton1", subject: "Physics", topic: "Newton's First Law (Inertia)", board: "General", grade: "11",
    content: "Newton's First Law (inertia): a body stays at rest or in uniform straight-line motion unless acted on by a net external force. Inertia increases with mass." },
  { id: "k-kinematics", subject: "Physics", topic: "Equations of Motion", board: "General", grade: "11",
    content: "For constant acceleration: v = u + a·t; s = u·t + ½·a·t²; v² = u² + 2·a·s. For a body thrown up, at the top v = 0 and a = -g (g ≈ 9.8 m/s², often 10 in problems)." },
  { id: "k-ohm", subject: "Physics", topic: "Ohm's Law", board: "General", grade: "10",
    content: "Ohm's Law: at constant temperature, V = I·R, current proportional to voltage; R in ohms (Ω). Does not hold for non-ohmic devices like diodes." },
  { id: "k-photosynthesis", subject: "Biology", topic: "Photosynthesis", board: "General", grade: "11",
    content: "Photosynthesis: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂, in chloroplasts. Light reactions in thylakoids make ATP/NADPH; the Calvin cycle fixes carbon in the stroma." },
  { id: "k-mitosis", subject: "Biology", topic: "Mitosis vs Meiosis", board: "General", grade: "11",
    content: "Mitosis → two identical diploid cells (growth/repair). Meiosis → four varied haploid gametes via two divisions; crossing over in prophase I creates variation." },
  { id: "k-quadratic", subject: "Mathematics", topic: "Quadratic Equations", board: "General", grade: "10",
    content: "ax² + bx + c = 0 (a≠0): roots x = (-b ± √(b²-4ac))/(2a). Discriminant D = b²-4ac: D>0 two distinct real, D=0 equal, D<0 no real roots. Sum = -b/a, product = c/a." },
  { id: "k-bonding", subject: "Chemistry", topic: "Chemical Bonding & Valency", board: "General", grade: "11",
    content: "Atoms bond to attain a stable (octet) configuration. Ionic bonds transfer electrons (metal+non-metal); covalent bonds share (non-metals). Valency = combining capacity." },
  { id: "k-mole", subject: "Chemistry", topic: "Mole Concept", board: "General", grade: "11",
    content: "One mole = 6.022×10²³ particles (Avogadro's number). Moles = mass / molar mass. Molar volume of an ideal gas at STP ≈ 22.4 L." },
  { id: "k-trig", subject: "Mathematics", topic: "Trigonometric Ratios", board: "General", grade: "10",
    content: "In a right triangle: sin θ = opposite/hypotenuse, cos θ = adjacent/hypotenuse, tan θ = opposite/adjacent. Identity: sin²θ + cos²θ = 1." },
  // Board / exam-specific framing (accurate study guidance, not invented facts)
  { id: "k-cbse", subject: "Study Guidance", topic: "CBSE approach", board: "CBSE", grade: "11",
    content: "CBSE exams are closely based on NCERT textbooks. Answers should be clear and to the point, with stepwise marking; practise NCERT examples and back-exercise questions and previous years' papers." },
  { id: "k-icse", subject: "Study Guidance", topic: "ICSE/ISC approach", board: "ICSE", grade: "10",
    content: "ICSE (Class 10) and ISC (Class 11-12, CISCE board) reward detailed, well-explained answers and full derivations/working. Definitions should be precise and answers more descriptive than CBSE." },
  { id: "k-jee", subject: "Study Guidance", topic: "JEE approach", board: "JEE", grade: "12",
    content: "JEE rewards application and multi-concept problem solving (e.g. kinematics with calculus, vectors). Build speed and accuracy on numericals; understand derivations, don't just memorise formulae." },
  { id: "k-neet", subject: "Study Guidance", topic: "NEET approach", board: "NEET", grade: "12",
    content: "NEET Biology is fact-dense and high-yield (Botany + Zoology), tightly NCERT-based. Focus on diagrams, classifications, examples and exceptions; Physics/Chemistry need accurate formula application." },
];

/** Embed and store the seed corpus once (idempotent). */
export async function ingestKnowledge(): Promise<void> {
  try {
    const existing = await knowledgeCount(pool);
    if (existing > 0) {
      console.log(`[RAG] Knowledge base ready (${existing} chunks, multi-board).`);
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
    console.log(`[RAG] Ingested ${ok}/${SEED_CORPUS.length} knowledge chunks (CBSE/ICSE/JEE/NEET/General)${ok ? "" : " (embeddings unavailable)"}.`);
  } catch (e: any) {
    console.warn("[RAG] Ingestion skipped:", e.message);
  }
}

/**
 * Top relevant notes for a question (given its embedding), preferring the
 * student's board and universal "General" facts. Returns null if nothing fits.
 */
export async function retrieveContext(
  queryEmbedding: number[] | null,
  board = "",
  k = 2,
  threshold = 0.6
): Promise<string | null> {
  if (!queryEmbedding) return null;
  let rows;
  try {
    rows = await knowledgeAll(pool);
  } catch {
    return null;
  }
  const b = (board || "").toLowerCase();
  const scored = rows
    .filter((r) => Array.isArray(r.embedding))
    .map((r) => {
      const cb = (r.board || "").toLowerCase();
      const boardMatch = cb === "general" || (b && cb === b);
      return { r, s: cosine(queryEmbedding, r.embedding) + (boardMatch ? 0.05 : -0.05) };
    })
    .filter((x) => x.s >= threshold)
    .sort((a, b2) => b2.s - a.s)
    .slice(0, k);
  if (!scored.length) return null;
  return scored
    .map((x) => `• [${x.r.subject} — ${x.r.topic}${x.r.board && x.r.board !== "General" ? " · " + x.r.board : ""}] ${x.r.content}`)
    .join("\n\n");
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
