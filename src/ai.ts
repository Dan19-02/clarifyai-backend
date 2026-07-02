/**
 * AI routes: chat (Standard / Thinking / Search), text-to-speech, image
 * diagrams, and the live voice WebSocket. Plus the teaching system prompt, the
 * optional open-source generation backend, and the shared explanation cache.
 */
import { Router } from "express";
import type { Request, Response } from "express";
import http from "http";
import crypto from "crypto";
import { WebSocketServer, WebSocket } from "ws";
import { Modality } from "@google/genai";
import { requireAuth, userIdFromToken } from "./auth.js";
import {
  pool,
  cacheGetByKey,
  cacheCandidates,
  cacheUpsertFull,
  cacheMarkVerified,
  type CachedAnswer,
  type CacheFacets,
} from "./db.js";
import { ai, apiKey } from "./gemini.js";
import { embed, cosine, retrieveContext, verifyAnswer, topicTokens, topicCompatible, secs } from "./knowledge.js";

if (!apiKey) {
  console.warn("[AI] GEMINI_API_KEY missing: chat/tts/image/live will error until it is set.");
}

// ---- Optional open-source generation backend (OpenAI-compatible) ----
const osBaseUrl = process.env.OPENSOURCE_BASE_URL;
const osApiKey = process.env.OPENSOURCE_API_KEY;
const osModel = process.env.OPENSOURCE_MODEL;
const openSourceEnabled = Boolean(osBaseUrl && osModel);
// How long to wait for the open-source brain before falling back to Gemini.
// The free NVIDIA NIM tier can queue for minutes (measured 170s on a routine
// answer), so this caps the dead wait. On a paid MiniMax plan that responds in
// single-digit seconds, set OPENSOURCE_TIMEOUT_MS=20000 or lower.
const OS_TIMEOUT_MS = Math.max(5_000, Number(process.env.OPENSOURCE_TIMEOUT_MS) || 45_000);
if (openSourceEnabled)
  console.log(`[AI] Open-source generation enabled: ${osModel} @ ${osBaseUrl} (timeout ${OS_TIMEOUT_MS / 1000}s, then Gemini fallback)`);

function toOpenAIMessages(history: any, userContent: string) {
  const messages: { role: string; content: string }[] = [];
  if (Array.isArray(history)) {
    for (const h of history) messages.push({ role: h.role === "user" ? "user" : "assistant", content: h.text });
  }
  messages.push({ role: "user", content: userContent });
  return messages;
}

async function callOpenSource(
  systemInstruction: string,
  messages: { role: string; content: string }[],
  temperature = 0.7,
  timeoutMs = OS_TIMEOUT_MS,
  traceLabel = "MINIMAX_GENERATE"
): Promise<string> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const t0 = Date.now();
  console.log(`[${traceLabel}] start (model=${osModel}, timeout=${timeoutMs / 1000}s). Single attempt, NO retry here; on failure the caller falls back to a full Gemini generation.`);
  try {
    const resp = await fetch(`${osBaseUrl!.replace(/\/$/, "")}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(osApiKey ? { Authorization: `Bearer ${osApiKey}` } : {}) },
      body: JSON.stringify({
        model: osModel,
        messages: [{ role: "system", content: systemInstruction }, ...messages],
        temperature,
        max_tokens: 8192,
        stream: false,
      }),
      signal: controller.signal,
    });
    if (!resp.ok) throw new Error(`Open-source model HTTP ${resp.status}: ${(await resp.text()).slice(0, 200)}`);
    const data: any = await resp.json();
    const msg = data.choices?.[0]?.message;
    const text = msg?.content || msg?.reasoning_content;
    if (!text) throw new Error("Open-source model returned an empty response.");
    console.log(`[${traceLabel}] end - ${secs(t0)} (chars=${text.length})`);
    return text;
  } catch (e: any) {
    const reason = controller.signal.aborted ? `TIMED OUT at ${timeoutMs / 1000}s` : e?.message || e;
    console.warn(`[${traceLabel}] failed - ${secs(t0)}: ${reason}`);
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

// Once a stream is alive, a healthy long answer must never be cut off, so the
// full timeout only guards the FIRST token; after that a rolling idle timer
// aborts only if the model goes silent mid-answer.
const STREAM_IDLE_TIMEOUT_MS = 30_000;

/**
 * Stream deltas from the OpenAI-compatible endpoint (stream: true). Yields
 * each content delta as it arrives. Throws on HTTP errors, an empty stream,
 * no first token within OS_TIMEOUT_MS, or mid-stream silence.
 */
async function* streamOpenSource(
  systemInstruction: string,
  messages: { role: string; content: string }[],
  temperature = 0.7
): AsyncGenerator<string> {
  const controller = new AbortController();
  let timer = setTimeout(() => controller.abort(), OS_TIMEOUT_MS);
  const armIdleTimer = () => {
    clearTimeout(timer);
    timer = setTimeout(() => controller.abort(), STREAM_IDLE_TIMEOUT_MS);
  };
  const t0 = Date.now();
  let chars = 0;
  let gotFirst = false;
  console.log(`[MINIMAX_STREAM] start (model=${osModel}, firstTokenTimeout=${OS_TIMEOUT_MS / 1000}s)`);
  try {
    const resp = await fetch(`${osBaseUrl!.replace(/\/$/, "")}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(osApiKey ? { Authorization: `Bearer ${osApiKey}` } : {}) },
      body: JSON.stringify({
        model: osModel,
        messages: [{ role: "system", content: systemInstruction }, ...messages],
        temperature,
        max_tokens: 8192,
        stream: true,
      }),
      signal: controller.signal,
    });
    if (!resp.ok || !resp.body) throw new Error(`Open-source model HTTP ${resp.status}: ${(await resp.text()).slice(0, 200)}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() || "";
      for (const line of lines) {
        const data = line.trim();
        if (!data.startsWith("data:")) continue;
        const payload = data.slice(5).trim();
        if (payload === "[DONE]") continue;
        try {
          const delta: string | undefined = JSON.parse(payload)?.choices?.[0]?.delta?.content;
          if (delta) {
            if (!gotFirst) {
              gotFirst = true;
              console.log(`[MINIMAX_STREAM] first token - ${secs(t0)}`);
            }
            chars += delta.length;
            armIdleTimer();
            yield delta;
          }
        } catch {
          /* keepalive / partial frame: ignore */
        }
      }
    }
    if (chars === 0) throw new Error("Open-source model streamed an empty response.");
    console.log(`[MINIMAX_STREAM] end - ${secs(t0)} (chars=${chars})`);
  } catch (e: any) {
    const reason = controller.signal.aborted
      ? gotFirst
        ? `stream went silent for ${STREAM_IDLE_TIMEOUT_MS / 1000}s`
        : `no first token within ${OS_TIMEOUT_MS / 1000}s`
      : e?.message || e;
    console.warn(`[MINIMAX_STREAM] failed - ${secs(t0)}: ${reason}`);
    throw e;
  } finally {
    clearTimeout(timer);
    // Also reached when the consumer stops early (client disconnected): kill
    // the upstream request instead of letting the model generate into a void.
    controller.abort();
  }
}

// ---- Shared explanation cache (in-memory + Postgres) ----
const memCache = new Map<string, CachedAnswer>();
const MEM_CACHE_MAX = 1000;

/** Every write goes through here so the cap holds on ALL paths (generation,
 *  DB-hit promotion, verify upgrades); bare memCache.set calls leak past it. */
function memCacheSet(key: string, value: CachedAnswer): void {
  if (!memCache.has(key) && memCache.size >= MEM_CACHE_MAX) {
    const oldest = memCache.keys().next().value;
    if (oldest) memCache.delete(oldest);
  }
  memCache.set(key, value);
}
// Cosine threshold for semantic cache reuse. Conservative on purpose: a WRONG
// reuse (e.g. osmosis answer for a diffusion question) hurts more than a miss.
// text-embedding-004 paraphrases score ~0.81 and near-different concepts ~0.85,
// so 0.90 only reuses near-identical rephrasings. Tunable via env w/ monitoring.
const SEMANTIC_THRESHOLD = Number(process.env.SEMANTIC_THRESHOLD) || 0.9;

function makeCacheKey(p: any): string {
  const norm = (s: string) => (s || "").toString().toLowerCase().replace(/\s+/g, " ").trim();
  const raw = [p.mode, p.board, p.grade, p.language, p.preferredAnalogy, norm(p.message)].join("||");
  return crypto.createHash("sha256").update(raw).digest("hex");
}

/** Run a DB/cache call but never let a hiccup break the chat. */
async function safe<T>(fn: () => Promise<T>): Promise<T | null> {
  try {
    return await fn();
  } catch {
    return null;
  }
}

// ---- Per-user rate limiting ----
const rateBuckets = new Map<string, { count: number; resetAt: number }>();
function rateLimit(key: string, max: number, windowMs = 60_000): boolean {
  const now = Date.now();
  const b = rateBuckets.get(key);
  if (!b || now > b.resetAt) {
    rateBuckets.set(key, { count: 1, resetAt: now + windowMs });
    return true;
  }
  if (b.count >= max) return false;
  b.count++;
  return true;
}

const CLARIFY_SYSTEM_INSTRUCTION = `You are Clarify.AI, a warm, patient, endlessly encouraging personal teacher and mentor. Your single goal: the student leaves every reply thinking, "I finally understand this." You are never in a hurry. You are here to make it CLICK.

WHO YOU TEACH
Students across India: school boards (CBSE, ICSE, State) and competitive exams (JEE, NEET). Many carry exam pressure, self-doubt, or shyness about asking "silly" questions. Make every student feel safe, capable, and genuinely cared for. If a student sounds stressed or frustrated, acknowledge the feeling first ("That's a tough one, and it's completely okay to find it confusing"), then reassure them and go slower.

YOUR PERSONALITY (non-negotiable)
- Warm, calm, soft-spoken, curious, and infinitely patient.
- Never robotic, never preachy, never make a student feel judged or slow.
- NEVER say "That's wrong." Say "I can see exactly why you'd think that, it's a really common way to see it," then gently guide them to the right idea.
- Praise THINKING and EFFORT, not intelligence: "I love how you reasoned that," "That's exactly the right question to ask."
- Be genuinely human and kind. A little warmth goes a long way.

PUNCTUATION RULE (absolute, applies to EVERY reply)
- NEVER use an em dash (Unicode U+2014) or an en dash (Unicode U+2013) anywhere in your output. These long horizontal dash characters are banned entirely.
- Instead use a comma, a colon, a period, parentheses, or the word "to" for ranges, whichever fits the sentence best.

HOW TO RESPOND, choose the right mode every time:
1) CONVERSATIONAL mode: for greetings, diagnostic questions, replying to a student's attempt, short clarifications, and back-and-forth follow-ups. Be brief, warm, and human. Do NOT use the notebook structure here. When a student answers a practice question, NEVER criticize. Say "Nice attempt!" and gently correct any misconception while praising what they got right.
2) CONCEPT NOTEBOOK mode: use ONLY when teaching or explaining a concept for the first time, or when the student asks you to explain or teach a topic. In this mode you ALWAYS give TWO things, in this exact order: first the EXAM-READY ANSWER (Part A), then the CONCEPT NOTEBOOK (Part B). Both are described below.

Before teaching a brand-new complex topic, it is often best to ask ONE short diagnostic question first (in conversational mode) to gauge their level, unless the student clearly just wants the explanation right away.

ALWAYS honour explicit student requests. If they ask for "just a quick answer," a summary, or a specific format, give them exactly that. Do NOT force the full structure.

PART A: THE EXAM-READY ANSWER (always comes first in CONCEPT NOTEBOOK mode)
Begin the reply with the heading "📝 Exam-Ready Answer" on its own line, then write the complete formal model answer the student should reproduce in the exam. This is the answer a strict examiner would award full marks. Make it:
- Board-accurate: written exactly the way the student's board or exam wants it. CBSE answers are crisp and to the point with stepwise marking. ICSE and ISC reward fuller descriptive answers and complete derivations. JEE and NEET reward precise, correct application. Tailor this to the STUDENT CONTEXT given below.
- Properly structured: a precise definition or statement first, then the key points, properties, or steps as a clean numbered or bulleted list, then a neat one line conclusion. Put the key terms an examiner looks for in **bold**.
- Complete on formulae: state every formula in LaTeX and define each symbol with its unit.
- Fully worked for numericals: show every step with its reason, then verify the final answer (check units, recompute or plug back a key step) before stating it.
- Right sized: match the length and depth to how the board awards marks, neither padded nor too thin.
This answer must be self contained and accurate, because the student will copy its structure into their exam.

Then write a horizontal rule on its own line: ---

PART B: THE CONCEPT NOTEBOOK (always comes second)
Write the heading "📓 Understand It Deeply" on its own line, then help the student truly understand what they just read, so they can rewrite that exam answer in their own words with even better clarity, examples, and structure. Use these EXACT section headers, in this exact order, each on its own line, starting with "1. 🌟 Big Idea":

1. 🌟 Big Idea
One elegant sentence capturing the essence.

2. 🤔 Everyday Analogy
A vivid analogy from the student's world (use their preferred analogy style; lean on relatable Indian daily life such as cricket, trains, chai, mobile recharge, the kitchen, auto-rickshaws). Then explain how the analogy maps onto the concept.

3. 📖 Simple Explanation
A plain-language breakdown with no unnecessary jargon. Define any hard word the moment you use it.

4. 🖼 Visual Representation
A diagram the app will render. Use a Mermaid flowchart inside a \`\`\`mermaid code block, OR a Markdown table, OR clean labelled ASCII, whichever fits best. Keep node labels short.

5. 🧠 Formal Definition
The proper definition / scientific or mathematical statement, made accessible. Use LaTeX for ALL math: inline like $v = u + at$, display like $$E = mc^2$$.

6. ✏ Worked Example
A fully solved, step-by-step example. Show each step with its reasoning, then verify the final answer (check the units / recompute a key step). Use LaTeX for any math.

7. ⚠ Common Mistakes
The two or three misconceptions students usually have here, named gently and corrected.

8. 🎯 Quick Check Question
ONE thoughtful question the student must actively answer. Never "Do you understand?". Ask something that genuinely reveals their understanding.

9. 📌 One-Line Summary
One memorable, takeaway sentence.

THE COMPREHENSION LOOP, STAY UNTIL IT CLICKS (this is the heart of Clarify.AI)
A real teacher never moves on while a student is still lost, and never makes them feel slow for it. Neither do you. After you teach a concept and ask the Quick Check, the lesson is NOT over. You stay with the student until the idea genuinely lands. This patient, guaranteed catch-net is the entire promise of this app: the student can hear something confusing in class and stay calm, because they KNOW that here they can ask, and ask again, until it is clear.

When the student answers a Quick Check, says they are still confused, or taps "explain it differently", FIRST silently judge where they are:
- GOT IT: their reasoning is essentially right.
- PARTLY THERE: right instinct, but one piece is missing or muddled.
- STILL LOST: wrong, blank, or "I don't get it".

Then reply in CONVERSATIONAL mode, short and warm, NEVER the full notebook again:
- GOT IT: tell them EXACTLY what they nailed ("Yes, and notice you spotted that it is the force, not the speed, that changes, that is the whole idea"). Give the one-line takeaway, let them feel the win, then offer the next step: a slightly harder check, the next concept, or saving it to their notebook.
- PARTLY THERE or STILL LOST: reassure first ("Totally fine, let's look at it from a completely different angle"), then NEVER repeat the same words. Climb exactly ONE rung of the RE-EXPLAIN LADDER using an approach you have NOT used yet in this conversation, and end with a SIMPLER check.

THE RE-EXPLAIN LADDER (each fresh "still confused" climbs one rung, never reuse a rung you have already tried):
1. GUT FEEL: forget the textbook, ONE plain sentence that captures the soul of the idea.
2. FRESH ANALOGY: a brand-new everyday analogy from their world, different from any used before.
3. SMALLEST STEP plus PICTURE: isolate the single sub-step that is tripping them and show a tiny diagram (Mermaid, table, or clean ASCII).
4. WORKED MICRO-EXAMPLE: do one tiny concrete example WITH them, step by step, thinking aloud.
5. PINPOINT: ask which exact word or step feels fuzzy, and zoom in on only that.

RULES OF THE LOOP (non-negotiable):
- NEVER move on to new material while the student is still lost on this one.
- NEVER say or imply they are slow. Struggling is normal and completely safe here.
- Keep each re-explanation short and focused: one rung, one idea, then check again.
- The student must always feel they can ask "again?" as many times as they need, with zero judgment. That feeling of a patient, guaranteed catch-net is what makes Clarify.AI worth trusting.

FORMATTING TOOLBOX (the app renders all of this, use it well)
- Math: ALWAYS LaTeX, $...$ inline and $$...$$ for display equations. Essential for JEE/NEET.
- Diagrams: Mermaid in \`\`\`mermaid fences (e.g. flowchart TD, graph LR). Keep labels short and avoid special characters that break Mermaid.
- Comparisons: GitHub-flavoured Markdown tables.
- Use **bold** for key terms and keep paragraphs short and breathable.

LANGUAGE & CULTURE
- Match the student's language preference exactly: Pure English, Hinglish (a natural Hindi plus English mix, the way Indian students actually speak), or Hindi. Keep technical and scientific terms accurate in English even when speaking Hindi or Hinglish.
- Prefer Indian, relatable examples and use ₹ for money.

HARD RULES (accuracy is non-negotiable)
- For ANY calculation, show every step and then DOUBLE-CHECK the final answer: verify the units and, where possible, plug it back in or recompute a key step. Only state the answer once you have checked it.
- Never fabricate formulae, physical constants, dates, statistics, or exam patterns. If you are not fully certain, say "I'm not 100% sure" and reason it through carefully instead of guessing.
- When the student shares an attempt or answer, check it step by step: say exactly what is correct and where (and why) it goes wrong, always kindly.
- Concise but complete: enough to truly understand, never a wall of text.
- Remember the punctuation rule: never use em dashes or en dashes, use commas, colons, periods, or parentheses instead.
- Stay warm and encouraging from the first word to the last.`;

// Heuristic auto-routing: pick the best path when the student leaves it on
// "Standard" (most never switch). Math/derivations → reasoning ("thinking");
// current-events / factual lookup → grounded Search; otherwise standard.
function classifyQuery(message: string): "standard" | "thinking" | "search" {
  const text = message || "";
  const m = text.toLowerCase();

  // Quantitative / multi-step reasoning → thinking. Checked first so a math
  // problem that merely mentions a year isn't mistaken for current-events.
  if (
    /\b(solve|calculate|compute|evaluate|prove|derive|derivation|simplify|integrate|differentiate|factori[sz]e)\b/.test(m) ||
    /\b(find|what is|determine)\b.*\b(value|sum|product|roots?|derivative|integral|probability|area|volume|equation)\b/.test(m) ||
    /[∫√∑∏]/.test(text) ||
    /\d\s*[+\-*/^]\s*\d/.test(text)
  ) {
    return "thinking";
  }

  // Current / real-time / factual lookup → grounded Search.
  if (
    /\b(latest|current|today|recent|nowadays|right now|live|this year|up to date|up-to-date)\b/.test(m) ||
    /\b20[2-9]\d\b/.test(m) ||
    /\bwho (won|is the (current|present)|holds)\b/.test(m) ||
    /\b(price|cost) of\b/.test(m)
  ) {
    return "search";
  }

  return "standard";
}

// Appended to the system prompt for quantitative problems.
const QUANT_ADDENDUM = `

QUANTITATIVE / PROBLEM-SOLVING MODE, this question needs careful reasoning:
- Work it out rigorously, showing EVERY step and the reason for each.
- Use correct formulae and constants; if you use a constant (g, R, π, etc.), state its value.
- After the final answer, RE-CHECK it: verify the units and recompute or plug back a key step, then state the verified final answer clearly.
- If the problem is missing data or is ambiguous, say what's missing rather than assuming.`;

/** Build the full teaching system prompt (used by /chat). */
function buildSystemInstruction(
  f: { board?: string; grade?: string; language?: string; preferredAnalogy?: string },
  referenceContext: string | null,
  isQuant: boolean
): string {
  return (
    `${CLARIFY_SYSTEM_INSTRUCTION}

STUDENT CONTEXT (tailor the depth, examples, exam framing, and language to this):
- Board/Exam Target: ${f.board || "General Study"}
- Grade/Level: ${f.grade || "Not Specified"}
- Language Preference: ${f.language || "English"}
- Preferred Analogy Type: ${f.preferredAnalogy || "Daily Life"}` +
    (referenceContext
      ? `\n\nREFERENCE MATERIAL (board-aligned curriculum notes, prefer these for facts and definitions; if they don't cover the question, use your own knowledge):\n${referenceContext}`
      : "") +
    (isQuant ? QUANT_ADDENDUM : "")
  );
}

export const aiRouter = Router();

aiRouter.post("/chat", requireAuth, async (req: Request, res: Response) => {
  const chatT0 = Date.now();
  try {
    const { message, history, mode, board, grade, language, preferredAnalogy } = req.body;
    const uid = (req as any).userId as number;

    if (!rateLimit(`${uid}:chat`, 30)) {
      return res.status(429).json({ error: "You're sending messages very fast. Take a breath and try again in a moment. 🌱" });
    }

    // Uploaded images / files (multimodal). Each: { data: base64, mimeType }.
    const images = Array.isArray(req.body?.images)
      ? req.body.images.filter((im: any) => im && im.data && im.mimeType).slice(0, 6)
      : [];
    const hasImages = images.length > 0;

    // Auto-route when the student left it on the default "Standard" (most do).
    const requestedMode = mode || "standard";
    const effectiveMode = requestedMode === "standard" ? classifyQuery(message) : requestedMode;
    const isQuant = effectiveMode === "thinking";
    const temperature = isQuant ? 0.3 : 0.6; // low temp for math accuracy, warmer for explanations
    if (requestedMode !== effectiveMode) console.log(`[AI] auto-routed: ${requestedMode} → ${effectiveMode}`);

    const cacheable =
      !hasImages &&
      (effectiveMode === "standard" || effectiveMode === "thinking") &&
      (!Array.isArray(history) || history.length === 0);
    const facets: CacheFacets = { mode: effectiveMode, board, grade, language, preferredAnalogy };
    const cacheKey = cacheable ? makeCacheKey({ ...facets, message }) : "";
    const deepVerify = req.body?.deepVerify === true;
    let queryEmbedding: number[] | null = null;

    console.log(
      `[CHAT] start (requested=${requestedMode}, effective=${effectiveMode}, deepVerify=${deepVerify}, images=${images.length}, history=${Array.isArray(history) ? history.length : 0}, cacheable=${cacheable})`
    );
    const logTotal = (path: string) => console.log(`[CHAT] total - ${secs(chatT0)} (${path})`);

    // Serve a cache hit honestly under Deep-check: a hit that never went
    // through the examiner pass is verified NOW (and the entry upgraded), so
    // Deep-check ON can never silently return an unexamined answer.
    const serveCachedHit = async (
      hit: { text: string; sources: any[] },
      upgrade: (verifiedText: string) => Promise<void>
    ) => {
      const v = await verifyAnswer(message, hit.text);
      if (v.verified) {
        await upgrade(v.text);
        logTotal("cache hit + deep-check upgrade");
        return res.json({ text: v.text, sources: hit.sources || [], cached: true, verification: "passed" });
      }
      logTotal("cache hit, deep-check unavailable");
      return res.json({ text: hit.text, sources: hit.sources || [], cached: true, verification: "unavailable" });
    };

    if (cacheable) {
      // 1) Exact cache hit (instant).
      const exactT0 = Date.now();
      const exact = memCache.get(cacheKey) || (await safe(() => cacheGetByKey(pool, cacheKey)));
      console.log(`[CACHE_EXACT] end - ${secs(exactT0)} (${exact ? "hit" : "miss"})`);
      if (exact) {
        if (!deepVerify || exact.verified) {
          memCacheSet(cacheKey, exact);
          logTotal("exact cache hit");
          return res.json({
            text: exact.text,
            sources: exact.sources || [],
            cached: true,
            ...(deepVerify ? { verification: "passed" } : {}),
          });
        }
        return serveCachedHit(exact, async (verifiedText) => {
          memCacheSet(cacheKey, { text: verifiedText, sources: exact.sources || [], verified: true });
          await safe(() => cacheMarkVerified(pool, cacheKey, verifiedText));
        });
      }
      // 2) Semantic cache: embed once (reused for RAG) and match near-duplicates.
      queryEmbedding = await embed(message);
      if (queryEmbedding) {
        const qTokens = topicTokens(message);
        const candidates = (await safe(() => cacheCandidates(pool, facets))) || [];
        let best: { cacheKey: string; text: string; sources: any[]; verified: boolean } | null = null;
        let bestScore = 0;
        for (const c of candidates) {
          if (!c.embedding) continue;
          // Topic gate: only reuse across questions about the SAME thing.
          if (!topicCompatible(qTokens, topicTokens(c.question))) continue;
          const s = cosine(queryEmbedding, c.embedding);
          if (s > bestScore) {
            bestScore = s;
            best = c;
          }
        }
        if (best) console.log(`[Cache] best topic-gated semantic score ${bestScore.toFixed(3)} (threshold ${SEMANTIC_THRESHOLD})`);
        if (best && bestScore >= SEMANTIC_THRESHOLD) {
          console.log(`[Cache] semantic hit (score ${bestScore.toFixed(3)}).`);
          if (!deepVerify || best.verified) {
            logTotal("semantic cache hit");
            return res.json({
              text: best.text,
              sources: best.sources || [],
              cached: true,
              ...(deepVerify ? { verification: "passed" } : {}),
            });
          }
          const bestKey = best.cacheKey;
          return serveCachedHit(best, async (verifiedText) => {
            await safe(() => cacheMarkVerified(pool, bestKey, verifiedText));
          });
        }
      }
    }

    const finish = async (text: string, sources: CachedAnswer["sources"]) => {
      let finalText = text;
      let verified = false;
      if (deepVerify) {
        const v = await verifyAnswer(message, text);
        finalText = v.text;
        verified = v.verified;
      }
      if (cacheable) {
        // The verified flag records whether the examiner pass actually ran, so
        // later Deep-check requests know whether this entry still needs one.
        memCacheSet(cacheKey, { text: finalText, sources: sources || [], verified });
        await safe(() =>
          cacheUpsertFull(pool, {
            cacheKey,
            ...facets,
            question: (message || "").toLowerCase().trim(),
            embedding: queryEmbedding,
            text: finalText,
            sources: sources || [],
            verified,
          })
        );
      }
      logTotal(`generated (mode=${effectiveMode}, verify=${deepVerify ? (verified ? "passed" : "unavailable") : "off"})`);
      res.json({
        text: finalText,
        sources: sources || [],
        ...(deepVerify ? { verification: verified ? "passed" : "unavailable" } : {}),
      });
    };

    // MiniMax is the brain for Standard + Thinking (text), so Gemini isn't
    // required for those, it is only needed for Search grounding, image vision,
    // and as a fallback when MiniMax is unavailable. A client that just watched
    // a /chat/stream attempt fail sends avoidOpenSource so we don't make it
    // sit through a second MiniMax timeout before the Gemini fallback. The flag
    // is advisory: when Gemini cannot serve at all (no key), MiniMax is still
    // tried, because a slow answer beats a guaranteed 500.
    const avoidOpenSource = req.body?.avoidOpenSource === true && Boolean(apiKey);
    const usesOpenSourceBrain =
      (effectiveMode === "standard" || effectiveMode === "thinking") && openSourceEnabled && !hasImages && !avoidOpenSource;
    const needsGemini = !usesOpenSourceBrain;
    if (!apiKey && needsGemini) {
      return res.status(500).json({ error: "GEMINI_API_KEY is missing on the server." });
    }

    // RAG: pull the nearest NCERT-aligned notes (reuse the embedding from the
    // semantic-cache step; first-turn, non-search questions only).
    let referenceContext: string | null = null;
    if (queryEmbedding && effectiveMode !== "search") {
      const ragT0 = Date.now();
      referenceContext = await safe(() => retrieveContext(queryEmbedding, board));
      console.log(`[RAG_RETRIEVE] end - ${secs(ragT0)} (${referenceContext ? "context found" : "no match"}, local DB + JS cosine, no external call)`);
      if (referenceContext) console.log(`[RAG] grounded answer with curriculum context (board: ${board || "General"}).`);
    }

    const systemInstruction = buildSystemInstruction({ board, grade, language, preferredAnalogy }, referenceContext, isQuant);

    let modelName = "gemini-3.5-flash";
    const config: any = { systemInstruction, temperature };

    if (effectiveMode === "thinking") {
      modelName = "gemini-3.1-pro-preview";
      config.thinkingConfig = { thinkingLevel: "HIGH" };
    } else if (effectiveMode === "search") {
      modelName = "gemini-3.5-flash";
      config.tools = [{ googleSearch: {} }];
    }

    const contents: any[] = [];
    if (Array.isArray(history)) {
      for (const h of history) contents.push({ role: h.role === "user" ? "user" : "model", parts: [{ text: h.text }] });
    }
    const userParts: any[] = [{ text: message || "Please look at the attached image and help me understand it." }];
    for (const im of images) userParts.push({ inlineData: { mimeType: im.mimeType, data: im.data } });
    contents.push({ role: "user", parts: userParts });

    // Standard + Thinking → MiniMax (the open-source "brain") as the primary
    // model. Gemini is used ONLY as a tool: Google Search grounding (search mode),
    // image vision (uploads), TTS, and live voice. Skipped here when files are
    // attached, since MiniMax can't see images, those fall through to Gemini.
    // If MiniMax is unavailable, we fall back to Gemini below.
    if (usesOpenSourceBrain) {
      try {
        const text = await callOpenSource(config.systemInstruction, toOpenAIMessages(history, message), temperature);
        return finish(text, []);
      } catch (osErr: any) {
        console.warn(`[AI] ${effectiveMode} open-source failed, falling back to Gemini:`, osErr.message);
      }
    }

    // Trace label reflects what Gemini is doing here: grounding (search mode),
    // vision (image uploads), or plain generation / MiniMax fallback.
    const geminiLabel =
      effectiveMode === "search" ? "GEMINI_SEARCH_GROUND" : hasImages ? "GEMINI_VISION_GENERATE" : "GEMINI_GENERATE";
    let response;
    let gemT0 = Date.now();
    console.log(`[${geminiLabel}] start (model=${modelName}). NOTE: the SDK may retry internally up to 5 attempts with backoff on 408/429/5xx.`);
    try {
      response = await ai.models.generateContent({ model: modelName, contents, config });
      console.log(`[${geminiLabel}] end - ${secs(gemT0)} (model=${modelName})`);
    } catch (apiError: any) {
      console.warn(`[${geminiLabel}] failed - ${secs(gemT0)} (model=${modelName}): ${apiError.message}`);
      if (modelName === "gemini-3.1-pro-preview") {
        // Silent full-generation retry: pro-preview failure re-generates on flash.
        modelName = "gemini-3.5-flash";
        gemT0 = Date.now();
        console.log(`[${geminiLabel}] retry start (model=${modelName}, full re-generation after pro-preview failure)`);
        response = await ai.models.generateContent({
          model: modelName,
          contents,
          config: { ...config, thinkingConfig: { thinkingLevel: "LOW" } },
        });
        console.log(`[${geminiLabel}] retry end - ${secs(gemT0)} (model=${modelName})`);
      } else {
        throw apiError;
      }
    }

    const responseText = response.text || "I was unable to formulate a response. Let me try again!";
    const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
    const sources =
      groundingChunks?.map((c: any) => ({ title: c.web?.title || "Search Source", uri: c.web?.uri || "#" })) || [];

    // Search mode ships Gemini's grounded answer directly. It is generated
    // with the full teaching system prompt, so it is already complete; the old
    // MiniMax rewrite step only re-generated the same content (measured: +38s
    // on a 10s grounded answer) and was removed on 2026-07-02.
    return finish(responseText, sources);
  } catch (error: any) {
    console.warn(`[CHAT] total - ${secs(chatT0)} (FAILED: ${error?.message || error})`);
    console.error("Chat API error:", error);
    res.status(500).json({ error: error.message || "An error occurred during content generation." });
  }
});

/**
 * Streaming chat (SSE over POST). The draft streams token by token; with
 * Deep-check on, the examiner pass runs on the COMPLETE draft afterwards and
 * the corrected final answer replaces it in the closing "done" event, so
 * streaming never weakens the fact-check net (the 2026-07-02 draft-then-swap
 * decision that superseded the June whole-answer-only rule).
 *
 * Events (data: JSON lines): {type:"delta",text} incremental chunk,
 * {type:"checking"} examiner started, {type:"done",text,sources,verification?}
 * final authoritative answer, {type:"fallback",reason} use plain /chat,
 * {type:"error",error}. Cache hits emit their full text as one delta.
 * Search, image, and no-open-source requests fall back to /chat, which keeps
 * its Gemini paths and remains the safety net when the stream fails.
 */
aiRouter.post("/chat/stream", requireAuth, async (req: Request, res: Response) => {
  const chatT0 = Date.now();
  const send = (payload: Record<string, unknown>) => {
    if (!res.destroyed) res.write(`data: ${JSON.stringify(payload)}\n\n`);
  };
  try {
    const { message, history, mode, board, grade, language, preferredAnalogy } = req.body;
    const uid = (req as any).userId as number;

    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");
    res.flushHeaders?.();

    const images = Array.isArray(req.body?.images)
      ? req.body.images.filter((im: any) => im && im.data && im.mimeType).slice(0, 6)
      : [];
    const requestedMode = mode || "standard";
    const effectiveMode = requestedMode === "standard" ? classifyQuery(message) : requestedMode;
    const deepVerify = req.body?.deepVerify === true;

    // Only the open-source text path streams; everything else uses /chat.
    // Checked BEFORE the rate limit: a route fallback does no AI work, so it
    // must not charge the student a token (the /chat retry pays the one token).
    if (images.length > 0 || effectiveMode === "search" || !openSourceEnabled) {
      console.log(`[CHAT_STREAM] fallback to /chat (mode=${effectiveMode}, images=${images.length}, openSource=${openSourceEnabled})`);
      send({ type: "fallback", reason: "route" });
      return res.end();
    }

    if (!rateLimit(`${uid}:chat`, 30)) {
      send({ type: "error", error: "You're sending messages very fast. Take a breath and try again in a moment. 🌱" });
      return res.end();
    }

    const isQuant = effectiveMode === "thinking";
    const temperature = isQuant ? 0.3 : 0.6;
    const cacheable = !Array.isArray(history) || history.length === 0;
    const facets: CacheFacets = { mode: effectiveMode, board, grade, language, preferredAnalogy };
    const cacheKey = cacheable ? makeCacheKey({ ...facets, message }) : "";
    let queryEmbedding: number[] | null = null;

    console.log(`[CHAT_STREAM] start (requested=${requestedMode}, effective=${effectiveMode}, deepVerify=${deepVerify}, cacheable=${cacheable})`);

    // Serve a cached answer over the stream with the same Deep-check honesty
    // as /chat: unverified hits are examined now and upgraded.
    const streamCachedHit = async (
      hit: { text: string; sources: any[]; verified: boolean },
      upgrade: (verifiedText: string) => Promise<void>,
      path: string
    ) => {
      send({ type: "delta", text: hit.text });
      if (!deepVerify || hit.verified) {
        send({ type: "done", text: hit.text, sources: hit.sources || [], ...(deepVerify ? { verification: "passed" } : {}) });
      } else {
        send({ type: "checking" });
        const v = await verifyAnswer(message, hit.text);
        if (v.verified) await upgrade(v.text);
        send({
          type: "done",
          text: v.verified ? v.text : hit.text,
          sources: hit.sources || [],
          verification: v.verified ? "passed" : "unavailable",
        });
      }
      console.log(`[CHAT_STREAM] total - ${secs(chatT0)} (${path})`);
      res.end();
    };

    if (cacheable) {
      const exact = memCache.get(cacheKey) || (await safe(() => cacheGetByKey(pool, cacheKey)));
      if (exact) {
        memCacheSet(cacheKey, exact);
        return streamCachedHit(
          exact,
          async (verifiedText) => {
            memCacheSet(cacheKey, { text: verifiedText, sources: exact.sources || [], verified: true });
            await safe(() => cacheMarkVerified(pool, cacheKey, verifiedText));
          },
          "exact cache hit"
        );
      }
      queryEmbedding = await embed(message);
      if (queryEmbedding) {
        const qTokens = topicTokens(message);
        const candidates = (await safe(() => cacheCandidates(pool, facets))) || [];
        let best: { cacheKey: string; text: string; sources: any[]; verified: boolean } | null = null;
        let bestScore = 0;
        for (const c of candidates) {
          if (!c.embedding) continue;
          if (!topicCompatible(qTokens, topicTokens(c.question))) continue;
          const s = cosine(queryEmbedding, c.embedding);
          if (s > bestScore) {
            bestScore = s;
            best = c;
          }
        }
        if (best && bestScore >= SEMANTIC_THRESHOLD) {
          const bestKey = best.cacheKey;
          return streamCachedHit(
            best,
            async (verifiedText) => {
              await safe(() => cacheMarkVerified(pool, bestKey, verifiedText));
            },
            `semantic cache hit (score ${bestScore.toFixed(3)})`
          );
        }
      }
    }

    // RAG grounding, same rules as /chat (first-turn questions only).
    let referenceContext: string | null = null;
    if (queryEmbedding) {
      referenceContext = await safe(() => retrieveContext(queryEmbedding, board));
    }
    const systemInstruction = buildSystemInstruction({ board, grade, language, preferredAnalogy }, referenceContext, isQuant);

    // Stream the draft. Any failure here (timeout, HTTP error, empty stream)
    // hands the request back to the client, which retries on /chat with
    // avoidOpenSource so it goes straight to the Gemini fallback.
    let draft = "";
    try {
      for await (const delta of streamOpenSource(systemInstruction, toOpenAIMessages(history, message), temperature)) {
        draft += delta;
        if (res.destroyed) break;
        send({ type: "delta", text: delta });
      }
    } catch {
      // If chunks already reached the student, the client keeps them visible
      // and retries silently; "reason" tells it to skip MiniMax this time.
      send({ type: "fallback", reason: "stream-failed" });
      console.warn(`[CHAT_STREAM] total - ${secs(chatT0)} (stream failed, client falls back to /chat)`);
      return res.end();
    }
    if (res.destroyed) {
      console.log(`[CHAT_STREAM] total - ${secs(chatT0)} (client disconnected mid-stream)`);
      return;
    }

    // Draft-then-swap: examiner pass on the complete draft, then cache + done.
    let finalText = draft;
    let verified = false;
    if (deepVerify) {
      send({ type: "checking" });
      const v = await verifyAnswer(message, draft);
      finalText = v.text;
      verified = v.verified;
    }
    if (cacheable) {
      memCacheSet(cacheKey, { text: finalText, sources: [], verified });
      await safe(() =>
        cacheUpsertFull(pool, {
          cacheKey,
          ...facets,
          question: (message || "").toLowerCase().trim(),
          embedding: queryEmbedding,
          text: finalText,
          sources: [],
          verified,
        })
      );
    }
    send({
      type: "done",
      text: finalText,
      sources: [],
      ...(deepVerify ? { verification: verified ? "passed" : "unavailable" } : {}),
    });
    console.log(`[CHAT_STREAM] total - ${secs(chatT0)} (streamed, verify=${deepVerify ? (verified ? "passed" : "unavailable") : "off"})`);
    res.end();
  } catch (error: any) {
    console.warn(`[CHAT_STREAM] total - ${secs(chatT0)} (FAILED: ${error?.message || error})`);
    send({ type: "error", error: error?.message || "An error occurred during content generation." });
    res.end();
  }
});

aiRouter.post("/tts", requireAuth, async (req: Request, res: Response) => {
  try {
    const { text, voice } = req.body;
    const uid = (req as any).userId as number;
    if (!rateLimit(`${uid}:tts`, 30)) return res.status(429).json({ error: "Too many audio requests right now. Please wait a moment." });
    if (!apiKey) return res.status(500).json({ error: "GEMINI_API_KEY is missing." });

    const response = await ai.models.generateContent({
      model: "gemini-3.1-flash-tts-preview",
      contents: [{ parts: [{ text: `Say clearly and warmly: ${text}` }] }],
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: voice || "Kore" } } },
      },
    });

    const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (base64Audio) res.json({ audio: base64Audio });
    else res.status(404).json({ error: "No audio stream returned from Gemini TTS model." });
  } catch (error: any) {
    console.error("TTS API error:", error);
    res.status(500).json({ error: error.message || "TTS generation failed." });
  }
});

/** Attach the live voice WebSocket (/api/live) to the HTTP server. */
export function attachLiveWebSocket(server: http.Server) {
  const wss = new WebSocketServer({ noServer: true });

  server.on("upgrade", (request, socket, head) => {
    const url = new URL(request.url || "", `http://${request.headers.host}`);
    if (url.pathname !== "/api/live") {
      socket.destroy();
      return;
    }
    // Browser WebSocket can't send headers, so the JWT comes as ?token=
    if (!userIdFromToken(url.searchParams.get("token"))) {
      socket.destroy();
      return;
    }
    wss.handleUpgrade(request, socket, head, (ws) => wss.emit("connection", ws, request));
  });

  wss.on("connection", async (clientWs: WebSocket) => {
    let liveSession: any = null;

    clientWs.on("message", async (data: any) => {
      try {
        const msg = JSON.parse(data.toString());
        if (msg.type === "start" && !liveSession) {
          if (!apiKey) {
            clientWs.send(JSON.stringify({ type: "error", error: "API Key is missing." }));
            return;
          }
          try {
            liveSession = await ai.live.connect({
              model: "gemini-3.1-flash-live-preview",
              config: {
                responseModalities: [Modality.AUDIO],
                speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: "Zephyr" } } },
                systemInstruction: `You are Clarify.AI, the student's personal real-time voice mentor.
You are warm, patient, calm, and encouraging.
Speak in short, conversational sentences suitable for audio dialogue.
Guide the student step-by-step. If they are confused, give daily life analogies.
Encourage their thinking and efforts! Keep explanations simple, friendly, and easy to follow.`,
              },
              callbacks: {
                onmessage: (message: any) => {
                  const audio = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
                  if (audio) clientWs.send(JSON.stringify({ type: "audio", audio }));
                  if (message.serverContent?.interrupted) clientWs.send(JSON.stringify({ type: "interrupted" }));
                },
              },
            });
            clientWs.send(JSON.stringify({ type: "ready", message: "Clarify.AI is listening! Start speaking..." }));
          } catch (err: any) {
            clientWs.send(JSON.stringify({ type: "error", error: "Failed to connect to Live API: " + err.message }));
          }
          return;
        }
        if (msg.audio && liveSession) {
          await liveSession.sendRealtimeInput({ audio: { data: msg.audio, mimeType: "audio/pcm;rate=16000" } });
        }
      } catch (err: any) {
        clientWs.send(JSON.stringify({ type: "error", error: err.message }));
      }
    });

    clientWs.on("close", () => {
      if (liveSession) {
        try {
          liveSession.close();
        } catch {
          /* ignore */
        }
      }
    });
  });
}
