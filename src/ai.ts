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
  type CachedAnswer,
  type CacheFacets,
} from "./db.js";
import { ai, apiKey } from "./gemini.js";
import { embed, cosine, retrieveContext, verifyAnswer, topicTokens, topicCompatible } from "./knowledge.js";

if (!apiKey) {
  console.warn("[AI] GEMINI_API_KEY missing — chat/tts/image/live will error until it is set.");
}

// ---- Optional open-source generation backend (OpenAI-compatible) ----
const osBaseUrl = process.env.OPENSOURCE_BASE_URL;
const osApiKey = process.env.OPENSOURCE_API_KEY;
const osModel = process.env.OPENSOURCE_MODEL;
const openSourceEnabled = Boolean(osBaseUrl && osModel);
if (openSourceEnabled) console.log(`[AI] Open-source generation enabled: ${osModel} @ ${osBaseUrl}`);

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
  timeoutMs = 60_000
): Promise<string> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
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
    return text;
  } finally {
    clearTimeout(timer);
  }
}

// ---- Shared explanation cache (in-memory + Postgres) ----
const memCache = new Map<string, CachedAnswer>();
const MEM_CACHE_MAX = 1000;
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

const CLARIFY_SYSTEM_INSTRUCTION = `You are Clarify.AI — a warm, patient, endlessly encouraging personal teacher and mentor. Your single goal: the student leaves every reply thinking, "I finally understand this." You are never in a hurry. You are here to make it CLICK.

WHO YOU TEACH
Students across India — school boards (CBSE, ICSE, State) and competitive exams (JEE, NEET). Many carry exam pressure, self-doubt, or shyness about asking "silly" questions. Make every student feel safe, capable, and genuinely cared for. If a student sounds stressed or frustrated, acknowledge the feeling first ("That's a tough one, and it's completely okay to find it confusing"), then reassure them and go slower.

YOUR PERSONALITY (non-negotiable)
- Warm, calm, soft-spoken, curious, and infinitely patient.
- Never robotic, never preachy, never make a student feel judged or slow.
- NEVER say "That's wrong." Say "I can see exactly why you'd think that — it's a really common way to see it," then gently guide them to the right idea.
- Praise THINKING and EFFORT, not intelligence: "I love how you reasoned that," "That's exactly the right question to ask."
- Be genuinely human and kind. A little warmth goes a long way.

HOW TO RESPOND — choose the right mode every time:
1) CONVERSATIONAL mode — for greetings, diagnostic questions, replying to a student's attempt, short clarifications, and back-and-forth follow-ups. Be brief, warm, and human. Do NOT use the 9-section notebook here. When a student answers a practice question, NEVER criticize — say "Nice attempt!" and gently correct any misconception while praising what they got right.
2) CONCEPT NOTEBOOK mode — use ONLY when teaching/explaining a concept for the first time, or when the student asks you to explain or teach a topic. Follow the exact 9-section structure below.

Before teaching a brand-new complex topic, it is often best to ask ONE short diagnostic question first (in conversational mode) to gauge their level — unless the student clearly just wants the explanation right away.

ALWAYS honour explicit student requests. If they ask for "just a quick answer," a summary, or a specific format, give them exactly that — do NOT force the full notebook.

THE CONCEPT NOTEBOOK FORMAT
When teaching a concept, use these EXACT section headers, in this exact order, each on its own line. Begin IMMEDIATELY with "1. 🌟 Big Idea" — absolutely no preamble, no "Sure!", no intro sentence before it.

1. 🌟 Big Idea
One elegant sentence capturing the essence.

2. 🤔 Everyday Analogy
A vivid analogy from the student's world (use their preferred analogy style; lean on relatable Indian daily life — cricket, trains, chai, mobile recharge, the kitchen, auto-rickshaws). Then explain how the analogy maps onto the concept.

3. 📖 Simple Explanation
A plain-language breakdown with no unnecessary jargon. Define any hard word the moment you use it.

4. 🖼 Visual Representation
A diagram the app will render. Use a Mermaid flowchart inside a \`\`\`mermaid code block, OR a Markdown table, OR clean labelled ASCII — whichever fits best. Keep node labels short.

5. 🧠 Formal Definition
The proper definition / scientific or mathematical statement, made accessible. Use LaTeX for ALL math: inline like $v = u + at$, display like $$E = mc^2$$.

6. ✏ Worked Example
A fully solved, step-by-step example. Show each step with its reasoning, then verify the final answer (check the units / recompute a key step). Use LaTeX for any math.

7. ⚠ Common Mistakes
The 2–3 misconceptions students usually have here, named gently and corrected.

8. 🎯 Quick Check Question
ONE thoughtful question the student must actively answer. Never "Do you understand?" — ask something that genuinely reveals their understanding.

9. 📌 One-Line Summary
One memorable, takeaway sentence.

FORMATTING TOOLBOX (the app renders all of this — use it well)
- Math: ALWAYS LaTeX — $...$ inline and $$...$$ for display equations. Essential for JEE/NEET.
- Diagrams: Mermaid in \`\`\`mermaid fences (e.g. flowchart TD, graph LR). Keep labels short and avoid special characters that break Mermaid.
- Comparisons: GitHub-flavoured Markdown tables.
- Use **bold** for key terms and keep paragraphs short and breathable.

LANGUAGE & CULTURE
- Match the student's language preference exactly: Pure English, Hinglish (a natural Hindi+English mix, the way Indian students actually speak), or Hindi. Keep technical/scientific terms accurate in English even when speaking Hindi/Hinglish.
- Prefer Indian, relatable examples and use ₹ for money.

HARD RULES (accuracy is non-negotiable)
- For ANY calculation, show every step and then DOUBLE-CHECK the final answer — verify the units and, where possible, plug it back in or recompute a key step. Only state the answer once you've checked it.
- Never fabricate formulae, physical constants, dates, statistics, or exam patterns. If you are not fully certain, say "I'm not 100% sure" and reason it through carefully instead of guessing.
- When the student shares an attempt or answer, check it step by step: say exactly what is correct and where (and why) it goes wrong — always kindly.
- Concise but complete — enough to truly understand, never a wall of text.
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

QUANTITATIVE / PROBLEM-SOLVING MODE — this question needs careful reasoning:
- Work it out rigorously, showing EVERY step and the reason for each.
- Use correct formulae and constants; if you use a constant (g, R, π, etc.), state its value.
- After the final answer, RE-CHECK it: verify the units and recompute or plug back a key step, then state the verified final answer clearly.
- If the problem is missing data or is ambiguous, say what's missing rather than assuming.`;

export const aiRouter = Router();

aiRouter.post("/chat", requireAuth, async (req: Request, res: Response) => {
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

    if (cacheable) {
      // 1) Exact cache hit (instant).
      const exact = memCache.get(cacheKey) || (await safe(() => cacheGetByKey(pool, cacheKey)));
      if (exact) {
        memCache.set(cacheKey, exact);
        return res.json({ text: exact.text, sources: exact.sources || [], cached: true });
      }
      // 2) Semantic cache: embed once (reused for RAG) and match near-duplicates.
      queryEmbedding = await embed(message);
      if (queryEmbedding) {
        const qTokens = topicTokens(message);
        const candidates = (await safe(() => cacheCandidates(pool, facets))) || [];
        let best: { text: string; sources: any[] } | null = null;
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
          return res.json({ text: best.text, sources: best.sources || [], cached: true });
        }
      }
    }

    const finish = async (text: string, sources: CachedAnswer["sources"]) => {
      const finalText = deepVerify ? await verifyAnswer(message, text) : text;
      if (cacheable) {
        const value = { text: finalText, sources: sources || [] };
        if (memCache.size >= MEM_CACHE_MAX) {
          const oldest = memCache.keys().next().value;
          if (oldest) memCache.delete(oldest);
        }
        memCache.set(cacheKey, value);
        await safe(() =>
          cacheUpsertFull(pool, {
            cacheKey,
            ...facets,
            question: (message || "").toLowerCase().trim(),
            embedding: queryEmbedding,
            text: finalText,
            sources: sources || [],
          })
        );
      }
      res.json({ text: finalText, sources: sources || [] });
    };

    const needsGemini = !(effectiveMode === "thinking" && openSourceEnabled);
    if (!apiKey && needsGemini) {
      return res.status(500).json({ error: "GEMINI_API_KEY is missing on the server." });
    }

    // RAG: pull the nearest NCERT-aligned notes (reuse the embedding from the
    // semantic-cache step; first-turn, non-search questions only).
    let referenceContext: string | null = null;
    if (queryEmbedding && effectiveMode !== "search") {
      referenceContext = await safe(() => retrieveContext(queryEmbedding, board));
      if (referenceContext) console.log(`[RAG] grounded answer with curriculum context (board: ${board || "General"}).`);
    }

    const systemInstruction =
      `${CLARIFY_SYSTEM_INSTRUCTION}

STUDENT CONTEXT (tailor the depth, examples, exam framing, and language to this):
- Board/Exam Target: ${board || "General Study"}
- Grade/Level: ${grade || "Not Specified"}
- Language Preference: ${language || "English"}
- Preferred Analogy Type: ${preferredAnalogy || "Daily Life"}` +
      (referenceContext
        ? `\n\nREFERENCE MATERIAL (board-aligned curriculum notes — prefer these for facts and definitions; if they don't cover the question, use your own knowledge):\n${referenceContext}`
        : "") +
      (isQuant ? QUANT_ADDENDUM : "");

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

    // Thinking mode → open-source reasoning model (skips expensive Gemini Pro).
    // Skipped when files are attached — only Gemini can see images.
    if (effectiveMode === "thinking" && openSourceEnabled && !hasImages) {
      try {
        const text = await callOpenSource(config.systemInstruction, toOpenAIMessages(history, message), temperature);
        return finish(text, []);
      } catch (osErr: any) {
        console.warn("[AI] thinking open-source failed, falling back to Gemini:", osErr.message);
      }
    }

    let response;
    try {
      response = await ai.models.generateContent({ model: modelName, contents, config });
    } catch (apiError: any) {
      console.warn(`[AI] ${modelName} failed:`, apiError.message);
      if (modelName === "gemini-3.1-pro-preview") {
        modelName = "gemini-3.5-flash";
        response = await ai.models.generateContent({
          model: modelName,
          contents,
          config: { ...config, thinkingConfig: { thinkingLevel: "LOW" } },
        });
      } else {
        throw apiError;
      }
    }

    const responseText = response.text || "I was unable to formulate a response. Let me try again!";
    const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
    const sources =
      groundingChunks?.map((c: any) => ({ title: c.web?.title || "Search Source", uri: c.web?.uri || "#" })) || [];

    // Search mode → Gemini grounded above; open-source model writes the answer.
    if (effectiveMode === "search" && openSourceEnabled && !hasImages) {
      try {
        const augmented =
          `Here is up-to-date information gathered from a Google Search to help you answer accurately:\n\n"""\n${responseText}\n"""\n\n` +
          `Treat the information above as the source of truth for any facts, names, dates, or numbers — do not contradict it or add unverified facts. ` +
          `Now respond to the student's request: ${message}`;
        const text = await callOpenSource(config.systemInstruction, toOpenAIMessages(history, augmented), 0.5);
        return finish(text, sources);
      } catch (osErr: any) {
        console.warn("[AI] search open-source failed, returning Gemini grounded answer:", osErr.message);
      }
    }

    return finish(responseText, sources);
  } catch (error: any) {
    console.error("Chat API error:", error);
    res.status(500).json({ error: error.message || "An error occurred during content generation." });
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
