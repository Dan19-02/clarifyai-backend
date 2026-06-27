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
import { GoogleGenAI, Modality } from "@google/genai";
import { requireAuth, userIdFromToken } from "./auth.js";
import { pool, cacheGet as dbCacheGet, cacheSet as dbCacheSet, type CachedAnswer } from "./db.js";

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.warn("[AI] GEMINI_API_KEY missing — chat/tts/image/live will error until it is set.");
}

const ai = new GoogleGenAI({
  apiKey: apiKey || "",
  httpOptions: { headers: { "User-Agent": "aistudio-build" } },
});

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
        temperature: 0.7,
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

function makeCacheKey(p: any): string {
  const norm = (s: string) => (s || "").toString().toLowerCase().replace(/\s+/g, " ").trim();
  const raw = [p.mode, p.board, p.grade, p.language, p.preferredAnalogy, norm(p.message)].join("||");
  return crypto.createHash("sha256").update(raw).digest("hex");
}

async function cacheGet(key: string): Promise<CachedAnswer | null> {
  const inMem = memCache.get(key);
  if (inMem) return inMem;
  try {
    const fromDb = await dbCacheGet(pool, key);
    if (fromDb) memCache.set(key, fromDb);
    return fromDb;
  } catch {
    return null;
  }
}

async function cacheSet(key: string, value: CachedAnswer): Promise<void> {
  if (memCache.size >= MEM_CACHE_MAX) {
    const oldest = memCache.keys().next().value;
    if (oldest) memCache.delete(oldest);
  }
  memCache.set(key, value);
  try {
    await dbCacheSet(pool, key, value);
  } catch {
    /* best-effort */
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
A fully solved, step-by-step example. Show each step and the reasoning. Use LaTeX for any math.

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

HARD RULES
- Be accurate. Never invent facts, formulae, or exam data. If you are unsure, say so honestly and reason it through carefully.
- Concise but complete — enough to truly understand, never a wall of text.
- Stay warm and encouraging from the first word to the last.`;

export const aiRouter = Router();

aiRouter.post("/chat", requireAuth, async (req: Request, res: Response) => {
  try {
    const { message, history, mode, board, grade, language, preferredAnalogy } = req.body;
    const uid = (req as any).userId as number;

    if (!rateLimit(`${uid}:chat`, 30)) {
      return res.status(429).json({ error: "You're sending messages very fast. Take a breath and try again in a moment. 🌱" });
    }

    const cacheable =
      (mode === "standard" || mode === "thinking") && (!Array.isArray(history) || history.length === 0);
    const cacheKey = cacheable ? makeCacheKey({ mode, board, grade, language, preferredAnalogy, message }) : "";

    if (cacheable) {
      const hit = await cacheGet(cacheKey);
      if (hit) return res.json({ text: hit.text, sources: hit.sources || [], cached: true });
    }

    const finish = async (text: string, sources: CachedAnswer["sources"]) => {
      if (cacheable) await cacheSet(cacheKey, { text, sources: sources || [] });
      res.json({ text, sources: sources || [] });
    };

    const needsGemini = !(mode === "thinking" && openSourceEnabled);
    if (!apiKey && needsGemini) {
      return res.status(500).json({ error: "GEMINI_API_KEY is missing on the server." });
    }

    let modelName = "gemini-3.5-flash";
    const config: any = {
      systemInstruction: `${CLARIFY_SYSTEM_INSTRUCTION}

STUDENT CONTEXT (tailor the depth, examples, exam framing, and language to this):
- Board/Exam Target: ${board || "General Study"}
- Grade/Level: ${grade || "Not Specified"}
- Language Preference: ${language || "English"}
- Preferred Analogy Type: ${preferredAnalogy || "Daily Life"}`,
    };

    if (mode === "thinking") {
      modelName = "gemini-3.1-pro-preview";
      config.thinkingConfig = { thinkingLevel: "HIGH" };
    } else if (mode === "search") {
      modelName = "gemini-3.5-flash";
      config.tools = [{ googleSearch: {} }];
    }

    const contents: any[] = [];
    if (Array.isArray(history)) {
      for (const h of history) contents.push({ role: h.role === "user" ? "user" : "model", parts: [{ text: h.text }] });
    }
    contents.push({ role: "user", parts: [{ text: message }] });

    // Thinking mode → open-source model (skips expensive Gemini Pro).
    if (mode === "thinking" && openSourceEnabled) {
      try {
        const text = await callOpenSource(config.systemInstruction, toOpenAIMessages(history, message));
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
    if (mode === "search" && openSourceEnabled) {
      try {
        const augmented =
          `Here is up-to-date information gathered from a Google Search to help you answer accurately:\n\n"""\n${responseText}\n"""\n\n` +
          `Using the information above where relevant (plus your own knowledge), respond to the student's request: ${message}`;
        const text = await callOpenSource(config.systemInstruction, toOpenAIMessages(history, augmented));
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

aiRouter.post("/generate-image", requireAuth, async (req: Request, res: Response) => {
  try {
    const { prompt, size } = req.body;
    const uid = (req as any).userId as number;
    if (!rateLimit(`${uid}:image`, 10)) return res.status(429).json({ error: "Image limit reached for now. Please wait a minute and try again." });
    if (!apiKey) return res.status(500).json({ error: "GEMINI_API_KEY is missing." });

    const buildPrompt = `Create a clean, elegant, clear educational diagram or high-quality illustration. It should be perfect for a student learning about: ${prompt}. Style: Crisp educational diagram, no clutter, visually pleasing colors, labels if necessary.`;

    let response;
    let modelName = "gemini-3.1-flash-image";
    try {
      response = await ai.models.generateContent({
        model: modelName,
        contents: { parts: [{ text: buildPrompt }] },
        config: { imageConfig: { aspectRatio: "1:1", imageSize: size || "1K" } },
      });
    } catch (imageErr: any) {
      console.warn(`[AI] image ${modelName} failed:`, imageErr.message);
      modelName = "gemini-2.5-flash-image";
      try {
        response = await ai.models.generateContent({
          model: modelName,
          contents: { parts: [{ text: buildPrompt }] },
          config: { imageConfig: { aspectRatio: "1:1" } },
        });
      } catch (fallbackErr: any) {
        return res.status(429).json({
          error: "Image generation quota exceeded or requires a paid API key. Please try again later.",
          details: fallbackErr.message,
        });
      }
    }

    let imageUrl: string | null = null;
    let fallbackText = "";
    for (const part of response.candidates?.[0]?.content?.parts || []) {
      if (part.inlineData) imageUrl = `data:image/png;base64,${part.inlineData.data}`;
      else if (part.text) fallbackText += part.text;
    }

    if (imageUrl) res.json({ imageUrl });
    else res.status(404).json({ error: "No image data found in response.", details: fallbackText });
  } catch (error: any) {
    console.error("Image generation API error:", error);
    res.status(500).json({ error: error.message || "Image generation failed." });
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
