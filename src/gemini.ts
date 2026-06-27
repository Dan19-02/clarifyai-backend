/** Shared Gemini client (used by ai.ts and knowledge.ts). */
import { GoogleGenAI } from "@google/genai";

export const apiKey = process.env.GEMINI_API_KEY;

export const ai = new GoogleGenAI({
  apiKey: apiKey || "",
  httpOptions: { headers: { "User-Agent": "aistudio-build" } },
});
