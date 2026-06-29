// Load env FIRST (before db/ai/auth modules read process.env at import time).
import "dotenv/config";

import express from "express";
import cors from "cors";
import http from "http";
import { initDb } from "./db.js";
import { authRouter } from "./auth.js";
import { aiRouter, attachLiveWebSocket } from "./ai.js";
import { ingestKnowledge } from "./knowledge.js";

const app = express();
app.use(cors({ origin: process.env.CORS_ORIGIN || "*" }));
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

app.get("/api/health", (_req, res) => res.json({ ok: true }));
app.use("/api", authRouter);
app.use("/api", aiRouter);

const PORT = Number(process.env.PORT) || 4000;
const server = http.createServer(app);
attachLiveWebSocket(server);

// await initDb();
// server.listen(PORT, () => console.log(`Clarify.AI backend running on http://localhost:${PORT}`));

// // Seed the RAG knowledge base in the background (doesn't block startup).
// ingestKnowledge().catch((e) => console.warn("[RAG] ingest error:", e.message));

async function start() {
  await initDb();

  server.listen(PORT, () => {
    console.log(`Clarify.AI backend running on http://localhost:${PORT}`);
  });

  ingestKnowledge().catch((e) =>
    console.warn("[RAG] ingest error:", e.message)
  );
}

start().catch((err) => {
  console.error("Startup failed:", err);
  process.exit(1);
});
