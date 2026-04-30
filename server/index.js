import cors from "cors";
import dotenv from "dotenv";
import express from "express";
import Groq from "groq-sdk";

import path from "path";

dotenv.config({
  path: path.resolve(process.cwd(), ".env"),
});

const app = express();
const PORT = process.env.PORT || 3001;

function getGroqApiKey() {
  return process.env.GROQ_API_KEY?.trim();
}

app.use(cors());
app.use(express.json({ limit: "2mb" }));

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, groqKeyConfigured: Boolean(getGroqApiKey()) });
});

app.post("/api/chat", async (req, res) => {
  const apiKey = getGroqApiKey();

  if (!apiKey) {
    res.status(500).json({ error: "GROQ_API_KEY is not configured." });
    return;
  }

  try {
    const { system: systemPrompt, messages: conversationHistory = [], model, max_tokens = 1000 } = req.body;
    const groq = new Groq({ apiKey });
    const completion = await groq.chat.completions.create({
      model: model || "llama-3.3-70b-versatile",
      messages: [{ role: "system", content: systemPrompt }, ...conversationHistory],
      max_tokens,
    });
    const responseText = completion.choices[0].message.content;

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");
    res.write(`data: ${JSON.stringify({
      type: "content_block_delta",
      delta: { type: "text_delta", text: responseText },
    })}\n\n`);
    res.write("data: [DONE]\n\n");
    res.end();
  } catch (error) {
    if (!res.headersSent) {
      res.status(500).json({ error: error.message });
      return;
    }
    res.write(`event: error\ndata: ${JSON.stringify({ error: error.message })}\n\n`);
    res.end();
  }
});

app.listen(PORT, () => {
  console.log(`RAG API proxy listening on http://localhost:${PORT}`);
  console.log(`Groq API key configured: ${Boolean(getGroqApiKey())}`);
});
console.log("Loaded GROQ KEY:", process.env.GROQ_API_KEY);