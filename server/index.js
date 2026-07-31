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

  const {
    system: systemPrompt,
    messages: conversationHistory,
    question,
    history = [],
    model,
    max_tokens = 1000,
  } = req.body;

  let messages = [];

  if (typeof systemPrompt === "string" && systemPrompt.trim()) {
    messages.push({ role: "system", content: systemPrompt.trim() });
  }

  if (Array.isArray(conversationHistory) && conversationHistory.length > 0) {
    messages.push(...conversationHistory);
  } else if (Array.isArray(history) && history.length > 0) {
    messages.push(...history);
  }

  if (question && typeof question === "string" && question.trim()) {
    messages.push({ role: "user", content: question.trim() });
  }

  if (messages.length === 0) {
    res.status(400).json({ error: "Messages array or question is required." });
    return;
  }

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  if (typeof res.flushHeaders === "function") {
    res.flushHeaders();
  }

  try {
    const groq = new Groq({ apiKey });
    const stream = await groq.chat.completions.create({
      model: model || "llama-3.3-70b-versatile",
      messages,
      max_tokens,
      stream: true,
    });

    let fullText = "";

    for await (const chunk of stream) {
      const contentDelta = chunk.choices?.[0]?.delta?.content || "";
      if (contentDelta) {
        fullText += contentDelta;
        res.write(`data: ${JSON.stringify({
          type: "content_block_delta",
          delta: { type: "text_delta", text: contentDelta },
        })}\n\n`);
      }
    }

    res.write(`data: ${JSON.stringify({
      type: "final",
      result: { answer: fullText, citations: [] },
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