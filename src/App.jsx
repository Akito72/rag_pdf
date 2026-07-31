import { useCallback, useEffect, useRef, useState } from "react";
import { chatWithAgent, deleteDocument, uploadDocument } from "./api";

const APP_TITLE = import.meta.env.VITE_APP_TITLE || "RAG Pipeline";
const MODEL_OPTIONS = [
  { value: "llama-3.3-70b-versatile", label: "llama-3.3-70b-versatile", note: "Best quality" },
  { value: "llama-3.1-8b-instant", label: "llama-3.1-8b-instant", note: "Fastest" },
  { value: "mixtral-8x7b-32768", label: "mixtral-8x7b-32768", note: "High context" },
];

const STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,300&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0c0e0f;
    --surface: #131618;
    --surface2: #1a1e20;
    --border: #252a2d;
    --border2: #2e3538;
    --amber: #e8a84c;
    --amber-dim: #a87230;
    --green: #5ec98b;
    --red: #e05c5c;
    --text: #d4cfc8;
    --text-dim: #7d858a;
    --text-bright: #f0ebe3;
    --mono: 'IBM Plex Mono', monospace;
    --serif: 'Fraunces', Georgia, serif;
  }

  body { background: var(--bg); color: var(--text); font-family: var(--mono); }
  .app { min-height: 100vh; display: grid; grid-template-rows: auto 1fr; max-width: 1200px; margin: 0 auto; padding: 0 20px; }
  .header { padding: 28px 0 20px; border-bottom: 1px solid var(--border); display: flex; align-items: baseline; gap: 16px; }
  .header-title { font-family: var(--serif); font-size: 1.6rem; font-weight: 600; color: var(--text-bright); }
  .header-badge { font-size: 0.65rem; font-weight: 500; color: var(--amber); border: 1px solid var(--amber-dim); padding: 2px 8px; border-radius: 2px; letter-spacing: 0.08em; text-transform: uppercase; }
  .header-sub { margin-left: auto; font-size: 0.7rem; color: var(--text-dim); letter-spacing: 0.04em; }
  .layout { display: grid; grid-template-columns: 300px 1fr; min-height: 0; }
  .sidebar { border-right: 1px solid var(--border); padding: 20px 16px 20px 0; display: flex; flex-direction: column; gap: 16px; overflow-y: auto; }
  .section-label { font-size: 0.62rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-dim); margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
  .section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }
  .drop-zone { border: 1px dashed var(--border2); border-radius: 4px; padding: 24px 16px; text-align: center; cursor: pointer; transition: all 0.15s ease; background: var(--surface); }
  .drop-zone:hover, .drop-zone.dragging { border-color: var(--amber-dim); background: #1a1a0e; }
  .drop-zone input { display: none; }
  .drop-zone-icon { font-size: 1.4rem; margin-bottom: 8px; display: block; color: var(--amber); }
  .drop-zone-text { font-size: 0.72rem; color: var(--text-dim); line-height: 1.6; }
  .drop-zone-text span { color: var(--amber); text-decoration: underline; cursor: pointer; }
  .doc-item { display: flex; align-items: center; gap: 8px; padding: 8px 10px; background: var(--surface); border: 1px solid var(--border); border-radius: 3px; font-size: 0.7rem; animation: slideIn 0.2s ease; }
  .doc-item-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text); }
  .doc-item-chunks { color: var(--green); font-size: 0.62rem; white-space: nowrap; }
  .doc-remove { background: none; border: none; color: var(--text-dim); cursor: pointer; font-size: 0.8rem; padding: 0 2px; line-height: 1; }
  .doc-remove:hover { color: var(--red); }
  .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .stat-box { background: var(--surface); border: 1px solid var(--border); border-radius: 3px; padding: 10px; }
  .stat-val { font-size: 1.1rem; font-weight: 500; color: var(--amber); }
  .stat-label { font-size: 0.6rem; color: var(--text-dim); letter-spacing: 0.06em; text-transform: uppercase; margin-top: 2px; }
  .config-row { display: flex; align-items: center; justify-content: space-between; font-size: 0.68rem; color: var(--text-dim); margin-bottom: 6px; }
  .config-val { color: var(--amber); font-size: 0.68rem; }
  .slider { width: 100%; accent-color: var(--amber); cursor: pointer; margin-top: 2px; }
  .main { display: flex; flex-direction: column; padding: 0 0 0 20px; min-height: 0; }
  .chat-area { flex: 1; overflow-y: auto; padding: 20px 0; display: flex; flex-direction: column; gap: 20px; }
  .empty-state { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; color: var(--text-dim); text-align: center; padding: 40px; }
  .empty-icon { font-size: 2.5rem; opacity: 0.4; }
  .empty-title { font-family: var(--serif); font-size: 1.1rem; font-style: italic; color: var(--text-dim); }
  .empty-sub { font-size: 0.7rem; max-width: 280px; line-height: 1.7; }
  .msg { display: flex; flex-direction: column; gap: 4px; animation: fadeUp 0.2s ease; }
  .msg-user { align-items: flex-end; }
  .msg-assistant { align-items: flex-start; }
  .msg-role { font-size: 0.6rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-dim); padding: 0 4px; }
  .msg-bubble { max-width: 85%; padding: 12px 16px; border-radius: 4px; font-size: 0.82rem; line-height: 1.75; }
  .msg-user .msg-bubble { background: #1a1a0e; border: 1px solid var(--amber-dim); color: var(--text-bright); }
  .msg-assistant .msg-bubble { background: var(--surface); border: 1px solid var(--border); color: var(--text); width: 100%; max-width: 100%; }
  .cite-inline { display: inline-block; background: #1a2a1a; border: 1px solid #3a5a3a; color: var(--green); font-size: 0.6rem; padding: 1px 5px; border-radius: 2px; cursor: pointer; margin: 0 1px; vertical-align: middle; font-weight: 500; letter-spacing: 0.04em; transition: background 0.1s; }
  .cite-inline:hover { background: #243a24; }
  .citation-cards { margin-top: 12px; display: flex; flex-direction: column; gap: 6px; }
  .citation-card { border: 1px solid var(--border); border-left: 3px solid var(--green); border-radius: 3px; padding: 8px 12px; font-size: 0.7rem; background: #0f1a0f; animation: slideIn 0.15s ease; }
  .citation-card-header { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
  .citation-num { color: var(--green); font-weight: 600; font-size: 0.65rem; }
  .citation-source { color: var(--text-dim); font-size: 0.62rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .citation-text { color: var(--text); line-height: 1.6; font-size: 0.7rem; }
  .processing-bar { display: flex; align-items: center; gap: 8px; font-size: 0.7rem; color: var(--amber); padding: 8px 0; }
  .dot-anim { display: flex; gap: 3px; }
  .dot { width: 4px; height: 4px; background: var(--amber); border-radius: 50%; animation: dotBounce 1s infinite; }
  .dot:nth-child(2) { animation-delay: 0.15s; }
  .dot:nth-child(3) { animation-delay: 0.3s; }
  .input-area { padding: 16px 0 20px; border-top: 1px solid var(--border); display: flex; gap: 10px; align-items: flex-end; }
  .input-wrap { flex: 1; background: var(--surface); border: 1px solid var(--border2); border-radius: 4px; display: flex; align-items: flex-end; transition: border-color 0.15s; }
  .input-wrap:focus-within { border-color: var(--amber-dim); }
  .chat-input { flex: 1; background: none; border: none; outline: none; color: var(--text-bright); font-family: var(--mono); font-size: 0.8rem; padding: 12px 14px; resize: none; line-height: 1.5; max-height: 120px; overflow-y: auto; }
  .chat-input::placeholder { color: var(--text-dim); }
  .send-btn { background: none; border: none; color: var(--amber-dim); cursor: pointer; padding: 10px 14px; font-size: 1rem; transition: color 0.15s, transform 0.1s; line-height: 1; }
  .send-btn:hover:not(:disabled) { color: var(--amber); transform: translateX(2px); }
  .send-btn:disabled { color: var(--border2); cursor: default; }
  .clear-btn { background: none; border: 1px solid var(--border); color: var(--text-dim); cursor: pointer; padding: 10px 12px; font-family: var(--mono); font-size: 0.68rem; border-radius: 4px; transition: all 0.15s; white-space: nowrap; }
  .clear-btn:hover { border-color: var(--red); color: var(--red); }
  .error-msg { background: #1a0f0f; border: 1px solid #4a2020; border-left: 3px solid var(--red); color: #e08080; font-size: 0.72rem; padding: 10px 14px; border-radius: 3px; max-width: 85%; }
  @keyframes slideIn { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes fadeUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes dotBounce { 0%, 80%, 100% { transform: translateY(0); } 40% { transform: translateY(-5px); } }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
`;


export default function RAGPipeline() {
  const [docs, setDocs] = useState([]);
  const [topK, setTopK] = useState(5);
  const [chunkSize] = useState(400); // chunking now happens server-side; kept for display only
  const [selectedModel, setSelectedModel] = useState("llama-3.3-70b-versatile");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [processing, setProcessing] = useState(null);
  const [agentStatus, setAgentStatus] = useState("");
  const fileInputRef = useRef(null);
  const chatRef = useRef(null);

  const totalChunks = docs.reduce((sum, doc) => sum + doc.chunks, 0);

  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages, loading, processing]);

  const processFile = useCallback(async (file) => {
    setProcessing({ name: file.name, stage: "Uploading & indexing (FAISS + BM25)" });

    try {
      const result = await uploadDocument(file);
      setDocs((prev) => {
        const filtered = prev.filter((doc) => doc.name !== file.name);
        return [...filtered, { name: file.name, chunks: result.chunks, size: result.size }];
      });
    } catch (error) {
      setMessages((prev) => [...prev, {
        role: "error",
        content: `Failed to process "${file.name}": ${error.message}`,
      }]);
    } finally {
      setProcessing(null);
    }
  }, []);

  const handleFiles = useCallback(async (files) => {
    for (const file of Array.from(files)) {
      const isSupported = file.type === "text/plain"
        || file.type === "application/pdf"
        || file.name.endsWith(".md")
        || file.name.endsWith(".txt");
      if (isSupported) await processFile(file);
    }
  }, [processFile]);

  const onDrop = useCallback((event) => {
    event.preventDefault();
    setDragging(false);
    handleFiles(event.dataTransfer.files);
  }, [handleFiles]);

  const removeDoc = async (name) => {
    setDocs((prev) => prev.filter((doc) => doc.name !== name));
    try {
      await deleteDocument(name);
    } catch (error) {
      setMessages((prev) => [...prev, {
        role: "error",
        content: `Failed to remove "${name}": ${error.message}`,
      }]);
    }
  };

  const handleSend = async () => {
    const q = input.trim();
    if (!q || loading) return;

    if (docs.length === 0) {
      setMessages((prev) => [...prev, {
        role: "error",
        content: "Please upload at least one document before asking questions.",
      }]);
      return;
    }

    const historySource = messages;
    const assistantId = crypto.randomUUID();
    setInput("");
    setLoading(true);
    setMessages((prev) => [
      ...prev,
      { role: "user", content: q },
      { id: assistantId, role: "assistant", content: "", citations: [], retrievedChunks: [] },
    ]);

    try {
      setAgentStatus("Agent is reasoning and searching documents...");

      const history = historySource
        .filter((message) => message.role === "user" || message.role === "assistant")
        .slice(-8)
        .map((message) => ({ role: message.role, content: message.content }));

      await chatWithAgent(
        { question: q, history, model: selectedModel, topK, maxTokens: 1000 },
        (result) => {
          setMessages((prev) => prev.map((message) =>
            message.id === assistantId
              ? {
                  ...message,
                  content: result.answer,
                  citations: result.citations || [],
                  retrievedChunks: result.retrieved_chunks || [],
                }
              : message
          ));
        }
      );
    } catch (error) {
      setMessages((prev) => prev
        .filter((message) => message.id !== assistantId)
        .concat({ role: "error", content: `API error: ${error.message}` }));
    } finally {
      setAgentStatus("");
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  function renderWithCitations(text, citations) {
    return text.split(/(\[\d+\])/g).map((part, i) => {
      const match = part.match(/^\[(\d+)\]$/);
      if (!match) return <span key={i}>{part}</span>;

      const num = Number(match[1]);
      const cite = citations?.find((citation) => citation.num === num);
      return (
        <span key={i} className="cite-inline" title={cite?.text || ""}>
          [{num}]
        </span>
      );
    });
  }

  const isEmpty = messages.length === 0 && !loading;
  const busyLabel = processing ? `${processing.stage}: ${processing.name}` : agentStatus;

  return (
    <>
      <style>{STYLES}</style>
      <div className="app">
        <header className="header">
          <h1 className="header-title">{APP_TITLE}</h1>
          <span className="header-badge">Hybrid RAG</span>
          <span className="header-sub">FAISS + BM25 + LangChain ReAct agent</span>
        </header>

        <div className="layout">
          <aside className="sidebar">
            <div>
              <div className="section-label">Documents</div>
              <div
                className={`drop-zone${dragging ? " dragging" : ""}`}
                onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".txt,.md,.pdf"
                  multiple
                  onChange={(event) => handleFiles(event.target.files)}
                />
                <span className="drop-zone-icon">+</span>
                <div className="drop-zone-text">
                  Drop <b>.txt</b>, <b>.md</b>, or <b>.pdf</b><br />
                  or <span>browse files</span>
                </div>
              </div>
              {busyLabel && (
                <div className="processing-bar" style={{ marginTop: 8 }}>
                  <div className="dot-anim">
                    <div className="dot" /><div className="dot" /><div className="dot" />
                  </div>
                  {busyLabel}
                </div>
              )}
            </div>

            {docs.length > 0 && (
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {docs.map((doc) => (
                  <div key={doc.name} className="doc-item">
                    <span style={{ fontSize: "0.9rem" }}>#</span>
                    <span className="doc-item-name" title={doc.name}>{doc.name}</span>
                    <span className="doc-item-chunks">{doc.chunks}c</span>
                    <button className="doc-remove" onClick={() => removeDoc(doc.name)}>x</button>
                  </div>
                ))}
              </div>
            )}

            {docs.length > 0 && (
              <div>
                <div className="section-label">Index</div>
                <div className="stats-grid">
                  <div className="stat-box"><div className="stat-val">{docs.length}</div><div className="stat-label">Docs</div></div>
                  <div className="stat-box"><div className="stat-val">{totalChunks}</div><div className="stat-label">Chunks</div></div>
                  <div className="stat-box"><div className="stat-val">{chunkSize}</div><div className="stat-label">Chunk sz</div></div>
                  <div className="stat-box"><div className="stat-val">{topK}</div><div className="stat-label">Top-K</div></div>
                </div>
              </div>
            )}

            <div>
              <div className="section-label">Retrieval Config</div>
              <div className="config-row"><span>Top-K chunks</span><span className="config-val">{topK}</span></div>
              <input type="range" min={1} max={10} value={topK} onChange={(event) => setTopK(Number(event.target.value))} className="slider" />
              <div style={{ fontSize: "0.6rem", color: "var(--text-dim)", marginTop: 6, lineHeight: 1.6 }}>
                Chunking (400 words, 20% overlap), FAISS indexing, and BM25 indexing all run server-side on upload.
              </div>
            </div>

            <div>
              <div className="section-label">Model</div>
              <select
                className="chat-input"
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
                style={{
                  width: "100%",
                  background: "var(--surface)",
                  border: "1px solid var(--border2)",
                  borderRadius: 4,
                  color: "var(--text-bright)",
                  padding: "9px 10px",
                  maxHeight: "none",
                }}
              >
                {MODEL_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label} ({option.note})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <div className="section-label">Method</div>
              <div style={{ fontSize: "0.65rem", color: "var(--text-dim)", lineHeight: 1.8 }}>
                FAISS dense + BM25 lexical retrieval<br />
                Reciprocal Rank Fusion<br />
                LangChain ReAct agent<br />
                Per-session vector-store isolation<br />
                FastAPI backend
              </div>
            </div>
          </aside>

          <main className="main">
            <div className="chat-area" ref={chatRef}>
              {isEmpty ? (
                <div className="empty-state">
                  <span className="empty-icon">[]</span>
                  <p className="empty-title">Upload documents to begin</p>
                  <p className="empty-sub">
                    Add .txt, .md, or .pdf files. The pipeline extracts, chunks, embeds, and retrieves passages for grounded answers.
                  </p>
                </div>
              ) : (
                messages.map((msg, i) => {
                  if (msg.role === "error") return <div key={i} className="error-msg">{msg.content}</div>;
                  return (
                    <div key={msg.id || i} className={`msg msg-${msg.role}`}>
                      <span className="msg-role">{msg.role === "user" ? "you" : "assistant"}</span>
                      <div className="msg-bubble">
                        <div style={{ whiteSpace: "pre-wrap" }}>
                          {msg.role === "assistant" ? renderWithCitations(msg.content, msg.citations) : msg.content}
                        </div>
                        {msg.citations?.length > 0 && (
                          <div className="citation-cards">
                            {msg.citations.map((citation) => (
                              <div key={citation.num} className="citation-card">
                                <div className="citation-card-header">
                                  <span className="citation-num">[{citation.num}]</span>
                                  <span className="citation-source">{citation.source}</span>
                                </div>
                                <div className="citation-text">
                                  {citation.text?.slice(0, 280)}{citation.text?.length > 280 ? "..." : ""}
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })
              )}
              {loading && messages.at(-1)?.role !== "assistant" && (
                <div className="msg msg-assistant">
                  <span className="msg-role">assistant</span>
                  <div className="msg-bubble">
                    <div className="processing-bar">
                      <div className="dot-anim">
                        <div className="dot" /><div className="dot" /><div className="dot" />
                      </div>
                      Retrieving and generating...
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="input-area">
              <div className="input-wrap">
                <textarea
                  className="chat-input"
                  placeholder={docs.length === 0 ? "Upload a document first..." : "Ask anything about your documents... (Enter to send)"}
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={handleKeyDown}
                  rows={1}
                  disabled={loading}
                />
                <button className="send-btn" onClick={handleSend} disabled={loading || !input.trim() || docs.length === 0}>
                  -&gt;
                </button>
              </div>
              {messages.length > 0 && (
                <button className="clear-btn" onClick={() => setMessages([])}>
                  clear
                </button>
              )}
            </div>
          </main>
        </div>
      </div>
    </>
  );
}
