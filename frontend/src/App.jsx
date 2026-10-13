
import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";
const MAX_FILE_SIZE = 2 * 1024 * 1024;
const ALLOWED_EXTENSIONS = [".txt", ".pdf", ".docx"];

const suggestions = [
  "How many days of annual leave do employees get?",
  "How early must employees submit leave requests?",
  "Who approves or rejects leave requests?",
  "Can unused annual leave automatically carry over?",
  "What is the company's parental leave policy?",
];

function getErrorMessage(data, fallback) {
  if (typeof data?.detail === "string") return data.detail;

  if (Array.isArray(data?.detail)) {
    return data.detail.map((item) => item.msg).join(", ");
  }

  return data?.message || fallback;
}

async function readResponse(response) {
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      getErrorMessage(data, `Request failed (${response.status})`)
    );
  }

  return data;
}

function getFileType(filename = "") {
  const extension = filename.split(".").pop()?.toLowerCase();

  if (extension === "pdf") return "PDF";
  if (extension === "docx") return "DOCX";
  return "TXT";
}

function formatFileSize(bytes) {
  if (typeof bytes !== "number") return "Document";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [validationError, setValidationError] = useState("");

  const [activeTab, setActiveTab] = useState("chat");
  const [documents, setDocuments] = useState([]);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [documentError, setDocumentError] = useState("");
  const [documentNotice, setDocumentNotice] = useState("");
  const [uploading, setUploading] = useState(false);
  const [deletingFilename, setDeletingFilename] = useState("");

  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  async function handleSubmit(event) {
    event.preventDefault();

    const currentQuestion = question.trim();

    if (loading) return;

    if (!currentQuestion) {
      setValidationError("Please enter a question before clicking Ask.");
      return;
    }

    setValidationError("");
    setMessages((previous) => [
      ...previous,
      {
        id: crypto.randomUUID(),
        role: "user",
        text: currentQuestion,
      },
    ]);

    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: currentQuestion, top_k: 3 }),
      });

      const data = await readResponse(response);

      setMessages((previous) => [
        ...previous,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: data.answer,
          sources: data.sources || [],
        },
      ]);
    } catch (error) {
      setMessages((previous) => [
        ...previous,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: `Unable to get an answer: ${error.message}. Check that FastAPI and Ollama are running.`,
          sources: [],
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function selectSuggestion(suggestion) {
    setQuestion(suggestion);
    setValidationError("");
  }

  function handleQuestionChange(event) {
    setQuestion(event.target.value);
    setValidationError("");
  }

  async function loadDocuments() {
    setDocumentsLoading(true);
    setDocumentError("");

    try {
      const response = await fetch(`${API_URL}/documents`);
      const data = await readResponse(response);
      const items = Array.isArray(data)
        ? data
        : data.documents || data.files || [];

      setDocuments(
        items.map((item) => {
          if (typeof item === "string") {
            return { filename: item, indexed: null };
          }

          return {
            ...item,
            filename:
              item.filename || item.name || item.source || "Unnamed document",
            indexed:
              item.indexed ??
              item.is_indexed ??
              item.is_indexed_in_vector_store ??
              null,
          };
        })
      );
    } catch (error) {
      setDocumentError(
        `Could not load documents: ${error.message}. Check that FastAPI is running.`
      );
    } finally {
      setDocumentsLoading(false);
    }
  }

  useEffect(() => {
    if (activeTab === "documents") {
      loadDocuments();
    }
  }, [activeTab]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleUpload(event) {
    event.preventDefault();

    const file = fileInputRef.current?.files?.[0];

    if (!file) {
      setDocumentError("Choose a document before uploading.");
      return;
    }

    const extension = `.${file.name.split(".").pop()}`.toLowerCase();

    if (!ALLOWED_EXTENSIONS.includes(extension)) {
      setDocumentError("Only PDF, DOCX, and TXT files are supported.");
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      setDocumentError("The file must be 2 MB or smaller.");
      return;
    }

    setUploading(true);
    setDocumentError("");
    setDocumentNotice("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await readResponse(response);

      setDocumentNotice(data.message || `${file.name} uploaded successfully.`);

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }

      await loadDocuments();
    } catch (error) {
      setDocumentError(`Upload failed: ${error.message}`);
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(filename) {
    const confirmed = window.confirm(
      `Delete "${filename}" and its indexed chunks? This cannot be undone.`
    );

    if (!confirmed) return;

    setDeletingFilename(filename);
    setDocumentError("");
    setDocumentNotice("");

    try {
      const response = await fetch(
        `${API_URL}/documents/${encodeURIComponent(filename)}`,
        { method: "DELETE" }
      );

      const data = await readResponse(response);

      setDocumentNotice(data.message || `${filename} deleted successfully.`);
      await loadDocuments();
    } catch (error) {
      setDocumentError(`Delete failed: ${error.message}`);
    } finally {
      setDeletingFilename("");
    }
  }

  const indexedCount = documents.filter(
    (document) => document.indexed === true
  ).length;

  return (
    <main className="workspace">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-mark">EK</div>
          <div>
            <h1>Knowbase</h1>
            <p>Enterprise intelligence</p>
          </div>
        </div>

        <div className="workspace-label">WORKSPACE</div>

        <nav className="sidebar-nav" aria-label="Main navigation">
          <button
            type="button"
            className={`sidebar-link ${activeTab === "chat" ? "active" : ""}`}
            onClick={() => setActiveTab("chat")}
          >
            <span className="nav-icon">✳</span>
            <span>AI Assistant</span>
            <span className="nav-arrow">↗</span>
          </button>

          <button
            type="button"
            className={`sidebar-link ${
              activeTab === "documents" ? "active" : ""
            }`}
            onClick={() => setActiveTab("documents")}
          >
            <span className="nav-icon">▤</span>
            <span>Knowledge Library</span>
            <span className="nav-count">{documents.length || ""}</span>
          </button>
        </nav>

        <div className="sidebar-divider" />

        <div className="sidebar-section-title">YOUR KNOWLEDGE</div>

        <div className="sidebar-stat">
          <span className="stat-icon">⌘</span>
          <div>
            <strong>{documents.length} documents</strong>
            <span>In your library</span>
          </div>
        </div>

        <div className="sidebar-stat">
          <span className="stat-icon green">✓</span>
          <div>
            <strong>{indexedCount} indexed</strong>
            <span>Ready for retrieval</span>
          </div>
        </div>

        <div className="sidebar-spacer" />

        <div className="sidebar-status">
          <span className="status-dot" />
          <div>
            <strong>Local AI engine</strong>
            <span>Private workspace</span>
          </div>
          <span className="status-check">✓</span>
        </div>

        <div className="sidebar-footer">
          <div className="footer-avatar">EK</div>
          <div>
            <strong>Knowledge Workspace</strong>
            <span>Personal environment</span>
          </div>
        </div>
      </aside>

      <section className="main-panel">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="breadcrumb-separator">/</span>
            <strong>
              {activeTab === "chat" ? "AI Assistant" : "Knowledge Library"}
            </strong>
          </div>

          <div className="topbar-right">
            <div className="connection-pill">
              <span className="status-dot" />
              Local RAG
            </div>
            <div className="topbar-avatar">EK</div>
          </div>
        </header>

        {activeTab === "chat" ? (
          <section className="chat-page">
            <div className="page-heading chat-heading">
              <div>
                <div className="eyebrow">
                  <span className="eyebrow-dot" />
                  YOUR KNOWLEDGE, CONNECTED
                </div>
                <h2>Ask your knowledge base<span>.</span></h2>
                <p>
                  Get grounded answers from your company documents, with
                  evidence you can verify.
                </p>
              </div>

              <div className="heading-decoration" aria-hidden="true">
                <span className="decoration-orbit orbit-one" />
                <span className="decoration-orbit orbit-two" />
                <span className="decoration-core">✳</span>
              </div>
            </div>

            <div className="chat-canvas">
              {messages.length === 0 ? (
                <div className="welcome-panel">
                  <div className="welcome-symbol">✳</div>
                  <h3>What would you like to know?</h3>
                  <p>
                    Ask a question or choose one of the suggestions below to
                    get started.
                  </p>

                  <div className="suggestion-heading">
                    <span>TRY A SUGGESTED QUESTION</span>
                    <span className="suggestion-line" />
                  </div>

                  <div className="suggestions">
                    {suggestions.map((suggestion, index) => (
                      <button
                        type="button"
                        className="suggestion-card"
                        key={suggestion}
                        onClick={() => selectSuggestion(suggestion)}
                      >
                        <span className="suggestion-number">
                          {String(index + 1).padStart(2, "0")}
                        </span>
                        <span className="suggestion-text">{suggestion}</span>
                        <span className="suggestion-arrow">↗</span>
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="messages" aria-live="polite">
                  {messages.map((message) => (
                    <article
                      className={`message ${message.role}`}
                      key={message.id}
                    >
                      <div className="message-heading">
                        <div
                          className={`avatar ${
                            message.role === "assistant"
                              ? "assistant-avatar"
                              : "user-avatar"
                          }`}
                        >
                          {message.role === "assistant" ? "✳" : "Y"}
                        </div>
                        <div>
                          <strong>
                            {message.role === "assistant"
                              ? "Knowledge Assistant"
                              : "You"}
                          </strong>
                          <span>
                            {message.role === "assistant"
                              ? "AI response"
                              : "Your question"}
                          </span>
                        </div>
                      </div>

                      <div
                        className={`message-content ${
                          message.isError ? "error-message" : ""
                        }`}
                      >
                        <p>{message.text}</p>
                      </div>

                      {message.role === "assistant" &&
                        message.sources?.length > 0 && (
                          <details className="sources" open>
                            <summary>
                              <span>▤</span>
                              <strong>Evidence & sources</strong>
                              <span className="source-count">
                                {message.sources.length}
                              </span>
                            </summary>

                            <div className="source-list">
                              {message.sources.map((source, index) => (
                                <article
                                  className="source-card"
                                  key={
                                    source.chunk_id ??
                                    `${source.source}-${index}`
                                  }
                                >
                                  <div className="source-title">
                                    <span className="file-icon">
                                      {getFileType(source.source || "")}
                                    </span>
                                    <div>
                                      <strong>{source.source}</strong>
                                      {source.chunk_id && (
                                        <small>{source.chunk_id}</small>
                                      )}
                                    </div>
                                  </div>
                                  <p>{source.text}</p>
                                  {typeof source.distance === "number" && (
                                    <span className="source-distance">
                                      Retrieval distance:{" "}
                                      {source.distance.toFixed(4)}
                                    </span>
                                  )}
                                </article>
                              ))}
                            </div>
                          </details>
                        )}
                    </article>
                  ))}

                  {loading && (
                    <div className="message assistant">
                      <div className="message-heading">
                        <div className="avatar assistant-avatar">✳</div>
                        <div>
                          <strong>Knowledge Assistant</strong>
                          <span>Searching your documents</span>
                        </div>
                      </div>
                      <div className="loading-indicator">
                        <span />
                        <span />
                        <span />
                        <p>Finding evidence and preparing an answer…</p>
                      </div>
                    </div>
                  )}

                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            <div className="composer-wrap">
              <form className="composer" onSubmit={handleSubmit} noValidate>
                <div className="composer-leading">✳</div>
                <input
                  type="text"
                  value={question}
                  onChange={handleQuestionChange}
                  placeholder="Ask anything about your company knowledge..."
                  aria-label="Your question"
                  aria-invalid={Boolean(validationError)}
                  aria-describedby={
                    validationError ? "question-error" : undefined
                  }
                  disabled={loading}
                />
                <button type="submit" disabled={loading}>
                  {loading ? "Thinking…" : "Ask AI"}
                  <span>↑</span>
                </button>
              </form>

              {validationError && (
                <p id="question-error" className="error-message" role="alert">
                  {validationError}
                </p>
              )}

              <div className="composer-meta">
                <span>↳ Answers are grounded in your uploaded documents</span>
                <span>Enter to send</span>
              </div>
              <p className="disclaimer">
                AI can make mistakes. Verify important decisions against the
                original source documents.
              </p>
            </div>
          </section>
        ) : (
          <section className="document-page">
            <div className="page-heading library-heading">
              <div>
                <div className="eyebrow">
                  <span className="eyebrow-dot" />
                  KNOWLEDGE MANAGEMENT
                </div>
                <h2>Your knowledge library<span>.</span></h2>
                <p>
                  Manage the files your AI assistant uses to find answers.
                </p>
              </div>

              <button
                type="button"
                className="secondary-button"
                onClick={loadDocuments}
                disabled={documentsLoading}
              >
                <span>↻</span>
                {documentsLoading ? "Refreshing…" : "Refresh library"}
              </button>
            </div>

            <div className="library-stats">
              <div className="library-stat-card">
                <div className="library-stat-icon">▤</div>
                <div>
                  <span>Total documents</span>
                  <strong>{documents.length}</strong>
                </div>
              </div>
              <div className="library-stat-card">
                <div className="library-stat-icon green-icon">✓</div>
                <div>
                  <span>Indexed documents</span>
                  <strong>{indexedCount}</strong>
                </div>
              </div>
              <div className="library-stat-card">
                <div className="library-stat-icon purple-icon">✳</div>
                <div>
                  <span>Supported formats</span>
                  <strong>PDF · DOCX · TXT</strong>
                </div>
              </div>
            </div>

            <form className="upload-card" onSubmit={handleUpload}>
              <div className="upload-visual">
                <span>↑</span>
              </div>
              <div className="upload-copy">
                <h3>Add to your knowledge base</h3>
                <p>
                  Select a company document to upload and index for AI search.
                </p>
                <div className="upload-guidelines">
                  <span>PDF</span>
                  <span>DOCX</span>
                  <span>TXT</span>
                  <span>Maximum 2 MB</span>
                </div>
              </div>

              <div className="upload-actions">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.docx,.txt"
                  aria-label="Choose a document"
                  disabled={uploading}
                />
                <button type="submit" disabled={uploading}>
                  {uploading ? "Uploading…" : "↑ Upload document"}
                </button>
              </div>
            </form>

            {documentError && (
              <div className="document-alert error-alert" role="alert">
                <span>!</span>
                {documentError}
              </div>
            )}

            {documentNotice && (
              <div className="document-alert success-alert" role="status">
                <span>✓</span>
                {documentNotice}
              </div>
            )}

            <div className="document-list-heading">
              <div>
                <h3>All documents</h3>
                <p>
                  Your files and their indexing status
                </p>
              </div>
              <span className="document-total">
                {documents.length} {documents.length === 1 ? "file" : "files"}
              </span>
            </div>

            {documentsLoading && documents.length === 0 ? (
              <div className="document-empty">
                <div className="loading-spinner" />
                <h3>Loading your library…</h3>
                <p>Fetching documents from your local knowledge base.</p>
              </div>
            ) : documents.length === 0 ? (
              <div className="document-empty">
                <div className="empty-icon">▤</div>
                <h3>Your library is ready for its first document</h3>
                <p>
                  Upload a PDF, Word document, or text file to start asking
                  questions about your knowledge base.
                </p>
              </div>
            ) : (
              <div className="document-list">
                {documents.map((document) => (
                  <article className="document-row" key={document.filename}>
                    <div
                      className={`document-file-icon file-${getFileType(
                        document.filename
                      ).toLowerCase()}`}
                    >
                      {getFileType(document.filename)}
                    </div>

                    <div className="document-details">
                      <strong>{document.filename}</strong>
                      <span>
                        {formatFileSize(document.size_bytes)} ·{" "}
                        {document.indexed === true
                          ? "Indexed and available for search"
                          : document.indexed === false
                            ? "Not indexed"
                            : "Indexing status not provided by API"}
                      </span>
                    </div>

                    <span
                      className={`index-status ${
                        document.indexed === true
                          ? "indexed"
                          : document.indexed === false
                            ? "not-indexed"
                            : "status-unknown"
                      }`}
                    >
                      <span className="index-status-dot" />
                      {document.indexed === true
                        ? "Indexed"
                        : document.indexed === false
                          ? "Not indexed"
                          : "Unknown"}
                    </span>

                    <button
                      type="button"
                      className="delete-button"
                      onClick={() => handleDelete(document.filename)}
                      disabled={uploading || Boolean(deletingFilename)}
                      aria-label={`Delete ${document.filename}`}
                    >
                      {deletingFilename === document.filename
                        ? "Deleting…"
                        : "Delete"}
                    </button>
                  </article>
                ))}
              </div>
            )}

            <p className="document-footnote">
              Deleting a document also requests removal of its indexed chunks.
              Changes are reflected after the library refreshes.
            </p>
          </section>
        )}
      </section>
    </main>
  );
}

export default App;
