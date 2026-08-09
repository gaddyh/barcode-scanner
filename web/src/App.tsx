import { useRef, useState } from "react";
import {
  scanBarcode,
  submitSessionImage,
  selectCandidate,
  submitFeedback,
  type ScanResponse,
  type SessionResult,
} from "./api";
import { useHashRoute } from "./router";
import { AdminApp } from "./admin/AdminApp";

type Source = "camera" | "gallery";
type Mode = "session" | "scanner";

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

export default function App() {
  const route = useHashRoute();
  const [file, setFile] = useState<File | null>(null);
  const [source, setSource] = useState<Source | null>(null);
  const [mode, setMode] = useState<Mode>("session");
  const [loading, setLoading] = useState(false);
  const [scanResult, setScanResult] = useState<ScanResponse | null>(null);
  const [sessionResult, setSessionResult] = useState<SessionResult | null>(null);
  const [totalMs, setTotalMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [feedbackError, setFeedbackError] = useState<string | null>(null);

  const cameraInputRef = useRef<HTMLInputElement>(null);
  const galleryInputRef = useRef<HTMLInputElement>(null);

  if (route === "#/admin") return <AdminApp />;

  function handleFile(e: React.ChangeEvent<HTMLInputElement>, src: Source) {
    const f = e.target.files?.[0] ?? null;
    if (!f) return;
    setFile(f);
    setSource(src);
    setScanResult(null);
    setSessionResult(null);
    setTotalMs(null);
    setError(null);
    setFeedbackSent(false);
    setFeedbackError(null);
  }

  async function onAnalyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setScanResult(null);
    setSessionResult(null);
    setTotalMs(null);
    setFeedbackSent(false);
    setFeedbackError(null);
    const t0 = performance.now();
    try {
      if (mode === "scanner") {
        const res = await scanBarcode(file);
        setScanResult(res);
      } else {
        const res = await submitSessionImage(file);
        setSessionResult(res);
      }
      setTotalMs(performance.now() - t0);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function onSelectCandidate(barcodeValue: string) {
    setLoading(true);
    setError(null);
    try {
      const res = await selectCandidate(barcodeValue);
      setSessionResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function sendFeedback(correct: boolean) {
    if (!traceId) return;
    setFeedbackError(null);
    try {
      await submitFeedback(traceId, correct);
      setFeedbackSent(true);
    } catch (err) {
      setFeedbackError(err instanceof Error ? err.message : String(err));
    }
  }

  const traceId = scanResult?.trace_id ?? null;

  const sessionActive = sessionResult?.status === "active";
  const sessionComplete = sessionResult?.status === "complete";
  const needsSelection = sessionResult?.status === "needs_user_selection";

  return (
    <div style={styles.container}>
      <h1 style={styles.h1}>Barcode Scanner</h1>

      {/* Mode toggle */}
      <div style={styles.toggleRow}>
        <button
          onClick={() => setMode("session")}
          style={{
            ...styles.toggleBtn,
            ...(mode === "session" ? styles.toggleActive : {}),
          }}
        >
          Session (multi-photo)
        </button>
        <button
          onClick={() => setMode("scanner")}
          style={{
            ...styles.toggleBtn,
            ...(mode === "scanner" ? styles.toggleActive : {}),
          }}
        >
          Scanner only
        </button>
      </div>

      <div style={styles.inputRow}>
        <label style={styles.button}>
          Take photo
          <input
            ref={cameraInputRef}
            hidden
            type="file"
            accept="image/*"
            capture="environment"
            onChange={(e) => handleFile(e, "camera")}
          />
        </label>
        <label style={styles.button}>
          Choose existing photo
          <input
            ref={galleryInputRef}
            hidden
            type="file"
            accept="image/*"
            onChange={(e) => handleFile(e, "gallery")}
          />
        </label>
      </div>

      {file && (
        <div style={styles.fileInfo}>
          <div><strong>Source:</strong> {source}</div>
          <div><strong>Filename:</strong> {file.name}</div>
          <div><strong>Client size:</strong> {formatBytes(file.size)} ({file.size} B)</div>
        </div>
      )}

      <button
        onClick={onAnalyze}
        disabled={!file || loading}
        style={{
          ...styles.analyze,
          opacity: !file || loading ? 0.5 : 1,
        }}
      >
        {loading ? "Analyzing…" : `Analyze (${mode === "session" ? "session" : "scanner"})`}
      </button>

      {error && (
        <div style={styles.error}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Scanner-only result */}
      {scanResult && (
        <div style={styles.results}>
          <h2 style={styles.h2}>Result</h2>
          <Row label="Upload ID" value={scanResult.upload_id ?? "—"} />
          <Row label="Trace ID" value={scanResult.trace_id ?? "—"} />
          <Row label="Status" value={scanResult.status} />
          <Row label="Count" value={String(scanResult.count)} />
          <Row label="Dimensions" value={`${scanResult.image_width} × ${scanResult.image_height}`} />
          <Row label="File size" value={`${formatBytes(scanResult.upload_bytes)} (${scanResult.upload_bytes} B)`} />
          <Row label="Server scan" value={`${scanResult.elapsed_ms} ms`} />
          <Row label="Total request" value={totalMs != null ? `${Math.round(totalMs)} ms` : "—"} />
          <h3 style={styles.h3}>Barcodes</h3>
          {scanResult.barcodes.length === 0 ? (
            <p style={styles.muted}>None decoded.</p>
          ) : (
            <ol style={styles.list}>
              {scanResult.barcodes.map((b, i) => (
                <li key={i} style={styles.listItem}>
                  <strong>{b.value}</strong> ({b.format})
                </li>
              ))}
            </ol>
          )}
          {traceId && !feedbackSent && (
            <FeedbackRow onFeedback={sendFeedback} feedbackError={feedbackError} />
          )}
          {feedbackSent && <p style={styles.feedbackDone}>Feedback recorded.</p>}
        </div>
      )}

      {/* Session result */}
      {sessionResult && (
        <div style={styles.results}>
          <h2 style={styles.h2}>Session</h2>
          <Row label="Session ID" value={sessionResult.session_id} />
          <Row
            label="Status"
            value={sessionResult.status}
            highlight={sessionComplete ? "#16a34a" : needsSelection ? "#f59e0b" : undefined}
          />
          <Row label="Found" value={`${sessionResult.found_count} / ${sessionResult.expected_count}`} />
          <Row label="Missing" value={String(sessionResult.missing_count)} />
          <Row label="Images" value={String(sessionResult.image_count)} />
          {totalMs != null && <Row label="Last request" value={`${Math.round(totalMs)} ms`} />}

          {/* Session message */}
          {sessionResult.message && (
            <div style={styles.sessionMessage}>
              {sessionResult.message}
            </div>
          )}

          {/* Needs user selection — show candidate buttons */}
          {needsSelection && sessionResult.candidates.length > 0 && (
            <div style={styles.candidates}>
              <h3 style={styles.h3}>Select a barcode to add:</h3>
              <div style={styles.candidateRow}>
                {sessionResult.candidates.map((c, i) => (
                  <button
                    key={i}
                    onClick={() => onSelectCandidate(c.barcode_value)}
                    disabled={loading}
                    style={styles.candidateBtn}
                  >
                    {c.barcode_value}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Active — prompt for more photos */}
          {sessionActive && sessionResult.missing_count > 0 && (
            <div style={styles.promptMore}>
              📸 Send another photo of the missing box(es).
            </div>
          )}

          {/* Complete — show items */}
          {sessionComplete && (
            <div style={styles.completeBadge}>
              ✅ All {sessionResult.expected_count} boxes scanned!
            </div>
          )}

          {/* Items list */}
          {sessionResult.items.length > 0 && (
            <>
              <h3 style={styles.h3}>Scanned barcodes ({sessionResult.items.length})</h3>
              <ol style={styles.list}>
                {sessionResult.items.map((item, i) => (
                  <li key={i} style={styles.listItem}>
                    <strong>{item.barcode_value}</strong>
                    {item.barcode_format ? ` (${item.barcode_format})` : ""}
                    {item.label_index != null && ` — label ${item.label_index}`}
                  </li>
                ))}
              </ol>
            </>
          )}

          {/* Missing labels */}
          {sessionResult.missing.length > 0 && !sessionComplete && (
            <>
              <h3 style={styles.h3}>Missing labels ({sessionResult.missing.length})</h3>
              <ul style={styles.list}>
                {sessionResult.missing.map((m, i) => (
                  <li key={i} style={styles.listItem}>
                    Label {m.label_index ?? "?"} — {m.status}
                    {m.resolved ? " (resolved)" : ""}
                  </li>
                ))}
              </ul>
            </>
          )}

          {/* Feedback (only for complete sessions) */}
          {sessionComplete && !feedbackSent && (
            <FeedbackRow onFeedback={sendFeedback} feedbackError={feedbackError} />
          )}
          {feedbackSent && <p style={styles.feedbackDone}>Feedback recorded.</p>}
        </div>
      )}

      <div style={{ marginTop: 32, paddingTop: 16, borderTop: "1px solid #e2e8f0" }}>
        <a href="#/admin" style={{ color: "#3b82f6", textDecoration: "none", fontSize: 14, fontWeight: 500 }}>
          Admin dashboard →
        </a>
      </div>
    </div>
  );
}

function FeedbackRow({
  onFeedback,
  feedbackError,
}: {
  onFeedback: (correct: boolean) => void;
  feedbackError: string | null;
}) {
  return (
    <div style={styles.feedback}>
      <p style={styles.feedbackQ}>Did the scanner find all barcodes correctly?</p>
      <div style={styles.feedbackRow}>
        <button onClick={() => onFeedback(true)} style={styles.feedbackBtn}>
          Correct
        </button>
        <button onClick={() => onFeedback(false)} style={styles.feedbackBtn}>
          Incorrect
        </button>
      </div>
      {feedbackError && <p style={styles.error}>Feedback error: {feedbackError}</p>}
    </div>
  );
}

function Row({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: string;
}) {
  return (
    <div style={styles.row}>
      <span style={styles.rowLabel}>{label}:</span>
      <span style={{ ...styles.rowValue, ...(highlight ? { color: highlight, fontWeight: 600 } : {}) }}>
        {value}
      </span>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    maxWidth: 480,
    margin: "0 auto",
    padding: 16,
    fontFamily: "system-ui, -apple-system, sans-serif",
    color: "#1e293b",
  },
  h1: { fontSize: 24, fontWeight: 700, marginBottom: 16 },
  h2: { fontSize: 20, fontWeight: 600, marginBottom: 12 },
  h3: { fontSize: 16, fontWeight: 600, marginTop: 16, marginBottom: 8 },
  toggleRow: { display: "flex", gap: 8, marginBottom: 16 },
  toggleBtn: {
    flex: 1,
    padding: "8px 12px",
    border: "1px solid #cbd5e1",
    borderRadius: 8,
    background: "#fff",
    cursor: "pointer",
    fontSize: 14,
    fontWeight: 500,
  },
  toggleActive: { background: "#3b82f6", color: "#fff", borderColor: "#3b82f6" },
  inputRow: { display: "flex", gap: 8, marginBottom: 16 },
  button: {
    flex: 1,
    padding: "10px 16px",
    background: "#f1f5f9",
    borderRadius: 8,
    textAlign: "center",
    cursor: "pointer",
    fontSize: 14,
    fontWeight: 500,
    border: "1px solid #cbd5e1",
  },
  fileInfo: {
    padding: 12,
    background: "#f8fafc",
    borderRadius: 8,
    marginBottom: 16,
    fontSize: 13,
    lineHeight: 1.6,
  },
  analyze: {
    width: "100%",
    padding: "12px 16px",
    background: "#3b82f6",
    color: "#fff",
    border: "none",
    borderRadius: 8,
    fontSize: 16,
    fontWeight: 600,
    cursor: "pointer",
    marginBottom: 16,
  },
  results: {
    padding: 16,
    background: "#fff",
    border: "1px solid #e2e8f0",
    borderRadius: 8,
    marginBottom: 16,
  },
  row: { display: "flex", justifyContent: "space-between", padding: "4px 0", fontSize: 14 },
  rowLabel: { color: "#64748b", fontWeight: 500 },
  rowValue: { color: "#1e293b", fontWeight: 500 },
  sessionMessage: {
    padding: 12,
    background: "#fef3c7",
    borderRadius: 8,
    marginTop: 12,
    marginBottom: 12,
    fontSize: 14,
    fontWeight: 500,
    color: "#92400e",
  },
  candidates: { marginTop: 12, marginBottom: 12 },
  candidateRow: { display: "flex", flexWrap: "wrap", gap: 8 },
  candidateBtn: {
    padding: "8px 14px",
    background: "#3b82f6",
    color: "#fff",
    border: "none",
    borderRadius: 8,
    fontSize: 14,
    fontWeight: 600,
    cursor: "pointer",
  },
  promptMore: {
    padding: 12,
    background: "#dbeafe",
    borderRadius: 8,
    marginTop: 12,
    marginBottom: 12,
    fontSize: 14,
    fontWeight: 500,
    color: "#1e40af",
  },
  completeBadge: {
    padding: 12,
    background: "#dcfce7",
    borderRadius: 8,
    marginTop: 12,
    marginBottom: 12,
    fontSize: 16,
    fontWeight: 600,
    color: "#166534",
    textAlign: "center",
  },
  list: { margin: "8px 0", paddingLeft: 20, fontSize: 14, lineHeight: 1.8 },
  listItem: { marginBottom: 4 },
  muted: { color: "#94a3b8", fontSize: 14 },
  error: { color: "#dc2626", fontSize: 14, padding: 8, background: "#fef2f2", borderRadius: 8 },
  feedback: { marginTop: 16, paddingTop: 16, borderTop: "1px solid #e2e8f0" },
  feedbackQ: { fontSize: 14, marginBottom: 8 },
  feedbackRow: { display: "flex", gap: 8 },
  feedbackBtn: {
    flex: 1,
    padding: "8px 16px",
    border: "1px solid #cbd5e1",
    borderRadius: 8,
    background: "#fff",
    cursor: "pointer",
    fontSize: 14,
    fontWeight: 500,
  },
  feedbackDone: { color: "#16a34a", fontSize: 14, fontWeight: 500 },
};
