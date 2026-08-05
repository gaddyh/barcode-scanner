import { useRef, useState } from "react";
import {
  analyzeImage,
  scanBarcode,
  type AnalyzeResponse,
  type ScanResponse,
} from "./api";

type Source = "camera" | "gallery";
type Mode = "scanner" | "pipeline";

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [source, setSource] = useState<Source | null>(null);
  const [mode, setMode] = useState<Mode>("pipeline");
  const [loading, setLoading] = useState(false);
  const [scanResult, setScanResult] = useState<ScanResponse | null>(null);
  const [analyzeResult, setAnalyzeResult] = useState<AnalyzeResponse | null>(null);
  const [totalMs, setTotalMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const cameraInputRef = useRef<HTMLInputElement>(null);
  const galleryInputRef = useRef<HTMLInputElement>(null);

  function handleFile(e: React.ChangeEvent<HTMLInputElement>, src: Source) {
    const f = e.target.files?.[0] ?? null;
    if (!f) return;
    setFile(f);
    setSource(src);
    setScanResult(null);
    setAnalyzeResult(null);
    setTotalMs(null);
    setError(null);
  }

  async function onAnalyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setScanResult(null);
    setAnalyzeResult(null);
    setTotalMs(null);
    const t0 = performance.now();
    try {
      if (mode === "scanner") {
        const res = await scanBarcode(file);
        setScanResult(res);
      } else {
        const res = await analyzeImage(file);
        setAnalyzeResult(res);
      }
      setTotalMs(performance.now() - t0);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  const elapsedMs =
    scanResult?.elapsed_ms ?? analyzeResult?.elapsed_ms ?? null;
  const imageWidth =
    scanResult?.image_width ?? analyzeResult?.image_width ?? null;
  const imageHeight =
    scanResult?.image_height ?? analyzeResult?.image_height ?? null;
  const uploadBytes =
    scanResult?.upload_bytes ?? analyzeResult?.upload_bytes ?? null;
  const filename = scanResult?.filename ?? analyzeResult?.filename ?? null;
  const uploadId =
    scanResult?.upload_id ?? analyzeResult?.upload_id ?? null;
  const resultSource =
    scanResult?.source ?? analyzeResult?.source ?? null;

  return (
    <div style={styles.container}>
      <h1 style={styles.h1}>Barcode Scanner</h1>

      {/* Mode toggle */}
      <div style={styles.toggleRow}>
        <button
          onClick={() => setMode("pipeline")}
          style={{
            ...styles.toggleBtn,
            ...(mode === "pipeline" ? styles.toggleActive : {}),
          }}
        >
          Full pipeline (Gemini)
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
        {loading ? "Analyzing…" : `Analyze (${mode === "pipeline" ? "full pipeline" : "scanner"})`}
      </button>

      {error && (
        <div style={styles.error}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Common metadata */}
      {elapsedMs !== null && (
        <div style={styles.results}>
          <h2 style={styles.h2}>Result</h2>
          <Row label="Upload ID" value={uploadId ?? "—"} />
          <Row label="Source" value={resultSource ?? "—"} />
          <Row label="Filename" value={filename ?? "—"} />
          <Row label="Source" value={source ?? "—"} />
          <Row
            label="Dimensions"
            value={imageWidth && imageHeight ? `${imageWidth} × ${imageHeight}` : "—"}
          />
          <Row
            label="File size"
            value={uploadBytes !== null ? `${formatBytes(uploadBytes)} (${uploadBytes} B)` : "—"}
          />
          <Row label="Server scan" value={`${elapsedMs} ms`} />
          <Row label="Total request" value={totalMs != null ? `${Math.round(totalMs)} ms` : "—"} />

          {/* Scanner-only result */}
          {scanResult && (
            <ScannerResult result={scanResult} />
          )}

          {/* Full pipeline result */}
          {analyzeResult && (
            <PipelineResult result={analyzeResult} />
          )}
        </div>
      )}
    </div>
  );
}

function ScannerResult({ result }: { result: ScanResponse }) {
  return (
    <>
      <Row label="Status" value={result.status} />
      <Row label="Count" value={String(result.count)} />
      <h3 style={styles.h3}>Barcodes</h3>
      {result.barcodes.length === 0 ? (
        <p style={styles.muted}>None decoded.</p>
      ) : (
        <ol style={styles.ol}>
          {result.barcodes.map((b, i) => (
            <li key={i} style={styles.li}>
              <code>{b.value}</code>
              <span style={styles.fmt}> · {b.format}</span>
            </li>
          ))}
        </ol>
      )}
    </>
  );
}

function PipelineResult({ result }: { result: AnalyzeResponse }) {
  return (
    <>
      <Row label="Outcome" value={result.outcome} />
      <Row label="Audit available" value={String(result.audit_available)} />
      <Row label="Visible labels" value={String(result.summary.visible_label_count)} />
      <Row label="Found" value={String(result.summary.found_count)} />
      <Row label="Missing" value={String(result.summary.missing_count)} />
      <Row label="Unassigned" value={String(result.summary.unassigned_count)} />

      {result.error && (
        <div style={styles.error}>
          <strong>Error:</strong> {result.error.code}: {result.error.message}
        </div>
      )}

      {/* Annotated image */}
      {result.annotated_image_b64 && (
        <div style={styles.annotation}>
          <h3 style={styles.h3}>Annotated — missing regions</h3>
          {result.message && (
            <p style={styles.message}>{result.message}</p>
          )}
          <img
            src={`data:image/png;base64,${result.annotated_image_b64}`}
            alt="Annotated — red circles around missing barcode regions"
            style={styles.annotatedImg}
          />
        </div>
      )}

      {/* Found barcodes */}
      <h3 style={styles.h3}>Found ({result.found.length})</h3>
      {result.found.length === 0 ? (
        <p style={styles.muted}>None.</p>
      ) : (
        <ol style={styles.ol}>
          {result.found.map((f, i) => (
            <li key={i} style={styles.li}>
              <strong>Label {f.label_index}:</strong>{" "}
              <code>{f.barcode_value}</code>
              <span style={styles.fmt}> · {f.barcode_format}</span>
            </li>
          ))}
        </ol>
      )}

      {/* Missing labels */}
      {result.missing.length > 0 && (
        <>
          <h3 style={styles.h3}>Missing ({result.missing.length})</h3>
          <ol style={styles.ol}>
            {result.missing.map((m, i) => (
              <li key={i} style={styles.li}>
                <strong>Label {m.label_index}</strong>{" "}
                <span style={styles.fmt}>· {m.status}</span>
              </li>
            ))}
          </ol>
        </>
      )}

      {/* Unassigned barcodes */}
      {result.unassigned.length > 0 && (
        <>
          <h3 style={styles.h3}>Unassigned ({result.unassigned.length})</h3>
          <ol style={styles.ol}>
            {result.unassigned.map((u, i) => (
              <li key={i} style={styles.li}>
                <code>{u.barcode_value}</code>
                <span style={styles.fmt}> · {u.barcode_format}</span>
              </li>
            ))}
          </ol>
        </>
      )}
    </>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div style={styles.row}>
      <span style={styles.rowLabel}>{label}</span>
      <span style={styles.rowValue}>{value}</span>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    fontFamily: "system-ui, -apple-system, sans-serif",
    maxWidth: 480,
    margin: "0 auto",
    padding: "20px 16px",
    color: "#111",
  },
  h1: { fontSize: 22, margin: "0 0 12px" },
  toggleRow: { display: "flex", gap: 6, marginBottom: 16 },
  toggleBtn: {
    flex: 1,
    padding: "8px 6px",
    border: "1px solid #d0d0d0",
    background: "#fff",
    borderRadius: 8,
    fontSize: 13,
    fontWeight: 500,
    cursor: "pointer",
  },
  toggleActive: {
    background: "#007aff",
    color: "#fff",
    borderColor: "#007aff",
  },
  inputRow: { display: "flex", gap: 10, marginBottom: 16 },
  button: {
    flex: 1,
    display: "inline-block",
    textAlign: "center",
    padding: "12px 8px",
    background: "#007aff",
    color: "#fff",
    borderRadius: 10,
    fontSize: 15,
    fontWeight: 500,
    cursor: "pointer",
  },
  fileInfo: {
    background: "#f5f5f7",
    borderRadius: 10,
    padding: "10px 12px",
    fontSize: 14,
    lineHeight: 1.6,
    marginBottom: 16,
  },
  analyze: {
    width: "100%",
    padding: "14px",
    background: "#34c759",
    color: "#fff",
    border: "none",
    borderRadius: 10,
    fontSize: 16,
    fontWeight: 600,
    cursor: "pointer",
  },
  error: {
    marginTop: 16,
    padding: "10px 12px",
    background: "#fff3f3",
    border: "1px solid #f0c0c0",
    borderRadius: 10,
    fontSize: 14,
    color: "#a00",
  },
  results: {
    marginTop: 20,
    borderTop: "1px solid #eee",
    paddingTop: 16,
  },
  h2: { fontSize: 18, margin: "0 0 12px" },
  h3: { fontSize: 15, margin: "16px 0 8px", color: "#333" },
  row: {
    display: "flex",
    justifyContent: "space-between",
    padding: "6px 0",
    borderBottom: "1px solid #f0f0f0",
    fontSize: 14,
  },
  rowLabel: { color: "#666" },
  rowValue: { color: "#111", fontWeight: 500, textAlign: "right" },
  ol: { margin: "8px 0 0", paddingLeft: 22, fontSize: 14, lineHeight: 1.7 },
  li: { marginBottom: 2 },
  fmt: { color: "#888", fontSize: 12 },
  muted: { color: "#888", fontSize: 14, margin: "8px 0 0" },
  annotation: {
    marginTop: 16,
    padding: 12,
    background: "#fff8e8",
    border: "1px solid #f0d878",
    borderRadius: 10,
  },
  message: {
    fontSize: 14,
    color: "#665000",
    margin: "0 0 10px",
    fontWeight: 500,
  },
  annotatedImg: {
    width: "100%",
    height: "auto",
    borderRadius: 8,
    display: "block",
  },
};
