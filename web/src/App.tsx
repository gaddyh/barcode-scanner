import { useRef, useState } from "react";
import { scanBarcode, type ScanResponse } from "./api";

type Source = "camera" | "gallery";

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [source, setSource] = useState<Source | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScanResponse | null>(null);
  const [totalMs, setTotalMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const cameraInputRef = useRef<HTMLInputElement>(null);
  const galleryInputRef = useRef<HTMLInputElement>(null);

  function handleFile(e: React.ChangeEvent<HTMLInputElement>, src: Source) {
    const f = e.target.files?.[0] ?? null;
    if (!f) return;
    setFile(f);
    setSource(src);
    setResult(null);
    setTotalMs(null);
    setError(null);
  }

  async function onAnalyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setTotalMs(null);
    const t0 = performance.now();
    try {
      const res = await scanBarcode(file);
      setResult(res);
      setTotalMs(performance.now() - t0);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={styles.container}>
      <h1 style={styles.h1}>Barcode Scanner</h1>
      <p style={styles.sub}>Direct upload — no compression, no WhatsApp.</p>

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
        {loading ? "Analyzing…" : "Analyze"}
      </button>

      {error && (
        <div style={styles.error}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div style={styles.results}>
          <h2 style={styles.h2}>Result</h2>
          <Row label="Filename" value={result.filename} />
          <Row label="Source" value={source ?? "—"} />
          <Row
            label="Dimensions"
            value={`${result.image_width} × ${result.image_height}`}
          />
          <Row
            label="File size"
            value={`${formatBytes(result.upload_bytes)} (${result.upload_bytes} B)`}
          />
          <Row label="Status" value={result.status} />
          <Row label="Count" value={String(result.count)} />
          <Row label="Server scan" value={`${result.elapsed_ms} ms`} />
          <Row label="Total request" value={totalMs != null ? `${Math.round(totalMs)} ms` : "—"} />

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
        </div>
      )}
    </div>
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
  h1: { fontSize: 22, margin: "0 0 4px" },
  sub: { fontSize: 13, color: "#666", margin: "0 0 20px" },
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
};
