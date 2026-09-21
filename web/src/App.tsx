import { useEffect, useRef, useState } from "react";
import {
  scanBarcode,
  createReceivingSession,
  uploadReceivingImage,
  getReceivingSession,
  submitReceivingSession,
  submitFeedback,
  fetchCustomers,
  fetchBranches,
  type OrderAction,
  type SelectOption,
  type ScanResponse,
  type ReceivingSessionResponse,
  type ReceivingImageResponse,
  type ReceivingSubmitResponse,
} from "./api";
import { useHashRoute } from "./router";
import { AdminApp } from "./admin/AdminApp";

type Source = "camera" | "gallery";
type Mode = "receiving" | "scanner";

export default function App() {
  const route = useHashRoute();
  const [file, setFile] = useState<File | null>(null);
  const [source, setSource] = useState<Source | null>(null);
  const [mode, setMode] = useState<Mode>("receiving");
  const [customers, setCustomers] = useState<SelectOption[]>([]);
  const [branches, setBranches] = useState<SelectOption[]>([]);
  const [customerId, setCustomerId] = useState("");
  const [branchId, setBranchId] = useState("");
  const [action, setAction] = useState<OrderAction | "">("");
  const [optionsLoading, setOptionsLoading] = useState(false);
  const [branchesLoading, setBranchesLoading] = useState(false);
  const [optionsError, setOptionsError] = useState<string | null>(null);
  const [branchesError, setBranchesError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [scanResult, setScanResult] = useState<ScanResponse | null>(null);
  const [receivingSession, setReceivingSession] =
    useState<ReceivingSessionResponse | null>(null);
  const [imageResult, setImageResult] =
    useState<ReceivingImageResponse | null>(null);
  const [submitResult, setSubmitResult] =
    useState<ReceivingSubmitResponse | null>(null);
  const [totalMs, setTotalMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [feedbackError, setFeedbackError] = useState<string | null>(null);

  const cameraInputRef = useRef<HTMLInputElement>(null);
  const galleryInputRef = useRef<HTMLInputElement>(null);

  // Retained for potential debug/diagnostics; not shown to end users.
  void source;
  void totalMs;

  useEffect(() => {
    let cancelled = false;
    setOptionsLoading(true);
    setOptionsError(null);
    fetchCustomers()
      .then((items) => {
        if (!cancelled) setCustomers(items);
      })
      .catch((err) => {
        if (!cancelled) setOptionsError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setOptionsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    setBranchId("");
    setBranches([]);
    setBranchesError(null);
    if (!customerId) return;

    let cancelled = false;
    setBranchesLoading(true);
    fetchBranches(customerId)
      .then((items) => {
        if (!cancelled) setBranches(items);
      })
      .catch((err) => {
        if (!cancelled) setBranchesError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setBranchesLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [customerId]);

  if (route === "#/admin") return <AdminApp />;

  function clearResults() {
    setScanResult(null);
    setReceivingSession(null);
    setImageResult(null);
    setSubmitResult(null);
    setTotalMs(null);
    setError(null);
    setFeedbackSent(false);
    setFeedbackError(null);
  }

  function handleCustomerChange(value: string) {
    setCustomerId(value);
    clearResults();
  }

  function handleFile(e: React.ChangeEvent<HTMLInputElement>, src: Source) {
    const f = e.target.files?.[0] ?? null;
    if (!f) return;
    setFile(f);
    setSource(src);
    setScanResult(null);
    setImageResult(null);
    setSubmitResult(null);
    setTotalMs(null);
    setError(null);
    setFeedbackSent(false);
    setFeedbackError(null);
  }

  async function onCreateSession() {
    if (!customerId || !branchId || !action) return;
    setLoading(true);
    setError(null);
    setReceivingSession(null);
    setImageResult(null);
    setSubmitResult(null);
    setFeedbackSent(false);
    setFeedbackError(null);
    try {
      const res = await createReceivingSession(
        customerId,
        branchId,
        action as OrderAction,
      );
      setReceivingSession(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function onUploadImage() {
    if (!file || !receivingSession) return;
    setLoading(true);
    setError(null);
    setImageResult(null);
    setSubmitResult(null);
    setTotalMs(null);
    const t0 = performance.now();
    try {
      const res = await uploadReceivingImage(receivingSession.session_id, file);
      setImageResult(res);
      // Refresh session state from server.
      const session = await getReceivingSession(receivingSession.session_id);
      setReceivingSession(session);
      setTotalMs(performance.now() - t0);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function onSubmitOrder() {
    if (!receivingSession) return;
    setLoading(true);
    setError(null);
    setSubmitResult(null);
    try {
      const res = await submitReceivingSession(receivingSession.session_id);
      setSubmitResult(res);
      // Refresh session state from server.
      const session = await getReceivingSession(receivingSession.session_id);
      setReceivingSession(session);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function onAnalyze() {
    if (!file) return;
    if (mode === "scanner") {
      setLoading(true);
      setError(null);
      setScanResult(null);
      setTotalMs(null);
      setFeedbackSent(false);
      setFeedbackError(null);
      const t0 = performance.now();
      try {
        const res = await scanBarcode(file);
        setScanResult(res);
        setTotalMs(performance.now() - t0);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoading(false);
      }
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

  const sessionActive = receivingSession?.status === "active";
  const sessionSubmitted = receivingSession?.status === "submitted";
  const sessionUnknown = receivingSession?.status === "submission_unknown";
  const needsSelection = (imageResult?.candidates?.length ?? 0) > 0;

  return (
    <div dir="rtl" lang="he" style={styles.container}>
      <h1 style={styles.h1}>קליטת קופסאות</h1>
      <p style={styles.subtitle}>צלם את הקופסאות, בדוק שהכול נקלט, וצור טיוטה ב־Priority</p>

      {/* Mode toggle */}
      <div style={styles.toggleRow}>
        <button
          onClick={() => setMode("receiving")}
          style={{
            ...styles.toggleBtn,
            ...(mode === "receiving" ? styles.toggleActive : {}),
          }}
        >
          קליטת סחורה
        </button>
        <button
          onClick={() => setMode("scanner")}
          style={{
            ...styles.toggleBtn,
            ...(mode === "scanner" ? styles.toggleActive : {}),
          }}
        >
          סורק בלבד
        </button>
      </div>

      <div style={styles.selectGroup}>
        <label style={styles.fieldLabel}>
          לקוח
          <select
            value={customerId}
            onChange={(e) => handleCustomerChange(e.target.value)}
            disabled={optionsLoading || loading}
            style={styles.select}
          >
            <option value="">{optionsLoading ? "טוען לקוחות…" : "בחר לקוח"}</option>
            {customers.map((customer) => (
              <option key={customer.id} value={customer.id}>{customer.name}</option>
            ))}
          </select>
        </label>
        {optionsError && <p style={styles.fieldError}>שגיאה: {optionsError}</p>}

        <label style={styles.fieldLabel}>
          פעולה
          <select
            value={action}
            onChange={(e) => {
              setAction(e.target.value as OrderAction | "");
              clearResults();
            }}
            disabled={loading}
            style={styles.select}
          >
            <option value="">בחר פעולה</option>
            <option value="create_order">יצירת הזמנה</option>
            <option value="verify_order_before_shipment">בדיקת הזמנה לפני משלוח</option>
          </select>
        </label>

        <label style={styles.fieldLabel}>
          סניף / מחסן
          <select
            value={branchId}
            onChange={(e) => setBranchId(e.target.value)}
            disabled={!customerId || branchesLoading || loading}
            style={styles.select}
          >
            <option value="">
              {branchesLoading ? "טוען סניפים…" : customerId ? "בחר סניף" : "בחר לקוח תחילה"}
            </option>
            {branches.map((branch) => (
              <option key={branch.id} value={branch.id}>{branch.name}</option>
            ))}
          </select>
        </label>
        {branchesError && <p style={styles.fieldError}>שגיאה: {branchesError}</p>}
      </div>

      <div style={styles.inputRow}>
        <label style={styles.button}>
          צלם קופסאות
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
          בחר תמונה
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
          <div><strong>תמונה נבחרה:</strong> {file.name}</div>
        </div>
      )}

      {/* Receiving flow buttons */}
      {mode === "receiving" && (
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button
            onClick={onCreateSession}
            disabled={!customerId || !branchId || !action || loading || !!receivingSession}
            style={{
              ...styles.analyze,
              opacity: !customerId || !branchId || !action || loading || !!receivingSession ? 0.5 : 1,
            }}
          >
            {receivingSession ? "הסריקה התחילה" : "התחל סריקה"}
          </button>
          <button
            onClick={onUploadImage}
            disabled={!file || !receivingSession || loading || !sessionActive}
            style={{
              ...styles.analyze,
              opacity: !file || !receivingSession || loading || !sessionActive ? 0.5 : 1,
            }}
          >
            {loading ? "סורק…" : "סרוק את התמונה"}
          </button>
          <button
            onClick={onSubmitOrder}
            disabled={!receivingSession || loading || !sessionActive}
            style={{
              ...styles.analyze,
              opacity: !receivingSession || loading || !sessionActive ? 0.5 : 1,
            }}
          >
            {loading ? "יוצר טיוטה…" : "צור טיוטה ב־Priority"}
          </button>
        </div>
      )}

      {/* Scanner-only analyze button */}
      {mode === "scanner" && (
        <button
          onClick={onAnalyze}
          disabled={!file || loading}
          style={{
            ...styles.analyze,
            opacity: !file || loading ? 0.5 : 1,
          }}
        >
          {loading ? "מנתח…" : "נתח (סורק)"}
        </button>
      )}

      {error && (
        <div style={styles.error}>
          <strong>שגיאה:</strong> {error}
        </div>
      )}

      {/* Scanner-only result */}
      {scanResult && (
        <div style={styles.results}>
          <h2 style={styles.h2}>תוצאות סריקה</h2>
          <Row label="סטטוס" value={scanResult.status === "found" ? "נמצא" : "לא נמצא"} />
          <Row label="כמות" value={String(scanResult.count)} />
          <Row label="מידות" value={`${scanResult.image_width} × ${scanResult.image_height}`} />
          <Row label="זמן סריקה" value={`${scanResult.elapsed_ms} מ״מ`} />
          <h3 style={styles.h3}>ברקודים</h3>
          {scanResult.barcodes.length === 0 ? (
            <p style={styles.muted}>לא זוהו ברקודים.</p>
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
          {feedbackSent && <p style={styles.feedbackDone}>תודה על המשוב.</p>}
        </div>
      )}

      {/* Receiving session result */}
      {receivingSession && (
        <div style={styles.results}>
          <h2 style={styles.h2}>מצב הקליטה</h2>

          {/* Progress summary — friendly, no raw status */}
          {sessionSubmitted ? (
            <div style={styles.completeBadge}>
              ✅ כל הקופסאות נקלטו — הטיוטה נוצרה בהצלחה ב־Priority
            </div>
          ) : receivingSession.discrepancy.missing > 0 ? (
            <div style={styles.promptMore}>
              נסרקו {receivingSession.box_count} מתוך {receivingSession.expected_count} קופסאות.
              {" "}
              {receivingSession.discrepancy.missing === 1
                ? "חסרה קופסה אחת."
                : `חסרות ${receivingSession.discrepancy.missing} קופסאות.`}
            </div>
          ) : receivingSession.box_count > 0 ? (
            <div style={styles.completeBadge}>
              ✅ כל הקופסאות נקלטו ({receivingSession.box_count}/{receivingSession.expected_count})
            </div>
          ) : (
            <div style={styles.muted}>הסריקה התחילה — צלם את הקופסאות.</div>
          )}

          {/* Image upload message */}
          {imageResult?.message && (
            <div style={styles.sessionMessage}>
              {imageResult.message}
            </div>
          )}

          {/* Needs user selection — show candidate buttons */}
          {needsSelection && imageResult?.candidates && (
            <div style={styles.candidates}>
              <h3 style={styles.h3}>בחר ברקוד להוספה:</h3>
              <div style={styles.candidateRow}>
                {imageResult.candidates.map((c, i) => (
                  <span key={i} style={styles.candidateBtn}>
                    {c.barcode_value}
                  </span>
                ))}
              </div>
              <p style={styles.muted}>
                בחירת מועמדים עדיין לא מחוברת בזרימת הקליטה.
              </p>
            </div>
          )}

          {/* Active — prompt for more photos */}
          {sessionActive && receivingSession.discrepancy.missing > 0 && (
            <div style={styles.promptMore}>
              📸 צלם עכשיו רק את הקופסה שחסרה.
            </div>
          )}

          {/* Submission unknown — show warning */}
          {sessionUnknown && (
            <div style={{ ...styles.error, borderColor: "#f59e0b" }}>
              ⚠️ לא הצלחנו לוודא אם הטיוטה נוצרה. אל תתחיל קליטה חדשה — נסה שוב.
            </div>
          )}

          {/* Submit result — friendly */}
          {submitResult && submitResult.error && (
            <div style={styles.error}>
              <strong>שגיאה ביצירת הטיוטה:</strong> {submitResult.error.message}
            </div>
          )}
          {submitResult && submitResult.retry_recommended && (
            <div style={styles.promptMore}>
              🔄 מומלץ לנסות שוב — ייתכן שההזמנה לא נוצרה.
            </div>
          )}

          {/* Items list */}
          {receivingSession.items.length > 0 && (
            <>
              <h3 style={styles.h3}>פריטים שנקלטו ({receivingSession.items.length})</h3>
              <ol style={styles.list}>
                {receivingSession.items.map((item, i) => (
                  <li key={i} style={styles.listItem}>
                    <strong>{item.barcode_value}</strong>
                    {item.barcode_format ? ` (${item.barcode_format})` : ""}
                    {` — כמות: ${item.quantity}`}
                  </li>
                ))}
              </ol>
            </>
          )}

          {/* Feedback (only for submitted sessions) */}
          {sessionSubmitted && !feedbackSent && (
            <FeedbackRow onFeedback={sendFeedback} feedbackError={feedbackError} />
          )}
          {feedbackSent && <p style={styles.feedbackDone}>תודה על המשוב.</p>}
        </div>
      )}

      <div style={{ marginTop: 32, paddingTop: 16, borderTop: "1px solid #e2e8f0" }}>
        <a href="#/admin" style={{ color: "#3b82f6", textDecoration: "none", fontSize: 14, fontWeight: 500 }}>
          לוח ניהול ←
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
      <p style={styles.feedbackQ}>האם כל הקופסאות זוהו נכון?</p>
      <div style={styles.feedbackRow}>
        <button onClick={() => onFeedback(true)} style={styles.feedbackBtn}>
          כן
        </button>
        <button onClick={() => onFeedback(false)} style={styles.feedbackBtn}>
          לא
        </button>
      </div>
      {feedbackError && <p style={styles.error}>שגיאה בשליחת משוב: {feedbackError}</p>}
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
  h1: { fontSize: 24, fontWeight: 700, marginBottom: 4 },
  subtitle: { fontSize: 14, color: "#64748b", marginBottom: 16, marginTop: 0 },
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
  selectGroup: { display: "grid", gap: 10, marginBottom: 16 },
  fieldLabel: { display: "grid", gap: 5, fontSize: 13, fontWeight: 600 },
  select: {
    width: "100%",
    padding: "10px 12px",
    border: "1px solid #cbd5e1",
    borderRadius: 8,
    background: "#fff",
    color: "#1e293b",
    fontSize: 14,
  },
  fieldError: { margin: "-4px 0 0", color: "#dc2626", fontSize: 12 },
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
  list: { margin: "8px 0", paddingRight: 20, paddingLeft: 0, fontSize: 14, lineHeight: 1.8 },
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
