import { useEffect, useRef, useState } from "react";
import {
  createReceivingSession,
  uploadReceivingImage,
  getReceivingSession,
  submitReceivingSession,
  attachReceivingSessionContext,
  submitFeedback,
  fetchCustomers,
  fetchBranches,
  getParticipantId,
  type OrderAction,
  type SelectOption,
  type ReceivingSessionResponse,
  type ReceivingImageResponse,
  type ReceivingSubmitResponse,
} from "./api";
import { useHashRoute } from "./router";
import { AdminApp } from "./admin/AdminApp";

type Source = "camera" | "gallery";

export default function App() {
  const route = useHashRoute();
  const [source, setSource] = useState<Source | null>(null);
  const [customers, setCustomers] = useState<SelectOption[]>([]);
  const [branches, setBranches] = useState<SelectOption[]>([]);
  const [customerId, setCustomerId] = useState("");
  const [branchId, setBranchId] = useState("");
  const [action, setAction] = useState<OrderAction | null>(null);
  const [optionsLoading, setOptionsLoading] = useState(false);
  const [branchesLoading, setBranchesLoading] = useState(false);
  const [optionsError, setOptionsError] = useState<string | null>(null);
  const [branchesError, setBranchesError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [receivingSession, setReceivingSession] =
    useState<ReceivingSessionResponse | null>(null);
  const [imageResult, setImageResult] =
    useState<ReceivingImageResponse | null>(null);
  const [submitResult, setSubmitResult] =
    useState<ReceivingSubmitResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [feedbackError, setFeedbackError] = useState<string | null>(null);

  const cameraInputRef = useRef<HTMLInputElement>(null);
  const galleryInputRef = useRef<HTMLInputElement>(null);

  // Retained for potential debug/diagnostics; not shown to end users.
  void source;

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

  // --- Derived screen state (no stage enum) ---------------------------
  // The screen is a function of the server-side session status plus the
  // local `action` choice (the user's pick before context is attached).
  const sessionActive = receivingSession?.status === "active";
  const sessionSubmitted = receivingSession?.status === "submitted";
  const sessionUnknown = receivingSession?.status === "submission_unknown";
  const needsSelection = (imageResult?.candidates?.length ?? 0) > 0;
  const contextAttached =
    !!receivingSession?.customer_id &&
    !!receivingSession?.branch_id &&
    !!receivingSession?.action;
  // The user has chosen an action locally but not yet attached context.
  const choosingContext = sessionActive && action !== null && !contextAttached;
  const showStartScreen = !receivingSession && !loading;

  function resetAll() {
    setReceivingSession(null);
    setImageResult(null);
    setSubmitResult(null);
    setError(null);
    setAction(null);
    setCustomerId("");
    setBranchId("");
    setFeedbackSent(false);
    setFeedbackError(null);
  }

  function handleFile(e: React.ChangeEvent<HTMLInputElement>, src: Source) {
    const f = e.target.files?.[0] ?? null;
    if (!f) return;
    setSource(src);
    void src;
    onPhotoSelected(f);
  }

  // The whole first action: get/create an active session → upload → refresh.
  // Also used by "הוסף צילום" to add more images to the same session.
  async function onPhotoSelected(f: File) {
    setLoading(true);
    setError(null);
    setImageResult(null);
    setSubmitResult(null);
    setFeedbackSent(false);
    setFeedbackError(null);
    try {
      // Get or create an active session for this participant. The server
      // resumes an existing ACTIVE session (200) or creates a new one (201).
      let session = receivingSession;
      if (!session || session.status !== "active") {
        session = await createReceivingSession(getParticipantId());
        setReceivingSession(session);
      }
      const res = await uploadReceivingImage(session.session_id, f);
      setImageResult(res);
      // Refresh session state from server.
      const refreshed = await getReceivingSession(session.session_id);
      setReceivingSession(refreshed);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function onSubmitOrder() {
    if (!receivingSession) return;
    if (!customerId || !branchId || !action) return;
    setLoading(true);
    setError(null);
    setSubmitResult(null);
    try {
      // Attach context first (if not already attached), then submit.
      if (!contextAttached) {
        const attached = await attachReceivingSessionContext(
          receivingSession.session_id,
          customerId,
          branchId,
          action,
        );
        setReceivingSession(attached);
      }
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

  // The receiving flow has no scanner trace_id; use the session_id
  // (a UUID) as the feedback trace identifier.
  const traceId = receivingSession?.session_id ?? null;

  return (
    <div dir="rtl" lang="he" style={styles.container}>
      <h1 style={styles.h1}>סריקת קופסאות</h1>
      <p style={styles.subtitle}>
        צלם מספר קופסאות יחד ונזהה אותן אוטומטית.
      </p>

      {/* Start screen — no session yet */}
      {showStartScreen && (
        <>
          <div style={styles.inputRow}>
            <label style={styles.primaryButton}>
              📷 התחל סריקה
              <input
                ref={cameraInputRef}
                hidden
                type="file"
                accept="image/*"
                capture="environment"
                onChange={(e) => handleFile(e, "camera")}
              />
            </label>
            <label style={styles.secondaryButton}>
              בחר תמונה מהטלפון
              <input
                ref={galleryInputRef}
                hidden
                type="file"
                accept="image/*"
                onChange={(e) => handleFile(e, "gallery")}
              />
            </label>
          </div>
        </>
      )}

      {/* Loading (first scan, no session yet) */}
      {loading && !receivingSession && (
        <div style={styles.loadingBox}>סורק…</div>
      )}

      {error && (
        <div style={styles.error}>
          <strong>שגיאה:</strong> {error}
        </div>
      )}

      {/* Active session — scan results + actions */}
      {sessionActive && receivingSession && (
        <div style={styles.results}>
          {receivingSession.box_count > 0 ? (
            <>
              <h2 style={styles.h2}>
                {receivingSession.box_count} קופסאות זוהו
              </h2>
              <p style={styles.muted}>
                {receivingSession.items.length} ברקודים שונים
              </p>

              {/* Aggregated barcode → quantity list */}
              {receivingSession.items.length > 0 && (
                <ol style={styles.list}>
                  {receivingSession.items.map((item, i) => (
                    <li key={i} style={styles.listItem}>
                      <strong>{item.barcode_value}</strong>
                      {` — כמות: ${item.quantity}`}
                    </li>
                  ))}
                </ol>
              )}

              {/* Missing-box prompt */}
              {receivingSession.discrepancy.missing > 0 ? (
                <div style={styles.promptMore}>
                  זוהו {receivingSession.box_count} מתוך{" "}
                  {receivingSession.expected_count}.{" "}
                  {receivingSession.discrepancy.missing === 1
                    ? "צלם עכשיו רק את הקופסה החסרה."
                    : `צלם עכשיו את ${receivingSession.discrepancy.missing} הקופסאות החסרות.`}
                </div>
              ) : (
                <div style={styles.completeBadge}>
                  ✅ כל הקופסאות נקלטו ({receivingSession.box_count}/
                  {receivingSession.expected_count})
                </div>
              )}

              {/* Add more photos */}
              <div style={styles.inputRow}>
                <label style={styles.secondaryButton}>
                  📷 הוסף צילום
                  <input
                    hidden
                    type="file"
                    accept="image/*"
                    capture="environment"
                    onChange={(e) => handleFile(e, "camera")}
                  />
                </label>
              </div>

              {/* Image upload message */}
              {imageResult?.message && (
                <div style={styles.sessionMessage}>{imageResult.message}</div>
              )}

              {/* Annotated image with red circles around missing boxes */}
              {imageResult?.annotated_image_b64 && (
                <div style={styles.annotatedImageWrap}>
                  <img
                    src={`data:image/png;base64,${imageResult.annotated_image_b64}`}
                    alt="סימון קופסאות חסרות"
                    style={styles.annotatedImage}
                  />
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

              {/* Choose action — only if context not yet attached */}
              {!choosingContext && (
                <>
                  <h3 style={styles.h3}>מה תרצה לעשות?</h3>
                  <div style={styles.actionRow}>
                    <button
                      onClick={() => setAction("create_order")}
                      style={styles.actionBtn}
                    >
                      צור הזמנה
                    </button>
                    <button
                      disabled
                      title="בקרוב"
                      style={{ ...styles.actionBtn, ...styles.actionDisabled }}
                    >
                      בדיקת הזמנה — בקרוב
                    </button>
                  </div>
                </>
              )}

              {/* Choose context — customer + branch */}
              {choosingContext && (
                <div style={styles.selectGroup}>
                  <h3 style={styles.h3}>
                    {action === "create_order"
                      ? "צור הזמנה"
                      : "בדיקת הזמנה"}
                  </h3>
                  <label style={styles.fieldLabel}>
                    בחר לקוח
                    <select
                      value={customerId}
                      onChange={(e) => setCustomerId(e.target.value)}
                      disabled={optionsLoading || loading}
                      style={styles.select}
                    >
                      <option value="">
                        {optionsLoading ? "טוען לקוחות…" : "בחר לקוח"}
                      </option>
                      {customers.map((customer) => (
                        <option key={customer.id} value={customer.id}>
                          {customer.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  {optionsError && (
                    <p style={styles.fieldError}>שגיאה: {optionsError}</p>
                  )}

                  <label style={styles.fieldLabel}>
                    בחר סניף
                    <select
                      value={branchId}
                      onChange={(e) => setBranchId(e.target.value)}
                      disabled={!customerId || branchesLoading || loading}
                      style={styles.select}
                    >
                      <option value="">
                        {branchesLoading
                          ? "טוען סניפים…"
                          : customerId
                            ? "בחר סניף"
                            : "בחר לקוח תחילה"}
                      </option>
                      {branches.map((branch) => (
                        <option key={branch.id} value={branch.id}>
                          {branch.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  {branchesError && (
                    <p style={styles.fieldError}>שגיאה: {branchesError}</p>
                  )}

                  <button
                    onClick={onSubmitOrder}
                    disabled={!customerId || !branchId || loading}
                    style={{
                      ...styles.analyze,
                      opacity: !customerId || !branchId || loading ? 0.5 : 1,
                    }}
                  >
                    {loading ? "יוצר טיוטה…" : "צור טיוטה ב־Priority"}
                  </button>

                  {/* Back to action choice */}
                  <button
                    onClick={() => {
                      setAction(null);
                      setCustomerId("");
                      setBranchId("");
                    }}
                    disabled={loading}
                    style={styles.backBtn}
                  >
                    ← חזרה
                  </button>
                </div>
              )}
            </>
          ) : (
            // Active session, no boxes yet — first image still processing or empty.
            <div style={styles.muted}>הסריקה התחילה — צלם את הקופסאות.</div>
          )}
        </div>
      )}

      {/* Submission unknown — show warning */}
      {sessionUnknown && (
        <div style={styles.results}>
          <div style={{ ...styles.error, borderColor: "#f59e0b" }}>
            ⚠️ לא הצלחנו לוודא אם הטיוטה נוצרה. אל תתחיל קליטה חדשה — נסה שוב.
          </div>
          {submitResult?.error && (
            <p style={styles.muted}>{submitResult.error.message}</p>
          )}
          <button
            onClick={onSubmitOrder}
            disabled={loading || !contextAttached}
            style={{
              ...styles.analyze,
              opacity: loading || !contextAttached ? 0.5 : 1,
            }}
          >
            {loading ? "מנסה שוב…" : "נסה שוב"}
          </button>
        </div>
      )}

      {/* Submitted — success */}
      {sessionSubmitted && (
        <div style={styles.results}>
          <div style={styles.completeBadge}>
            ✅ כל הקופסאות נקלטו — הטיוטה נוצרה בהצלחה ב־Priority
          </div>
          {submitResult?.error && (
            <div style={styles.error}>
              <strong>שגיאה ביצירת הטיוטה:</strong> {submitResult.error.message}
            </div>
          )}
          {submitResult?.retry_recommended && (
            <div style={styles.promptMore}>
              🔄 מומלץ לנסות שוב — ייתכן שההזמנה לא נוצרה.
            </div>
          )}
          {receivingSession && receivingSession.items.length > 0 && (
            <>
              <h3 style={styles.h3}>
                פריטים שנקלטו ({receivingSession.items.length})
              </h3>
              <ol style={styles.list}>
                {receivingSession.items.map((item, i) => (
                  <li key={i} style={styles.listItem}>
                    <strong>{item.barcode_value}</strong>
                    {` — כמות: ${item.quantity}`}
                  </li>
                ))}
              </ol>
            </>
          )}
          {!feedbackSent && (
            <FeedbackRow onFeedback={sendFeedback} feedbackError={feedbackError} />
          )}
          {feedbackSent && <p style={styles.feedbackDone}>תודה על המשוב.</p>}
          <button onClick={resetAll} style={styles.backBtn}>
            סריקה חדשה
          </button>
        </div>
      )}

      <div style={{ marginTop: 32, paddingTop: 16, borderTop: "1px solid #e2e8f0" }}>
        <a
          href="#/admin"
          style={{
            color: "#3b82f6",
            textDecoration: "none",
            fontSize: 14,
            fontWeight: 500,
          }}
        >
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
      {feedbackError && (
        <p style={styles.error}>שגיאה בשליחת משוב: {feedbackError}</p>
      )}
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
  h2: { fontSize: 20, fontWeight: 600, marginBottom: 4 },
  h3: { fontSize: 16, fontWeight: 600, marginTop: 16, marginBottom: 8 },
  inputRow: { display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap" },
  primaryButton: {
    flex: "1 1 100%",
    padding: "14px 16px",
    background: "#3b82f6",
    color: "#fff",
    borderRadius: 8,
    textAlign: "center",
    cursor: "pointer",
    fontSize: 16,
    fontWeight: 600,
    border: "none",
  },
  secondaryButton: {
    flex: "1 1 auto",
    padding: "10px 16px",
    background: "#f1f5f9",
    borderRadius: 8,
    textAlign: "center",
    cursor: "pointer",
    fontSize: 14,
    fontWeight: 500,
    border: "1px solid #cbd5e1",
  },
  loadingBox: {
    padding: 16,
    background: "#f8fafc",
    borderRadius: 8,
    textAlign: "center",
    fontSize: 16,
    fontWeight: 500,
    color: "#64748b",
  },
  selectGroup: { display: "grid", gap: 10, marginBottom: 16, marginTop: 8 },
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
  backBtn: {
    width: "100%",
    padding: "8px 16px",
    background: "transparent",
    color: "#3b82f6",
    border: "none",
    borderRadius: 8,
    fontSize: 14,
    fontWeight: 500,
    cursor: "pointer",
  },
  results: {
    padding: 16,
    background: "#fff",
    border: "1px solid #e2e8f0",
    borderRadius: 8,
    marginBottom: 16,
  },
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
  annotatedImageWrap: {
    marginTop: 12,
    marginBottom: 12,
    borderRadius: 8,
    overflow: "hidden",
    border: "1px solid #e5e7eb",
  },
  annotatedImage: {
    display: "block",
    width: "100%",
    height: "auto",
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
  actionRow: { display: "flex", gap: 8, flexWrap: "wrap" },
  actionBtn: {
    flex: "1 1 40%",
    padding: "12px 16px",
    background: "#fff",
    color: "#1e293b",
    border: "1px solid #cbd5e1",
    borderRadius: 8,
    fontSize: 15,
    fontWeight: 600,
    cursor: "pointer",
  },
  actionDisabled: {
    opacity: 0.5,
    cursor: "not-allowed",
    background: "#f1f5f9",
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
}
