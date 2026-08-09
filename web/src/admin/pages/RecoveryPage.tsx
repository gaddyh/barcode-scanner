import { KpiCard } from "../components/KpiCard";
import { useAdminMetrics } from "../useAdminMetrics";
import type { MetricsResponse } from "../types";

interface RecoveryPageProps {
  hours: number;
}

export function RecoveryPage({ hours }: RecoveryPageProps) {
  const { data, loading, error } = useAdminMetrics({ hours });

  if (loading) return <div style={styles.loading}>Loading…</div>;
  if (error) return <div style={styles.error}>Error: {error}</div>;
  if (!data) return null;

  const m = data as MetricsResponse;

  if (m.source === "unavailable") {
    return <div style={styles.error}>LangSmith is unavailable.</div>;
  }

  // Compute counts for the funnel.
  const totalRuns = m.images_processed;
  const recoveryAttempted = Math.round(
    (m.recovery_attempted_pct / 100) * totalRuns,
  );
  const recoverySucceeded = Math.round(
    (m.recovery_success_pct / 100) * recoveryAttempted,
  );
  const completedAfterRecovery = Math.round(
    (m.recovered_complete_pct / 100) * totalRuns,
  );

  return (
    <div>
      <h2 style={styles.h2}>Recovery KPIs</h2>
      <div style={styles.kpiRow}>
        <KpiCard label="Recovery attempted" value={m.recovery_attempted_pct.toFixed(1)} unit="%" />
        <KpiCard label="Recovery success" value={m.recovery_success_pct.toFixed(1)} unit="%" />
        <KpiCard label="Avg labels tried" value={m.avg_labels_tried.toFixed(1)} />
        <KpiCard label="Avg labels resolved" value={m.avg_labels_resolved.toFixed(1)} />
      </div>

      <h2 style={styles.h2}>Recovery funnel</h2>
      <div style={styles.funnel}>
        <div style={styles.funnelStep}>
          <div style={styles.funnelNum}>{totalRuns}</div>
          <div style={styles.funnelLabel}>total runs</div>
        </div>
        <div style={styles.funnelArrow}>↓</div>
        <div style={styles.funnelStep}>
          <div style={styles.funnelNum}>{recoveryAttempted}</div>
          <div style={styles.funnelLabel}>
            required recovery
            <span style={styles.funnelDenom}>
              {" "}{m.recovery_attempted_pct.toFixed(1)}% of all runs
            </span>
          </div>
        </div>
        <div style={styles.funnelArrow}>↓</div>
        <div style={styles.funnelStep}>
          <div style={styles.funnelNum}>{recoverySucceeded}</div>
          <div style={styles.funnelLabel}>
            recovered ≥1 label
            <span style={styles.funnelDenom}>
              {" "}{m.recovery_success_pct.toFixed(1)}% of recovery attempts
            </span>
          </div>
        </div>
        <div style={styles.funnelArrow}>↓</div>
        <div style={styles.funnelStep}>
          <div style={styles.funnelNum}>{completedAfterRecovery}</div>
          <div style={styles.funnelLabel}>
            completed after recovery
            <span style={styles.funnelDenom}>
              {" "}{m.recovered_complete_pct.toFixed(1)}% of all runs
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  h2: {
    fontSize: 16,
    fontWeight: 600,
    color: "#1e293b",
    margin: "24px 0 16px 0",
  },
  kpiRow: {
    display: "flex",
    flexWrap: "wrap",
    gap: 12,
  },
  funnel: {
    background: "#fff",
    border: "1px solid #e2e8f0",
    borderRadius: 8,
    padding: "20px 24px",
  },
  funnelStep: {
    textAlign: "center",
    padding: "12px 0",
  },
  funnelNum: {
    fontSize: 28,
    fontWeight: 700,
    color: "#1e293b",
  },
  funnelLabel: {
    fontSize: 13,
    color: "#64748b",
    marginTop: 4,
  },
  funnelDenom: {
    color: "#94a3b8",
    fontSize: 12,
  },
  funnelArrow: {
    fontSize: 18,
    color: "#cbd5e1",
    textAlign: "center",
  },
  loading: {
    color: "#64748b",
    padding: "24px 0",
  },
  error: {
    color: "#dc2626",
    padding: "24px 0",
  },
};
