import { useState } from "react";
import { BarChart } from "../components/BarChart";
import { VersionTable } from "../components/VersionTable";
import { useAdminMetrics } from "../useAdminMetrics";
import type { GroupedMetricsResponse, MetricsResponse } from "../types";

interface VersionsPageProps {
  hours: number;
}

const GROUP_OPTIONS = [
  { label: "Pipeline version", value: "pipeline_version" },
  { label: "Scanner version", value: "scanner_version" },
  { label: "Vision prompt", value: "vision_prompt_version" },
  { label: "Vision model", value: "vision_model" },
  { label: "Recovery version", value: "recovery_version" },
];

const METRIC_OPTIONS = [
  { label: "Complete %", key: "final_complete_pct", fmt: (v: number) => `${v.toFixed(1)}%` },
  { label: "Retry %", key: "user_retry_required_pct", fmt: (v: number) => `${v.toFixed(1)}%` },
  { label: "Match %", key: "scanner_vision_match_pct", fmt: (v: number) => `${v.toFixed(1)}%` },
  { label: "Recovery %", key: "recovery_success_pct", fmt: (v: number) => `${v.toFixed(1)}%` },
];

export function VersionsPage({ hours }: VersionsPageProps) {
  const [groupBy, setGroupBy] = useState("pipeline_version");
  const [metricIdx, setMetricIdx] = useState(0);

  const { data, loading, error } = useAdminMetrics({ hours, groupBy });

  if (loading) return <div style={styles.loading}>Loading…</div>;
  if (error) return <div style={styles.error}>Error: {error}</div>;
  if (!data) return null;

  const grouped = data as GroupedMetricsResponse;

  if (grouped.source === "unavailable") {
    return <div style={styles.error}>LangSmith is unavailable.</div>;
  }

  const groups = grouped.groups;
  const versions = Object.keys(groups).sort();
  const metric = METRIC_OPTIONS[metricIdx];

  // Build bar chart data from the selected metric.
  const barData = versions.map((v) => {
    const m = groups[v] as MetricsResponse;
    const value = (m as unknown as Record<string, number>)[metric.key];
    return {
      label: v,
      value,
      displayValue: metric.fmt(value),
    };
  });

  return (
    <div>
      {/* Group-by selector */}
      <div style={styles.selectorRow}>
        <label style={styles.selectorLabel}>Compare by:</label>
        <select
          value={groupBy}
          onChange={(e) => setGroupBy(e.target.value)}
          style={styles.select}
        >
          {GROUP_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>

      {/* Bar chart for selected metric */}
      <div style={styles.selectorRow}>
        <label style={styles.selectorLabel}>Metric:</label>
        <select
          value={metricIdx}
          onChange={(e) => setMetricIdx(Number(e.target.value))}
          style={styles.select}
        >
          {METRIC_OPTIONS.map((opt, i) => (
            <option key={opt.key} value={i}>{opt.label}</option>
          ))}
        </select>
      </div>

      <h2 style={styles.h2}>{metric.label} by {groupBy.replace(/_/g, " ")}</h2>
      <BarChart data={barData} />

      <h2 style={styles.h2}>Full comparison</h2>
      <VersionTable groups={groups} />
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
  selectorRow: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginBottom: 12,
  },
  selectorLabel: {
    fontSize: 13,
    fontWeight: 600,
    color: "#475569",
  },
  select: {
    padding: "6px 12px",
    borderRadius: 6,
    border: "1px solid #cbd5e1",
    fontSize: 13,
    background: "#fff",
    color: "#1e293b",
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
