import type { MetricsResponse } from "../types";

interface VersionTableProps {
  groups: Record<string, MetricsResponse>;
}

function fmtPct(v: number): string {
  return `${v.toFixed(1)}%`;
}

function fmtMs(v: number): string {
  if (v >= 1000) return `${(v / 1000).toFixed(1)}s`;
  return `${Math.round(v)}ms`;
}

export function VersionTable({ groups }: VersionTableProps) {
  const versions = Object.keys(groups).sort();

  if (versions.length === 0) {
    return <div style={styles.empty}>No data for this grouping.</div>;
  }

  return (
    <div style={styles.tableWrap}>
      <table style={styles.table}>
        <thead>
          <tr>
            <th style={styles.th}>Version</th>
            <th style={{ ...styles.th, ...styles.num }}>Runs</th>
            <th style={{ ...styles.th, ...styles.num }}>Complete %</th>
            <th style={{ ...styles.th, ...styles.num }}>Retry %</th>
            <th style={{ ...styles.th, ...styles.num }}>Match %</th>
            <th style={{ ...styles.th, ...styles.num }}>Recovery %</th>
            <th style={{ ...styles.th, ...styles.num }}>P95</th>
          </tr>
        </thead>
        <tbody>
          {versions.map((version) => {
            const m = groups[version];
            return (
              <tr key={version}>
                <td style={styles.td}><code>{version}</code></td>
                <td style={{ ...styles.td, ...styles.num }}>{m.images_processed}</td>
                <td style={{ ...styles.td, ...styles.num }}>{fmtPct(m.final_complete_pct)}</td>
                <td style={{ ...styles.td, ...styles.num }}>{fmtPct(m.user_retry_required_pct)}</td>
                <td style={{ ...styles.td, ...styles.num }}>{fmtPct(m.scanner_vision_match_pct)}</td>
                <td style={{ ...styles.td, ...styles.num }}>{fmtPct(m.recovery_success_pct)}</td>
                <td style={{ ...styles.td, ...styles.num }}>{fmtMs(m.p95_latency_ms)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  tableWrap: {
    overflowX: "auto",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: 13,
  },
  th: {
    textAlign: "left",
    padding: "10px 12px",
    borderBottom: "2px solid #e2e8f0",
    fontWeight: 600,
    color: "#64748b",
    fontSize: 12,
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  td: {
    padding: "10px 12px",
    borderBottom: "1px solid #f1f5f9",
    color: "#1e293b",
  },
  num: {
    textAlign: "right",
    fontVariantNumeric: "tabular-nums",
  },
  empty: {
    color: "#94a3b8",
    fontSize: 14,
    padding: "12px 0",
  },
};
