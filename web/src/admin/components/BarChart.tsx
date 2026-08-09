interface BarDatum {
  label: string;
  value: number;
  displayValue?: string;
}

interface BarChartProps {
  data: BarDatum[];
  /** Max value for scaling bars. If omitted, uses max of data. */
  maxValue?: number;
}

export function BarChart({ data, maxValue }: BarChartProps) {
  if (data.length === 0) {
    return <div style={styles.empty}>No data</div>;
  }

  const max = maxValue ?? Math.max(...data.map((d) => d.value), 1);

  return (
    <div style={styles.container}>
      {data.map((d, i) => {
        const pct = (d.value / max) * 100;
        return (
          <div key={i} style={styles.row}>
            <div style={styles.barLabel}>{d.label}</div>
            <div style={styles.barTrack}>
              <div style={{ ...styles.barFill, width: `${pct}%` }} />
            </div>
            <div style={styles.barValue}>
              {d.displayValue ?? d.value}
            </div>
          </div>
        );
      })}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    gap: 8,
  },
  empty: {
    color: "#94a3b8",
    fontSize: 14,
    padding: "12px 0",
  },
  row: {
    display: "flex",
    alignItems: "center",
    gap: 12,
  },
  barLabel: {
    width: 180,
    fontSize: 13,
    color: "#475569",
    textAlign: "right",
    flexShrink: 0,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  barTrack: {
    flex: 1,
    height: 20,
    background: "#f1f5f9",
    borderRadius: 4,
    overflow: "hidden",
  },
  barFill: {
    height: "100%",
    background: "#3b82f6",
    borderRadius: 4,
    transition: "width 0.3s ease",
  },
  barValue: {
    width: 60,
    fontSize: 13,
    fontWeight: 600,
    color: "#1e293b",
    flexShrink: 0,
  },
};
