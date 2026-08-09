interface KpiCardProps {
  label: string;
  value: string | number;
  unit?: string;
}

export function KpiCard({ label, value, unit }: KpiCardProps) {
  return (
    <div style={styles.card}>
      <div style={styles.label}>{label}</div>
      <div style={styles.value}>
        {value}
        {unit && <span style={styles.unit}> {unit}</span>}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  card: {
    background: "#fff",
    border: "1px solid #e2e8f0",
    borderRadius: 8,
    padding: "16px 20px",
    minWidth: 140,
  },
  label: {
    fontSize: 12,
    fontWeight: 600,
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: 0.5,
    marginBottom: 8,
  },
  value: {
    fontSize: 28,
    fontWeight: 700,
    color: "#1e293b",
  },
  unit: {
    fontSize: 14,
    fontWeight: 500,
    color: "#64748b",
  },
};
