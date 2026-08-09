interface TimeRangePickerProps {
  hours: number;
  onChange: (hours: number) => void;
}

const OPTIONS = [
  { label: "24h", value: 24 },
  { label: "7d", value: 168 },
  { label: "30d", value: 720 },
];

export function TimeRangePicker({ hours, onChange }: TimeRangePickerProps) {
  return (
    <div style={styles.container}>
      {OPTIONS.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          style={{
            ...styles.btn,
            ...(hours === opt.value ? styles.active : {}),
          }}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    gap: 4,
    background: "#f1f5f9",
    borderRadius: 8,
    padding: 4,
  },
  btn: {
    border: "none",
    background: "transparent",
    padding: "6px 16px",
    borderRadius: 6,
    fontSize: 13,
    fontWeight: 600,
    color: "#64748b",
    cursor: "pointer",
  },
  active: {
    background: "#fff",
    color: "#1e293b",
    boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
  },
};
