import { useState } from "react";
import { TimeRangePicker } from "./components/TimeRangePicker";
import { OverviewPage } from "./pages/OverviewPage";
import { QualityPage } from "./pages/QualityPage";
import { RecoveryPage } from "./pages/RecoveryPage";
import { VersionsPage } from "./pages/VersionsPage";

type Tab = "overview" | "quality" | "recovery" | "versions";

const TABS: { key: Tab; label: string }[] = [
  { key: "overview", label: "Overview" },
  { key: "quality", label: "Quality" },
  { key: "recovery", label: "Recovery" },
  { key: "versions", label: "Versions" },
];

export function AdminApp() {
  const [tab, setTab] = useState<Tab>("overview");
  const [hours, setHours] = useState(24);

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <h1 style={styles.title}>Barcode Scanner Admin</h1>
        <TimeRangePicker hours={hours} onChange={setHours} />
      </div>

      {/* Tabs */}
      <div style={styles.tabBar}>
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            style={{
              ...styles.tab,
              ...(tab === t.key ? styles.tabActive : {}),
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Page content */}
      <div style={styles.content}>
        {tab === "overview" && <OverviewPage hours={hours} />}
        {tab === "quality" && <QualityPage hours={hours} />}
        {tab === "recovery" && <RecoveryPage hours={hours} />}
        {tab === "versions" && <VersionsPage hours={hours} />}
      </div>

      {/* Footer */}
      <div style={styles.footer}>
        <a href="#" style={styles.backLink}>← Back to upload</a>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    maxWidth: 900,
    margin: "0 auto",
    padding: "24px 20px",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 16,
  },
  title: {
    fontSize: 22,
    fontWeight: 700,
    color: "#1e293b",
    margin: 0,
  },
  tabBar: {
    display: "flex",
    gap: 0,
    borderBottom: "2px solid #e2e8f0",
    marginBottom: 24,
  },
  tab: {
    border: "none",
    background: "transparent",
    padding: "10px 20px",
    fontSize: 14,
    fontWeight: 600,
    color: "#64748b",
    cursor: "pointer",
    borderBottom: "2px solid transparent",
    marginBottom: -2,
  },
  tabActive: {
    color: "#3b82f6",
    borderBottomColor: "#3b82f6",
  },
  content: {
    minHeight: 300,
  },
  footer: {
    marginTop: 32,
    paddingTop: 16,
    borderTop: "1px solid #e2e8f0",
  },
  backLink: {
    color: "#3b82f6",
    textDecoration: "none",
    fontSize: 14,
    fontWeight: 500,
  },
};
