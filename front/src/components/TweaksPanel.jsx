import { Icons } from "./Icons.jsx";

const segBtn = (active) => ({
  flex: 1, padding: "8px 10px", borderRadius: 8,
  background: active ? "var(--brand-wash)" : "var(--bg-sunk)",
  border: `1px solid ${active ? "rgba(79,70,229,0.3)" : "var(--line)"}`,
  color: active ? "var(--brand)" : "var(--ink-3)",
  fontSize: 12, fontWeight: 600,
  display: "flex", alignItems: "center", justifyContent: "center", gap: 4,
});

export function TweaksPanel({ state, setState, show }) {
  if (!show) return null;

  const persist = (edits) => {
    try {
      window.parent.postMessage({ type: "__edit_mode_set_keys", edits }, "*");
    } catch (e) {
      // noop when not in design iframe
    }
  };

  const set = (k, v) => {
    setState((s) => ({ ...s, [k]: v }));
    persist({ [k]: v });
  };

  return (
    <div style={{
      position: "absolute", right: 24, bottom: 56, zIndex: 200,
      width: 280,
      background: "var(--bg-elev)", borderRadius: 16,
      border: "1px solid var(--line)",
      boxShadow: "var(--shadow-lg)",
      overflow: "hidden",
    }}>
      <div style={{
        padding: "14px 18px",
        borderBottom: "1px solid var(--line-soft)",
        display: "flex", alignItems: "center", gap: 10,
      }}>
        <Icons.sparkle size={14} color="var(--brand)" />
        <div style={{ fontSize: 13, fontWeight: 700 }}>Tweaks</div>
      </div>
      <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 14 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: "var(--ink-3)", marginBottom: 6 }}>THEME</div>
          <div style={{ display: "flex", gap: 6 }}>
            {[
              { k: "light", label: "Light", icon: <Icons.sun size={12} /> },
              { k: "dark", label: "Dark", icon: <Icons.moon size={12} /> },
            ].map((t) => (
              <button key={t.k} onClick={() => set("theme", t.k)} style={segBtn(state.theme === t.k)}>
                {t.icon}{t.label}
              </button>
            ))}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: "var(--ink-3)", marginBottom: 6 }}>MAP STYLE</div>
          <div style={{ display: "flex", gap: 6 }}>
            {[
              { k: "light", label: "Light" },
              { k: "dark", label: "Dark" },
              { k: "satellite", label: "Satellite" },
            ].map((t) => (
              <button key={t.k} onClick={() => set("mapStyle", t.k)} style={segBtn(state.mapStyle === t.k)}>
                {t.label}
              </button>
            ))}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: "var(--ink-3)", marginBottom: 6 }}>LIVE LOG STREAM</div>
          <button
            onClick={() => set("autoPlay", !state.autoPlay)}
            style={{
              width: "100%", display: "flex", alignItems: "center", justifyContent: "space-between",
              padding: "8px 12px", borderRadius: 8,
              background: "var(--bg-sunk)", border: "1px solid var(--line)",
              fontSize: 12, fontWeight: 600, color: "var(--ink-2)",
            }}
          >
            {state.autoPlay ? "실시간 갱신 ON" : "실시간 갱신 OFF"}
            <span style={{
              width: 32, height: 18, borderRadius: 999, position: "relative",
              background: state.autoPlay ? "var(--brand)" : "var(--ink-4)",
              transition: "background 160ms",
            }}>
              <span style={{
                position: "absolute", top: 2, left: state.autoPlay ? 16 : 2,
                width: 14, height: 14, borderRadius: "50%", background: "#fff",
                transition: "left 160ms",
              }} />
            </span>
          </button>
        </div>
      </div>
    </div>
  );
}
