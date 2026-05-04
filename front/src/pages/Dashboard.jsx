import { useState, useEffect, useRef, useMemo } from "react";
import { Icons } from "../components/Icons.jsx";
import { MapPanel } from "../components/MapPanel.jsx";
import { devicesToMarkers } from "../api/client.js";

const statusChip = (status) => {
  const map = {
    normal:   { ko: "정상", fg: "#047857", bg: "rgba(16,185,129,0.14)", bd: "rgba(16,185,129,0.3)" },
    critical: { ko: "위험", fg: "#fff",     bg: "#dc2626",                bd: "#991b1b" },
    anomaly:  { ko: "이상", fg: "#b91c1c", bg: "rgba(239,68,68,0.12)",   bd: "rgba(239,68,68,0.3)" },
    warn:     { ko: "관찰", fg: "#b45309", bg: "rgba(245,158,11,0.14)",  bd: "rgba(245,158,11,0.3)" },
    offline:  { ko: "장애", fg: "#475569", bg: "rgba(100,116,139,0.14)", bd: "rgba(100,116,139,0.3)" },
  };
  return map[status] || map.normal;
};

function Kpi({ label, value, accent, icon, delta, active, onClick, danger }) {
  // 위험(danger=true) 카드는 0건이 아닐 때만 강조 (배경/펄스).
  const alarming = danger && value > 0;
  const cardBg   = alarming ? "linear-gradient(135deg, #dc2626, #991b1b)" : "var(--bg-elev)";
  const labelFg  = alarming ? "rgba(255,255,255,0.95)" : "var(--ink-3)";
  const valueFg  = alarming ? "#fff" : "var(--ink)";
  const subFg    = alarming ? "rgba(255,255,255,0.85)" : "var(--ink-3)";
  const iconCol  = alarming ? "rgba(255,255,255,0.95)" : accent;
  const stroke   = alarming ? "rgba(255,255,255,0.55)" : accent;
  return (
    <button
      onClick={onClick}
      style={{
        position: "relative", flex: 1, minWidth: 0, height: 112, textAlign: "left",
        background: cardBg, borderRadius: 16,
        border: `1px solid ${alarming ? "#7f1d1d" : (active ? accent : "var(--line)")}`,
        padding: "18px 20px",
        boxShadow: alarming
          ? `0 0 0 3px rgba(220,38,38,0.25), 0 12px 28px -10px rgba(220,38,38,0.55), var(--shadow-card)`
          : (active ? `0 0 0 3px ${accent}22, var(--shadow-card)` : "var(--shadow-card)"),
        cursor: "pointer", transition: "all 180ms",
        overflow: "hidden",
        animation: alarming ? "danger-pulse 1.6s ease-in-out infinite" : "none",
      }}
    >
      <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: 4, background: alarming ? "rgba(255,255,255,0.9)" : accent }} />
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: labelFg, letterSpacing: "0.02em" }}>{label}</div>
        <div style={{ color: iconCol, opacity: 0.95 }}>{icon}</div>
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <div className="num" style={{ fontSize: 44, fontWeight: 700, lineHeight: 1, color: valueFg }}>{value}</div>
        <div style={{ fontSize: 13, color: subFg, fontWeight: 500 }}>대</div>
        {delta && (
          <div className="mono" style={{
            marginLeft: "auto", fontSize: 11, fontWeight: 700,
            color: delta.startsWith("+") ? "var(--err)" : "var(--ok)",
          }}>{delta}</div>
        )}
      </div>
      <svg viewBox="0 0 200 30" style={{ position: "absolute", right: 14, bottom: 10, width: 130, height: 26, opacity: alarming ? 0.4 : 0.5 }}>
        <polyline
          fill="none"
          stroke={stroke}
          strokeWidth="1.5"
          points="0,20 20,18 40,22 60,12 80,16 100,8 120,14 140,6 160,10 180,4 200,12"
        />
      </svg>
    </button>
  );
}

function KPIRow({ active, setActive, counts }) {
  const items = [
    { k: "all",      label: "총 장비",   value: counts.all,      accent: "var(--brand)", icon: <Icons.box size={18} /> },
    { k: "normal",   label: "정상",      value: counts.normal,   accent: "var(--ok)",    icon: <Icons.check size={18} /> },
    { k: "critical", label: "위험",      value: counts.critical, accent: "#dc2626",      icon: <Icons.alert size={18} />, danger: true },
    { k: "anomaly",  label: "이상 의심", value: counts.anomaly,  accent: "var(--err)",   icon: <Icons.alert size={18} /> },
    { k: "warn",     label: "관찰 필요", value: counts.warn,     accent: "var(--warn)",  icon: <Icons.eye size={18} /> },
    { k: "offline",  label: "통신 장애", value: counts.offline,  accent: "var(--ink-3)", icon: <Icons.wifi_off size={18} /> },
  ];
  return (
    <div style={{ display: "flex", gap: 12 }}>
      {items.map((i) => (
        <Kpi key={i.k} {...i} active={active === i.k} onClick={() => setActive(i.k === active ? null : i.k)} />
      ))}
    </div>
  );
}

function PanelHeader({ children, right }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "16px 20px", borderBottom: "1px solid var(--line-soft)",
    }}>
      {children}
      {right}
    </div>
  );
}

function Panel({ children, style, className }) {
  return (
    <div
      className={className}
      style={{
        background: "var(--bg-elev)", borderRadius: 16,
        border: "1px solid var(--line)",
        boxShadow: "var(--shadow-card)",
        overflow: "hidden", ...style,
      }}
    >
      {children}
    </div>
  );
}

function AnomalyCard({ item, onClick, kind }) {
  const color = kind === "warn" ? "var(--warn)" : "var(--err)";
  return (
    <div
      onClick={() => onClick(item)}
      role="button"
      tabIndex={0}
      style={{
        display: "block", width: "100%", textAlign: "left",
        padding: "10px 12px",
        background: "var(--bg-sunk)", border: "1px solid var(--line-soft)",
        borderRadius: 10, marginBottom: 6,
        transition: "all 160ms", cursor: "pointer",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, marginBottom: 8 }}>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 0 }}>
            <span className="mono" style={{ fontSize: 11, fontWeight: 700, color: "var(--ink)", whiteSpace: "nowrap" }}>{item.node}</span>
            <span style={{
              fontSize: 9, color: "var(--ink-4)", fontWeight: 600,
              padding: "1px 5px", borderRadius: 3,
              background: "var(--bg-elev)", border: "1px solid var(--line)",
              flexShrink: 0,
            }}>
              {item.zone}
            </span>
          </div>
          <div style={{
            fontSize: 11, color, marginTop: 3, fontWeight: 600,
            whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
          }}>
            {item.label}
          </div>
        </div>
        <div style={{ textAlign: "right", flexShrink: 0 }}>
          <div className="mono" style={{ fontSize: 9, color: "var(--ink-4)", letterSpacing: "0.05em" }}>MSE</div>
          <div className="mono num" style={{ fontSize: 14, fontWeight: 700, color, lineHeight: 1 }}>{item.mse.toFixed(3)}</div>
        </div>
      </div>
      {item.contribution && item.contribution.length > 0 && (
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {item.contribution.map((c, i) => (
            <span
              key={i}
              style={{
                fontSize: 9, fontWeight: 600,
                padding: "2px 6px", borderRadius: 4,
                background: i === 0
                  ? color.replace("var(--err)", "rgba(239,68,68,0.15)").replace("var(--warn)", "rgba(245,158,11,0.15)")
                  : "var(--bg-elev)",
                color: i === 0 ? color : "var(--ink-3)",
                border: `1px solid ${i === 0 ? color : "var(--line)"}`,
                whiteSpace: "nowrap",
              }}
            >
              {c.sensor} {c.pct}%
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function AIPanels({ onAnalyze, anomalies, watch }) {
  // 이상 + 관찰 통합 리스트. MSE 내림차순 → 자연스러운 우선순위
  const combined = [
    ...anomalies.map((a) => ({ ...a, _kind: "anomaly" })),
    ...watch.map((w) => ({ ...w, _kind: "warn" })),
  ].sort((a, b) => b.mse - a.mse);

  return (
    <Panel style={{ height: "100%", display: "flex", flexDirection: "column", minHeight: 0 }}>
      <PanelHeader
        right={
          <div style={{ display: "flex", gap: 6 }}>
            <span style={{
              fontSize: 10, fontWeight: 700, padding: "2px 10px",
              background: "rgba(239,68,68,0.12)", color: "var(--err)",
              borderRadius: 999,
            }}>
              이상 {anomalies.length}건
            </span>
            <span style={{
              fontSize: 10, fontWeight: 700, padding: "2px 10px",
              background: "rgba(245,158,11,0.12)", color: "var(--warn)",
              borderRadius: 999,
            }}>
              관찰 {watch.length}건
            </span>
          </div>
        }
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Icons.alert size={16} color="var(--err)" />
          <div style={{ fontSize: 14, fontWeight: 700 }}>AI 탐지 목록</div>
          <span style={{ fontSize: 11, color: "var(--ink-3)", fontWeight: 500 }}>
            · MSE 내림차순
          </span>
        </div>
      </PanelHeader>
      <div className="scroll" style={{ padding: 12, flex: 1, overflowY: "auto", minHeight: 0 }}>
        {combined.map((a) => (
          <AnomalyCard key={a.node} item={a} kind={a._kind} onClick={onAnalyze} />
        ))}
      </div>
    </Panel>
  );
}

// 핵심 수치·장비 자동 강조 (TB24-XXXXXX, NN%, N시간, 제N구역 등)
function highlightBody(text) {
  const pattern = /(TB24-[\w-]+|\d+\.\d{3,4}|\d+\.?\d*%|\d+시간|\d+분|제\d+구역|\d+mV|\d+mA|\d+dBm)/g;
  const parts = text.split(pattern);
  return parts.map((part, i) =>
    pattern.test(part) ? (
      <strong key={i} style={{ color: "var(--ink)", fontWeight: 700 }}>{part}</strong>
    ) : (
      <span key={i}>{part}</span>
    )
  );
}

function AIAdvicePanel({ insights, onClickInsight }) {
  return (
    <Panel style={{ height: "100%", display: "flex", flexDirection: "column", minHeight: 0 }}>
      <PanelHeader
        right={
          <div style={{
            display: "flex", alignItems: "center", gap: 6,
            padding: "2px 10px", borderRadius: 999,
            background: "rgba(79,70,229,0.1)", color: "var(--brand)",
            fontSize: 10, fontWeight: 700,
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: "50%", background: "var(--brand)",
              animation: "pulse-dot 1.2s infinite",
            }} />
            분석 엔진 가동중
          </div>
        }
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Icons.sparkle size={16} color="var(--brand)" />
          <div style={{ fontSize: 14, fontWeight: 700 }}>AI 조치 권고</div>
        </div>
      </PanelHeader>
      <div className="scroll" style={{ padding: 12, overflowY: "auto", flex: 1 }}>
        {insights.map((it, i) => {
          // body 에서 첫 번째 노드 ID 추출 (TB24-5JN001 형식)
          const m = (it.body || "").match(/TB24-5JN\d+/);
          const node = m ? m[0] : null;
          const clickable = !!node && !!onClickInsight;
          return (
            <div
              key={i}
              onClick={clickable ? () => onClickInsight(node) : undefined}
              role={clickable ? "button" : undefined}
              tabIndex={clickable ? 0 : undefined}
              title={clickable ? `지도에서 ${node} 위치로 이동` : undefined}
              style={{
                display: "flex", gap: 12,
                padding: "12px 14px", marginBottom: 6,
                background: "var(--bg-sunk)",
                border: "1px solid var(--line-soft)",
                borderRadius: 10,
                alignItems: "flex-start",
                cursor: clickable ? "pointer" : "default",
                transition: "border-color 140ms ease, background 140ms ease",
              }}
              onMouseEnter={(e) => {
                if (clickable) e.currentTarget.style.borderColor = "var(--brand)";
              }}
              onMouseLeave={(e) => {
                if (clickable) e.currentTarget.style.borderColor = "var(--line-soft)";
              }}
            >
              <div style={{
                flexShrink: 0,
                width: 24, height: 24, borderRadius: "50%",
                background: "var(--brand-wash)", color: "var(--brand)",
                display: "grid", placeItems: "center",
                fontSize: 12, fontWeight: 800,
                fontFamily: "Space Grotesk, system-ui, sans-serif",
                marginTop: 1,
              }}>
                {i + 1}
              </div>
              <div style={{
                fontSize: 13, color: "var(--ink-2)", lineHeight: 1.6, flex: 1,
                letterSpacing: "-0.01em",
              }}>
                {highlightBody(it.body)}
              </div>
            </div>
          );
        })}
      </div>
    </Panel>
  );
}

const mapBtn = {
  width: 32, height: 32, display: "grid", placeItems: "center",
  color: "var(--ink-2)",
};

function Metric({ label, value, color, mono }) {
  return (
    <div>
      <div style={{ fontSize: 10, color: "var(--ink-4)", fontWeight: 500 }}>{label}</div>
      <div className={mono ? "mono" : ""} style={{ fontSize: 12, fontWeight: 700, color: color || "var(--ink)" }}>{value}</div>
    </div>
  );
}

function MarkerPopup({ m, onClose }) {
  const color = m.status === "anomaly" ? "var(--err)" : "var(--warn)";
  return (
    <div style={{
      position: "absolute",
      left: `${m.x * 100}%`, top: `${m.y * 100}%`,
      transform: "translate(-50%, calc(-100% - 50px))",
      background: "var(--bg-elev)",
      border: "1px solid var(--line-soft)",
      borderRadius: 12,
      boxShadow: "0 20px 40px -10px rgba(0,0,0,0.25)",
      padding: 14, minWidth: 240,
      animation: "slide-in-up 220ms ease",
      zIndex: 10,
      color: "var(--ink)",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: 8 }}>
        <div>
          <div className="mono" style={{ fontSize: 13, fontWeight: 700, color: "var(--ink)" }}>{m.node}</div>
          <div style={{ fontSize: 11, color, fontWeight: 600, marginTop: 2 }}>{m.label}</div>
        </div>
        <button onClick={onClose} style={{ color: "var(--ink-3)" }}><Icons.close size={14} /></button>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginTop: 8 }}>
        <Metric label="MSE" value={m.mse.toFixed(3)} color={color} />
        <Metric label="구역" value={m.zone} />
        <Metric label="상태" value={m.status === "anomaly" ? "이상" : "관찰"} color={color} />
      </div>
      <div style={{
        marginTop: 10, padding: "8px 10px", borderRadius: 8,
        background: "var(--brand-wash)", color: "var(--brand)",
        fontSize: 11, fontWeight: 600, display: "flex", alignItems: "center", gap: 6,
      }}>
        <Icons.sparkle size={12} />
        AI 이상 스코어 적용 · 실시간 분석 중
      </div>
      <div style={{
        position: "absolute", left: "50%", bottom: -6,
        transform: "translateX(-50%) rotate(45deg)",
        width: 12, height: 12, background: "var(--bg-elev)",
        borderRight: "1px solid var(--line-soft)",
        borderBottom: "1px solid var(--line-soft)",
      }} />
    </div>
  );
}

function MapPanelWrap({ markers, onMarker, mapStyle, setMapStyle, focus, fitTrigger }) {
  return (
    <Panel style={{ position: "relative", height: "100%", isolation: "isolate" }}>
      <MapPanel markers={markers} onMarker={onMarker} mapStyle={mapStyle} focus={focus} fitTrigger={fitTrigger} />

      {/* Legend */}
      <div style={{
        position: "absolute", left: 16, top: 16, zIndex: 1000,
        background: "var(--bg-elev)", backdropFilter: "blur(10px)",
        border: "1px solid var(--line-soft)", borderRadius: 10,
        padding: "10px 14px",
        boxShadow: "0 8px 24px -10px rgba(0,0,0,0.2)",
        color: "var(--ink)",
        pointerEvents: "none",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--ok)", animation: "pulse-dot 1.2s infinite" }} />
          <div style={{ fontSize: 12, fontWeight: 800 }}>GIS 관제</div>
        </div>
        <div style={{ display: "flex", gap: 12, fontSize: 10, color: "var(--ink-3)" }}>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--brand)" }} />정상
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--err)" }} />이상
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--warn)" }} />관찰
          </span>
        </div>
      </div>

      {/* 지도 스타일 스위처는 Header 설정 아이콘 드롭다운으로 이동 (2026-05-04) */}
    </Panel>
  );
}

function MiniTable({ data, onRowClick }) {
  const [hoverId, setHoverId] = useState(null);

  return (
    <div className="scroll" style={{ overflow: "auto", height: "100%" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, tableLayout: "fixed" }}>
        <colgroup>
          <col style={{ width: 130 }} />
          <col style={{ width: 130 }} />
          <col />
          <col style={{ width: 96 }} />
        </colgroup>
        <thead>
          <tr style={{ position: "sticky", top: 0, background: "var(--bg-elev)", zIndex: 1 }}>
            {["시설명", "장비명", "설치위치", "상태"].map((h, i) => {
              const align = i === 3 ? "center" : "left";
              return (
                <th
                  key={h}
                  style={{
                    textAlign: align,
                    padding: "14px 16px",
                    fontWeight: 700, fontSize: 12, color: "var(--ink-3)",
                    borderBottom: "1px solid var(--line)",
                    background: "var(--bg-elev)",
                    whiteSpace: "nowrap",
                    letterSpacing: "-0.01em",
                  }}
                >
                  {h}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {data.map((r, idx) => {
            const c = statusChip(r.status);
            const isHover = hoverId === r.id;
            const isZebra = idx % 2 === 1;
            return (
              <tr
                key={r.id}
                onClick={() => onRowClick && onRowClick(r)}
                onMouseEnter={() => setHoverId(r.id)}
                onMouseLeave={() => setHoverId(null)}
                style={{
                  borderBottom: "1px solid var(--line-soft)",
                  cursor: "pointer",
                  background: isHover
                    ? "var(--brand-wash)"
                    : isZebra
                    ? "var(--bg-sunk)"
                    : "transparent",
                  transition: "background 120ms",
                }}
              >
                <td className="mono" style={{
                  padding: "14px 16px", fontWeight: 700,
                  color: "var(--ink)", letterSpacing: "-0.01em",
                }}>
                  {r.facilityId}
                </td>
                <td className="mono" style={{
                  padding: "14px 16px", color: "var(--ink-2)",
                  fontWeight: 500,
                }}>
                  {r.deviceId}
                </td>
                <td style={{
                  padding: "14px 16px", color: "var(--ink-2)",
                  overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                }}>
                  {r.location}
                </td>
                <td style={{ padding: "14px 16px", textAlign: "center" }}>
                  <span style={{
                    display: "inline-block", padding: "4px 12px", borderRadius: 999,
                    fontSize: 12, fontWeight: 700,
                    background: c.bg, color: c.fg, border: `1px solid ${c.bd}`,
                    minWidth: 48,
                  }}>
                    {c.ko}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function TableSummary({ data, onRowClick }) {
  const [query, setQuery] = useState("");
  const filtered = data.filter(
    (r) =>
      !query ||
      r.facilityId.toLowerCase().includes(query.toLowerCase()) ||
      r.deviceId.toLowerCase().includes(query.toLowerCase()) ||
      r.location.includes(query)
  );
  return (
    <Panel style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <PanelHeader
        right={
          <div style={{
            display: "flex", alignItems: "center", gap: 8,
            padding: "6px 12px", borderRadius: 10,
            background: "var(--bg-sunk)", border: "1px solid var(--line)",
            width: 240,
          }}>
            <Icons.search size={13} color="var(--ink-4)" />
            <input
              placeholder="장비 검색..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              style={{
                flex: 1, border: "none", outline: "none", background: "transparent",
                fontSize: 12,
              }}
            />
          </div>
        }
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ width: 4, height: 18, background: "var(--brand)", borderRadius: 2 }} />
          <div style={{ fontSize: 14, fontWeight: 700 }}>전체 장비 현황 요약</div>
          <span style={{
            fontSize: 10, fontWeight: 700, padding: "2px 8px",
            background: "var(--brand-wash)", color: "var(--brand)", borderRadius: 999,
          }}>
            {filtered.length}개
          </span>
        </div>
      </PanelHeader>
      <div style={{ flex: 1, overflow: "hidden" }}>
        <MiniTable data={filtered} onRowClick={onRowClick} />
      </div>
    </Panel>
  );
}

function LogLine({ line }) {
  const colors = {
    ok: "var(--ok)",
    data: "var(--ink-2)",
    alert: "var(--err)",
    ai: "var(--brand)",
    auth: "var(--ink-3)",
    warn: "var(--warn)",
  };
  const bg =
    line.kind === "alert" ? "rgba(239,68,68,0.08)" :
    line.kind === "warn"  ? "rgba(245,158,11,0.08)" : "transparent";
  const border =
    line.kind === "alert" ? "1px solid rgba(239,68,68,0.3)" :
    line.kind === "warn"  ? "1px solid rgba(245,158,11,0.3)" : "1px solid transparent";
  return (
    <div
      className="mono"
      style={{
        padding: "4px 10px", borderRadius: 6, marginBottom: 3,
        fontSize: 11, display: "flex", gap: 10, alignItems: "center",
        background: bg, border: border,
        animation: "slide-in-up 220ms ease",
      }}
    >
      <span style={{ color: colors[line.kind] || "var(--ink-3)", fontWeight: 700, flexShrink: 0 }}>
        [{line.time}]
      </span>
      <span style={{
        flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
        color: line.kind === "alert" || line.kind === "warn" ? colors[line.kind] : "var(--ink-2)",
        fontWeight: line.kind === "alert" ? 700 : 400,
      }}>
        {line.text}
      </span>
      {line.tail && (
        <span style={{ color: colors[line.kind] || "var(--ok)", fontWeight: 700, flexShrink: 0 }}>
          {line.tail}
        </span>
      )}
    </div>
  );
}

function LogPanel({ lines }) {
  return (
    <Panel style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <PanelHeader>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{
            width: 8, height: 8, borderRadius: "50%", background: "var(--ok)",
            animation: "pulse-dot 1.2s infinite",
          }} />
          <div style={{ fontSize: 13, fontWeight: 700 }}>실시간 시스템 로그</div>
          <span className="mono" style={{ fontSize: 10, color: "var(--ink-3)", marginLeft: 4 }}>{lines.length} EVENTS</span>
        </div>
      </PanelHeader>
      <div className="scroll" style={{
        padding: 10, flex: 1, overflow: "auto",
        background: "var(--bg-sunk)",
      }}>
        {lines.map((l) => <LogLine key={l.id} line={l} />)}
      </div>
    </Panel>
  );
}

// ── AI 분석 어시스턴트 (mock 챗봇) ───────────────────────────
//   현재 LLM 미연동 — 키워드/노드 ID 매칭 기반 응답.
//   실제 백엔드 연결 시 mockAIResponse → fetch("/api/chat") 으로 교체 예정.

const STATUS_KO_BY_KEY = { normal: "정상", critical: "위험", anomaly: "이상 의심", warn: "관찰 필요", offline: "통신 장애" };

function mockAIResponse(input, ctx = {}) {
  const equipment = ctx.equipment || [];
  const text = (input || "").trim();
  const lower = text.toLowerCase();

  // 1) 노드 ID 직접 조회
  const nodeMatch = text.match(/TB24-5JN\d+/i);
  if (nodeMatch) {
    const node = nodeMatch[0].toUpperCase();
    const eq = equipment.find((e) => e.deviceId === node);
    if (!eq) return `${node} 는 등록된 장비가 아닙니다.`;
    const lines = [
      `📍 ${node} (${eq.zone || "-"})`,
      `• 상태: ${STATUS_KO_BY_KEY[eq.status] || eq.status}`,
      `• MSE: ${eq.mse != null ? eq.mse.toFixed(3) : "—"} (임계 ${eq.threshold ?? 0.409})`,
      `• 최근 라벨: ${eq.label || "정상"}`,
      eq.contribution?.length ? `• 기여도 1순위: ${eq.contribution[0].sensor} ${eq.contribution[0].pct}%` : null,
      `• 마지막 갱신: ${eq.updatedAt || "—"}`,
    ].filter(Boolean);
    return lines.join("\n");
  }

  // 2) 위험/이상/관찰 키워드 → 현재 목록
  if (/위험|critical/.test(lower)) {
    const c = equipment.filter((e) => e.status === "critical");
    if (c.length === 0) return "현재 위험 단계 장비가 없습니다.";
    return `🚨 위험 ${c.length}건:\n${c.map((e) => `• ${e.deviceId} · ${e.zone} — ${e.label}`).join("\n")}`;
  }
  if (/이상|의심|anomaly/.test(lower)) {
    const a = equipment.filter((e) => e.status === "anomaly");
    if (a.length === 0) return "현재 이상 의심 장비가 없습니다.";
    return `⚠️ 이상 의심 ${a.length}건:\n${a.slice(0, 6).map((e) => `• ${e.deviceId} · ${e.zone} — ${e.label}`).join("\n")}`;
  }
  if (/관찰|watch|warn/.test(lower)) {
    const w = equipment.filter((e) => e.status === "warn");
    if (w.length === 0) return "현재 관찰 필요 장비가 없습니다.";
    return `👁 관찰 필요 ${w.length}건:\n${w.slice(0, 6).map((e) => `• ${e.deviceId} · ${e.zone} — ${e.label}`).join("\n")}`;
  }
  if (/장애|offline|통신/.test(lower)) {
    const o = equipment.filter((e) => e.status === "offline");
    if (o.length === 0) return "현재 통신 장애 장비가 없습니다.";
    return `📵 통신 장애 ${o.length}건:\n${o.map((e) => `• ${e.deviceId} · ${e.zone}`).join("\n")}`;
  }
  if (/요약|상태|summary|현황/.test(lower)) {
    const c = { critical: 0, anomaly: 0, warn: 0, normal: 0, offline: 0 };
    equipment.forEach((e) => { if (c[e.status] !== undefined) c[e.status]++; });
    return `📊 전체 ${equipment.length}대\n• 위험 ${c.critical}대 · 이상 의심 ${c.anomaly}대\n• 관찰 ${c.warn}대 · 통신장애 ${c.offline}대\n• 정상 ${c.normal}대`;
  }

  // 3) 도메인 용어 설명
  if (/방식전위|방식 전위|cathodic/.test(lower)) {
    return "방식전위(P/S Potential)는 매설배관의 전기방식 효율 지표입니다. 일반 기준 -850 mV 이하면 양호, 초과하면 부식 진행 가능. 본 시스템은 일교차·AC 유입과 함께 추세 분석.";
  }
  if (/희생전류|희생양극|sacrificial/.test(lower)) {
    return "희생양극은 배관 대신 부식되어 본 배관을 보호합니다. 희생전류가 점차 감소하면 양극 소모 또는 접속부 불량 가능성. 1mA 이하면 교체 검토 필요.";
  }
  if (/ac\s*유입|ac\b/.test(lower)) {
    return "AC 유입은 인접 송전선·전철 등에서 유도되는 교류 전압. 200 mV 이상은 가속 부식 위험, 500 mV 이상은 즉각 차폐 또는 배수장치 점검 필요.";
  }
  if (/통신품질|dbm|rssi/.test(lower)) {
    return "통신 품질은 노드 신호 세기(dBm). -65 이상 양호, -75 이하 주의, -85 이하 통신 두절 임박. 게이트웨이 위치·안테나 점검.";
  }
  if (/임계|threshold|mse/.test(lower)) {
    return "MSE 임계값(현재 0.409) 초과 시 이상으로 분류. 0.85 이상은 위험, 0.42~0.85 이상 의심, 0.28~0.42 관찰. 임계는 모델 학습 시 결정.";
  }

  // 4) 도움말
  if (/도움|help|\?$|메뉴/.test(lower)) {
    return "사용 예시:\n• 'TB24-5JN042' 특정 장비 조회\n• '위험' / '이상' / '관찰' / '장애' 현재 목록\n• '요약' 전체 상태\n• '방식전위' / '희생전류' / 'AC유입' 도메인 설명";
  }

  // 5) fallback
  return `"${text}" — 아직 LLM 미연동 상태라 일반 응답이 어렵습니다.\n노드 ID(예: TB24-5JN042) 또는 도메인 키워드로 질문해 주세요. '도움'을 입력하면 사용법을 안내합니다.`;
}

function ChatPanel({ equipment = [] }) {
  const initialTime = (() => { const d = new Date(); return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`; })();
  const [messages, setMessages] = useState([
    { role: "ai", text: "안녕하세요. AI 분석 어시스턴트입니다.\n노드 ID 또는 키워드(위험/이상/방식전위 등)로 질문해 주세요.", time: initialTime },
  ]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const listRef = useRef(null);

  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [messages, sending]);

  const send = (e) => {
    e && e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || sending) return;
    const now = new Date();
    const time = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
    setMessages((m) => [...m, { role: "user", text: trimmed, time }]);
    setInput("");
    setSending(true);
    setTimeout(() => {
      const reply = mockAIResponse(trimmed, { equipment });
      const r = new Date();
      const rtime = `${String(r.getHours()).padStart(2, "0")}:${String(r.getMinutes()).padStart(2, "0")}`;
      setMessages((m) => [...m, { role: "ai", text: reply, time: rtime }]);
      setSending(false);
    }, 500 + Math.random() * 500);
  };

  return (
    <Panel style={{ height: "100%", display: "flex", flexDirection: "column", minHeight: 0 }}>
      <PanelHeader
        right={
          <div style={{
            display: "flex", alignItems: "center", gap: 6,
            padding: "2px 10px", borderRadius: 999,
            background: "rgba(79,70,229,0.10)", color: "var(--brand)",
            fontSize: 10, fontWeight: 700,
          }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--brand)", animation: "pulse-dot 1.2s infinite" }} />
            mock 모드
          </div>
        }
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Icons.sparkle size={16} color="var(--brand)" />
          <div style={{ fontSize: 13, fontWeight: 700 }}>AI 분석 어시스턴트</div>
        </div>
      </PanelHeader>

      <div ref={listRef} className="scroll" style={{
        flex: 1, overflow: "auto",
        padding: "12px 12px 6px",
        background: "var(--bg-sunk)",
        display: "flex", flexDirection: "column", gap: 8,
      }}>
        {messages.map((m, i) => <ChatMessage key={i} message={m} />)}
        {sending && <ChatTyping />}
      </div>

      <form onSubmit={send} style={{
        display: "flex", gap: 6,
        padding: 10,
        borderTop: "1px solid var(--line)",
        background: "var(--bg-elev)",
      }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="질문을 입력하세요…"
          disabled={sending}
          style={{
            flex: 1,
            padding: "0 12px",
            height: 36,
            background: "var(--bg-sunk)",
            border: "1px solid var(--line)",
            borderRadius: 9,
            fontSize: 13,
            color: "var(--ink)",
            outline: "none",
            fontFamily: "inherit",
          }}
        />
        <button
          type="submit"
          disabled={!input.trim() || sending}
          style={{
            padding: "0 16px",
            height: 36,
            background: !input.trim() || sending ? "var(--bg-sunk)" : "linear-gradient(135deg, #4f46e5, #8b83ff)",
            color: !input.trim() || sending ? "var(--ink-3)" : "#fff",
            fontSize: 12, fontWeight: 700,
            border: "none",
            borderRadius: 9,
            cursor: !input.trim() || sending ? "not-allowed" : "pointer",
            boxShadow: !input.trim() || sending ? "none" : "0 6px 14px -4px rgba(79,70,229,0.45)",
          }}
        >
          전송
        </button>
      </form>
    </Panel>
  );
}

function ChatMessage({ message }) {
  const isAi = message.role === "ai";
  return (
    <div style={{
      display: "flex",
      flexDirection: isAi ? "row" : "row-reverse",
      gap: 8,
      alignItems: "flex-end",
    }}>
      {isAi && (
        <div style={{
          width: 24, height: 24, borderRadius: "50%",
          background: "linear-gradient(135deg, #4f46e5, #8b83ff)",
          color: "#fff",
          display: "grid", placeItems: "center",
          flexShrink: 0,
          marginBottom: 14,
        }}>
          <Icons.sparkle size={12} />
        </div>
      )}
      <div style={{ maxWidth: "78%", display: "flex", flexDirection: "column", gap: 3 }}>
        <div style={{
          padding: "8px 11px",
          background: isAi ? "var(--bg-elev)" : "linear-gradient(135deg, #4f46e5, #8b83ff)",
          color: isAi ? "var(--ink)" : "#fff",
          border: isAi ? "1px solid var(--line)" : "none",
          borderRadius: 12,
          borderBottomLeftRadius: isAi ? 4 : 12,
          borderBottomRightRadius: isAi ? 12 : 4,
          fontSize: 12.5,
          lineHeight: 1.55,
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
          boxShadow: isAi ? "none" : "0 4px 10px -4px rgba(79,70,229,0.4)",
        }}>
          {message.text}
        </div>
        <div style={{
          fontSize: 9, color: "var(--ink-4)",
          textAlign: isAi ? "left" : "right",
          paddingLeft: isAi ? 4 : 0,
          paddingRight: isAi ? 0 : 4,
        }}>
          {message.time}
        </div>
      </div>
    </div>
  );
}

function ChatTyping() {
  return (
    <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
      <div style={{
        width: 24, height: 24, borderRadius: "50%",
        background: "linear-gradient(135deg, #4f46e5, #8b83ff)",
        color: "#fff",
        display: "grid", placeItems: "center",
        flexShrink: 0,
      }}>
        <Icons.sparkle size={12} />
      </div>
      <div style={{
        padding: "10px 14px",
        background: "var(--bg-elev)",
        border: "1px solid var(--line)",
        borderRadius: 12,
        borderBottomLeftRadius: 4,
        display: "flex", gap: 4, alignItems: "center",
      }}>
        <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--ink-3)", animation: "pulse-dot 1.2s 0s infinite" }} />
        <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--ink-3)", animation: "pulse-dot 1.2s 0.2s infinite" }} />
        <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--ink-3)", animation: "pulse-dot 1.2s 0.4s infinite" }} />
      </div>
    </div>
  );
}

function fmtTime(d) {
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
}

function useLogStream(externalEvents = []) {
  const [lines, setLines] = useState(() => {
    const now = new Date();
    const base = now.getTime() - 4000;
    return [
      { id: 1, time: fmtTime(new Date(base)),        kind: "ok", text: "SYS: 시스템 시작 · AI 엔진 초기화", tail: "OK" },
      { id: 2, time: fmtTime(new Date(base + 2000)), kind: "ai", text: "AI: LSTM-AutoEncoder 모델 로드 완료" },
      { id: 3, time: fmtTime(new Date(base + 4000)), kind: "ai", text: "AI: 백엔드 연결 대기 중..." },
    ];
  });
  const processedIds = useRef(new Set([1, 2, 3]));

  useEffect(() => {
    if (!externalEvents || externalEvents.length === 0) return;
    const fresh = externalEvents.filter((e) => !processedIds.current.has(e.id));
    if (fresh.length === 0) return;
    fresh.forEach((e) => processedIds.current.add(e.id));
    setLines((prev) => [...prev.slice(-50), ...fresh]);
  }, [externalEvents]);

  return lines;
}

function buildTrendPath(mse, threshold) {
  const thW = threshold || 0.409;
  const thA = thW * 1.5;
  const H = 140, W = 640, N = 21;
  const toY = (v) => Math.max(3, Math.min(H - 3, H - Math.max(0, Math.min(1, v)) * H));
  const startV = Math.max(0.02, mse * 0.12);
  const pts = Array.from({ length: N }, (_, i) => {
    const t = i / (N - 1);
    const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    const v = startV + (mse - startV) * ease;
    const noise = (Math.sin(i * 2.31 + mse * 7.3) * 0.018 + Math.cos(i * 1.71 + mse * 3.7) * 0.012) * mse;
    return [Math.round(t * W), Math.round(toY(v + noise))];
  });
  const lineD = pts.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x} ${y}`).join(" ");
  return {
    lineD,
    areaD: lineD + ` L ${W} ${H} L 0 ${H} Z`,
    lastX: pts[N - 1][0],
    lastY: pts[N - 1][1],
    yW: toY(thW),
    yA: toY(thA),
    thW, thA,
  };
}

export function AnalysisModal({ item, onClose }) {
  if (!item) return null;

  const isAnomaly = item._kind !== "warn";
  const color     = isAnomaly ? "var(--err)" : "var(--warn)";
  const { lineD, areaD, lastX, lastY, yW, yA, thW, thA } = buildTrendPath(item.mse, item.threshold);
  const mainSensor = item.contribution?.[0]?.sensor || "-";

  const statCards = [
    { label: "이상 스코어",  value: item.mse.toFixed(3),        accent: color },
    { label: "이상 임계값",  value: item.threshold?.toFixed(3) ?? "0.409", accent: "var(--ink-2)" },
    { label: "주요 센서",    value: mainSensor,                  accent: "var(--brand)" },
    { label: "판정",         value: isAnomaly ? "이상" : "관찰", accent: color },
  ];

  return (
    <div
      style={{
        position: "absolute", inset: 0, zIndex: 100,
        background: "rgba(10, 15, 30, 0.55)",
        backdropFilter: "blur(4px)",
        display: "grid", placeItems: "center",
        animation: "slide-in-up 200ms ease",
      }}
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "var(--bg-elev)", borderRadius: 20,
          border: "1px solid var(--line)",
          boxShadow: "var(--shadow-lg)",
          width: 760, maxHeight: "calc(100% - 80px)", overflow: "hidden",
          display: "flex", flexDirection: "column",
        }}
      >
        <div style={{
          padding: "18px 24px",
          borderBottom: "1px solid var(--line-soft)",
          display: "flex", alignItems: "center", justifyContent: "space-between",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <div style={{
              width: 40, height: 40, borderRadius: 12,
              background: isAnomaly
                ? "linear-gradient(135deg, var(--err), #ea580c)"
                : "linear-gradient(135deg, var(--warn), #d97706)",
              display: "grid", placeItems: "center", color: "#fff",
              boxShadow: isAnomaly
                ? "0 6px 14px -4px rgba(239,68,68,0.5)"
                : "0 6px 14px -4px rgba(245,158,11,0.5)",
            }}>
              <Icons.alert size={18} />
            </div>
            <div>
              <div className="mono" style={{ fontSize: 16, fontWeight: 700 }}>{item.node}</div>
              <div style={{ fontSize: 12, color: "var(--ink-3)", marginTop: 2 }}>{item.label} · {item.zone}</div>
            </div>
          </div>
          <button onClick={onClose} style={{ color: "var(--ink-3)", padding: 6 }}><Icons.close size={18} /></button>
        </div>

        <div className="scroll" style={{ padding: 24, overflowY: "auto" }}>
          {/* AI 데이터 기반 스탯 카드 */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 20 }}>
            {statCards.map((s) => (
              <div key={s.label} style={{
                padding: "12px 14px", borderRadius: 12,
                background: "var(--bg-sunk)", border: "1px solid var(--line-soft)",
              }}>
                <div style={{ fontSize: 10, color: "var(--ink-3)", fontWeight: 600 }}>{s.label}</div>
                <div className="num" style={{ fontSize: 22, fontWeight: 700, marginTop: 4, color: s.accent }}>{s.value}</div>
              </div>
            ))}
          </div>

          {/* MSE 추이 차트 (item 데이터 기반) */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
              <div style={{ fontSize: 13, fontWeight: 700 }}>MSE 추이 (이상 감지 직전 24시간)</div>
              <div style={{ display: "flex", gap: 10, fontSize: 10, color: "var(--ink-3)" }}>
                {[
                  { c: "rgba(16,185,129,0.35)", l: "정상" },
                  { c: "rgba(245,158,11,0.35)", l: `관찰 ≥${thW.toFixed(3)}` },
                  { c: "rgba(239,68,68,0.35)",  l: `이상 ≥${thA.toFixed(3)}` },
                ].map(({ c, l }) => (
                  <span key={l} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                    <span style={{ width: 10, height: 3, background: c }} />{l}
                  </span>
                ))}
              </div>
            </div>
            <div style={{
              padding: 16, borderRadius: 12,
              background: "var(--bg-sunk)", border: "1px solid var(--line-soft)",
              height: 200,
            }}>
              <svg viewBox="0 0 640 140" style={{ width: "100%", height: "100%" }}>
                <defs>
                  <linearGradient id="trend-grad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={isAnomaly ? "#ef4444" : "#f59e0b"} stopOpacity="0.4" />
                    <stop offset="100%" stopColor={isAnomaly ? "#ef4444" : "#f59e0b"} stopOpacity="0" />
                  </linearGradient>
                </defs>
                {/* 3-band 배경 */}
                <rect x="0" y="0"       width="640" height={yA}          fill="rgba(239,68,68,0.07)" />
                <rect x="0" y={yA}      width="640" height={yW - yA}     fill="rgba(245,158,11,0.07)" />
                <rect x="0" y={yW}      width="640" height={140 - yW}    fill="rgba(16,185,129,0.06)" />
                {/* Y축 레이블 */}
                <text x="4" y="12"   fontSize="9" fill="var(--ink-4)" fontFamily="JetBrains Mono">1.0</text>
                <text x="4" y={yA + 4} fontSize="9" fill="var(--err)"   fontFamily="JetBrains Mono" fontWeight="700">{thA.toFixed(3)} ── 이상</text>
                <text x="4" y={yW + 4} fontSize="9" fill="var(--warn)"  fontFamily="JetBrains Mono" fontWeight="700">{thW.toFixed(3)} ── 관찰</text>
                <text x="4" y="136"  fontSize="9" fill="var(--ink-4)" fontFamily="JetBrains Mono">0.0</text>
                {/* 임계선 */}
                <line x1="0" y1={yA} x2="640" y2={yA} stroke="var(--err)"  strokeWidth="1" strokeDasharray="4 4" opacity="0.5" />
                <line x1="0" y1={yW} x2="640" y2={yW} stroke="var(--warn)" strokeWidth="1" strokeDasharray="4 4" opacity="0.5" />
                {/* 추이 선 */}
                <path d={areaD} fill="url(#trend-grad)" />
                <path d={lineD} fill="none" stroke={isAnomaly ? "var(--err)" : "var(--warn)"} strokeWidth="2" />
                {/* 현재 MSE 포인트 */}
                <circle cx={lastX} cy={lastY} r="5" fill={isAnomaly ? "var(--err)" : "var(--warn)"} />
                <circle cx={lastX} cy={lastY} r="10" fill="none"
                  stroke={isAnomaly ? "var(--err)" : "var(--warn)"} strokeWidth="1.5" opacity="0.5">
                  <animate attributeName="r" values="5;14;5" dur="1.6s" repeatCount="indefinite" />
                  <animate attributeName="opacity" values="0.8;0;0.8" dur="1.6s" repeatCount="indefinite" />
                </circle>
                {/* 현재값 레이블 */}
                <rect x={lastX - 28} y={lastY - 22} width="56" height="16" rx="4"
                  fill={isAnomaly ? "var(--err)" : "var(--warn)"} />
                <text x={lastX} y={lastY - 11} fontSize="9" fontFamily="JetBrains Mono" fontWeight="700"
                  fill="#fff" textAnchor="middle">
                  MSE {item.mse.toFixed(3)}
                </text>
              </svg>
            </div>
          </div>

          {item.contribution && item.contribution.length > 0 && (
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>센서별 이상 기여도</div>
              <div style={{
                padding: 16, borderRadius: 12,
                background: "var(--bg-sunk)", border: "1px solid var(--line-soft)",
                display: "flex", flexDirection: "column", gap: 10,
              }}>
                {item.contribution.map((c, i) => (
                  <div key={c.sensor}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                      <span style={{ fontSize: 12, fontWeight: 600, color: i === 0 ? "var(--err)" : "var(--ink-2)" }}>{c.sensor}</span>
                      <span className="mono num" style={{ fontSize: 12, fontWeight: 700, color: i === 0 ? "var(--err)" : "var(--ink-2)" }}>{c.pct}%</span>
                    </div>
                    <div style={{ height: 6, background: "var(--bg-elev)", borderRadius: 3, overflow: "hidden" }}>
                      <div style={{
                        width: `${c.pct}%`, height: "100%",
                        background: i === 0 ? "var(--err)" : i === 1 ? "var(--warn)" : "var(--ink-3)",
                        transition: "width 400ms",
                      }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div style={{
            padding: 16, borderRadius: 12,
            background: "var(--brand-wash)",
            border: "1px solid rgba(79,70,229,0.2)",
            marginBottom: 14,
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
              <Icons.sparkle size={14} color="var(--brand)" />
              <div style={{ fontSize: 12, fontWeight: 700, color: "var(--brand)" }}>AI 분석 요약</div>
            </div>
            <div style={{ fontSize: 13, color: "var(--ink-2)", lineHeight: 1.6 }}>{item.summary}</div>
          </div>
          <div style={{
            padding: 16, borderRadius: 12,
            background: "rgba(16,185,129,0.06)",
            border: "1px solid rgba(16,185,129,0.2)",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
              <Icons.check size={14} color="var(--ok)" />
              <div style={{ fontSize: 12, fontWeight: 700, color: "var(--ok)" }}>권장 조치</div>
            </div>
            <div style={{ fontSize: 13, color: "var(--ink-2)", lineHeight: 1.6 }}>{item.action}</div>
          </div>
        </div>

        <div style={{
          padding: "14px 24px",
          borderTop: "1px solid var(--line-soft)",
          display: "flex", gap: 10, justifyContent: "flex-end",
        }}>
          <button onClick={onClose} style={{
            padding: "10px 18px", borderRadius: 10,
            background: "var(--bg-sunk)", border: "1px solid var(--line)",
            fontSize: 13, fontWeight: 600, color: "var(--ink-2)",
          }}>
            닫기
          </button>
          <button style={{
            padding: "10px 18px", borderRadius: 10,
            background: "var(--brand)", color: "#fff",
            fontSize: 13, fontWeight: 700,
            boxShadow: "0 6px 14px -4px rgba(79,70,229,0.5)",
          }}>
            점검 워크오더 생성 →
          </button>
        </div>
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────
// 방식전위 트렌드 차트 (6시간, 30분 간격 × 13점)
// 안전 범위: -850mV ~ -1200mV (국내 음극방식 기준)
// ────────────────────────────────────────────────
function VoltTrendChart({ item }) {
  if (item.status === "offline" || item.volt == null || item.volt === 0) {
    return (
      <div style={{
        height: 160, display: "flex", alignItems: "center", justifyContent: "center",
        color: "var(--ink-4)", fontSize: 12, fontFamily: "JetBrains Mono",
      }}>
        통신 두절 — 데이터 없음
      </div>
    );
  }

  const volt  = item.volt;
  const seed  = Math.abs(item.id ?? 1) + 1;
  const N     = 13;                // 30분 × 13 = 6시간
  const SAFE_HI = -850;            // 방식 최소 기준 (이보다 양극이면 위험)
  const SAFE_LO = -1200;           // 과방식 기준

  // 차트 레이아웃
  const CX = 50, CY = 12, CW = 390, CH = 110;
  const vTop = -150, vBot = -1450; // Y축 표시 범위
  const toY = (v) =>
    CY + Math.max(0, Math.min(CH, (vTop - v) / (vTop - vBot) * CH));
  const toX = (i) => CX + (i / (N - 1)) * CW;

  // 트렌드 데이터 생성 — 현재 volt 값으로 수렴하는 곡선
  const startVolt =
    item.status === "normal"
      ? volt + Math.sin(seed * 0.7) * 20
      : Math.max(-1100, Math.min(-880, -950 + Math.sin(seed * 0.7) * 40));

  const pts = Array.from({ length: N }, (_, i) => {
    const t    = i / (N - 1);
    const ease = t * t * (3 - 2 * t);
    const v    = startVolt + (volt - startVolt) * ease;
    const noise = Math.sin(i * 2.31 + seed * 0.73) * 10
                + Math.cos(i * 1.71 + seed * 1.13) * 6;
    return Math.round(v + noise);
  });

  const pathD = pts
    .map((v, i) => `${i === 0 ? "M" : "L"} ${toX(i).toFixed(1)} ${toY(v).toFixed(1)}`)
    .join(" ");
  const areaD = `${pathD} L ${toX(N - 1).toFixed(1)} ${(CY + CH).toFixed(1)} L ${toX(0).toFixed(1)} ${(CY + CH).toFixed(1)} Z`;

  // 현재 전위 상태색
  const color = volt > SAFE_HI ? "var(--err)" : volt < SAFE_LO ? "var(--warn)" : "var(--ok)";
  const stopC = volt > SAFE_HI ? "#ef4444"    : volt < SAFE_LO ? "#f59e0b"    : "#10b981";

  // 임계선 Y 좌표
  const ySafeHi = toY(SAFE_HI); // -850mV
  const ySafeLo = toY(SAFE_LO); // -1200mV
  const gradId  = `vg-${seed}`;

  // Y축 레이블
  const yLabels = [-300, -500, -850, -1000, -1200, -1400];
  // X축 시간 레이블
  const xLabels = [
    { i: 0, t: "6h전" }, { i: 2, t: "5h" }, { i: 4, t: "4h" },
    { i: 6, t: "3h"  }, { i: 8, t: "2h" }, { i: 10, t: "1h" }, { i: 12, t: "현재" },
  ];

  const lastX = toX(N - 1);
  const lastY = toY(volt);

  return (
    <svg viewBox={`0 0 ${CX + CW + 8} ${CY + CH + 30}`} style={{ width: "100%", height: "100%" }}>
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={stopC} stopOpacity="0.28" />
          <stop offset="100%" stopColor={stopC} stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* 구역 배경 (위험 / 안전 / 과방식) */}
      <rect x={CX} y={CY}       width={CW} height={ySafeHi - CY}          fill="rgba(239,68,68,0.06)" />
      <rect x={CX} y={ySafeHi}  width={CW} height={ySafeLo - ySafeHi}     fill="rgba(16,185,129,0.07)" />
      <rect x={CX} y={ySafeLo}  width={CW} height={CY + CH - ySafeLo}     fill="rgba(245,158,11,0.06)" />

      {/* 임계선 */}
      <line x1={CX} y1={ySafeHi} x2={CX + CW} y2={ySafeHi}
        stroke="var(--err)"  strokeWidth="1" strokeDasharray="4 3" opacity="0.55" />
      <line x1={CX} y1={ySafeLo} x2={CX + CW} y2={ySafeLo}
        stroke="var(--warn)" strokeWidth="1" strokeDasharray="4 3" opacity="0.55" />
      {/* 임계값 레이블 */}
      <text x={CX + CW + 3} y={ySafeHi + 3}  fontSize="7" fontFamily="JetBrains Mono" fill="var(--err)"  fontWeight="700">-850</text>
      <text x={CX + CW + 3} y={ySafeLo + 3}  fontSize="7" fontFamily="JetBrains Mono" fill="var(--warn)" fontWeight="700">-1200</text>

      {/* X축 그리드 & 레이블 */}
      {xLabels.map(({ i, t }) => (
        <g key={i}>
          <line x1={toX(i)} y1={CY} x2={toX(i)} y2={CY + CH}
            stroke="var(--line-soft)" strokeWidth="0.5" />
          <text x={toX(i)} y={CY + CH + 14}
            fontSize="8" fontFamily="JetBrains Mono" fill="var(--ink-4)"
            textAnchor="middle">
            {t}
          </text>
        </g>
      ))}

      {/* Y축 레이블 */}
      {yLabels.map((v) => (
        <text key={v} x={CX - 4} y={toY(v) + 3}
          fontSize="7.5" fontFamily="JetBrains Mono" fill="var(--ink-4)"
          textAnchor="end">
          {v}
        </text>
      ))}
      <text x={10} y={CY + CH / 2} fontSize="7.5" fontFamily="JetBrains Mono"
        fill="var(--ink-4)" textAnchor="middle"
        transform={`rotate(-90, 10, ${CY + CH / 2})`}>
        mV
      </text>

      {/* 면적 */}
      <path d={areaD} fill={`url(#${gradId})`} />
      {/* 추이선 */}
      <path d={pathD} fill="none" stroke={color} strokeWidth="1.8" strokeLinejoin="round" />

      {/* 현재 포인트 */}
      <circle cx={lastX} cy={lastY} r="4" fill={color} />
      <circle cx={lastX} cy={lastY} r="8" fill="none" stroke={color} strokeWidth="1.2" opacity="0.45">
        <animate attributeName="r" values="4;10;4" dur="2s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.6;0;0.6" dur="2s" repeatCount="indefinite" />
      </circle>
      {/* 현재값 말풍선 */}
      <rect x={lastX - 26} y={lastY - 20} width="52" height="14" rx="4" fill={color} />
      <text x={lastX} y={lastY - 10} fontSize="8" fontFamily="JetBrains Mono" fontWeight="700"
        fill="#fff" textAnchor="middle">
        {volt}mV
      </text>
    </svg>
  );
}

function DashboardEquipmentDrawer({ item, onClose }) {
  if (!item) return null;
  const c = statusChip(item.status);
  return (
    <div style={{ position: "absolute", inset: 0, zIndex: 90, pointerEvents: "none" }}>
      <div
        onClick={onClose}
        style={{
          position: "absolute", inset: 0,
          background: "rgba(10,15,30,0.3)",
          backdropFilter: "blur(2px)",
          pointerEvents: "auto",
          animation: "slide-in-up 180ms ease",
        }}
      />
      <div style={{
        position: "absolute", right: 0, top: 0, bottom: 0, width: 520,
        background: "var(--bg-elev)",
        borderLeft: "1px solid var(--line)",
        boxShadow: "var(--shadow-lg)",
        pointerEvents: "auto",
        display: "flex", flexDirection: "column",
        animation: "slide-in-up 220ms ease",
      }}>
        <div style={{
          padding: 24, borderBottom: "1px solid var(--line-soft)",
          display: "flex", justifyContent: "space-between", alignItems: "start",
        }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
              <span className="mono" style={{ fontSize: 18, fontWeight: 800 }}>{item.deviceId}</span>
              <span style={{
                padding: "3px 10px", borderRadius: 6, fontSize: 11, fontWeight: 700,
                background: c.bg, color: c.fg, border: `1px solid ${c.bd}`,
              }}>
                {c.ko}
              </span>
            </div>
            <div className="mono" style={{ fontSize: 12, color: "var(--ink-3)" }}>{item.facilityId} · {item.zone}</div>
            <div style={{ fontSize: 13, color: "var(--ink-2)", marginTop: 10 }}>{item.location}</div>
          </div>
          <button onClick={onClose} style={{ color: "var(--ink-3)" }}><Icons.close size={18} /></button>
        </div>
        <div className="scroll" style={{ padding: 24, overflowY: "auto", flex: 1 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: "var(--ink-3)", marginBottom: 10 }}>실시간 측정값</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 20 }}>
            {[
              { l: "방식전위", v: `${item.volt}mV`, a: item.status === "anomaly" && (item.label === "방식전위 이탈" || item.label === "위상차 급변") ? "var(--err)" : "var(--ok)" },
              { l: "AC 유입", v: `${item.ac.toLocaleString()}mV`, a: item.status === "anomaly" && item.label === "AC 유입 과다" ? "var(--err)" : null },
              { l: "희생전류",  v: `${item.sacrificial}mA`, a: item.status === "anomaly" && item.label === "희생전류 저하" ? "var(--err)" : null },
              { l: "온도",     v: `${item.temp}°C` },
              { l: "습도",     v: `${item.hum}%` },
              { l: "통신품질", v: item.commOk ? `${item.commDbm}dBm` : "단절", a: !item.commOk || (item.commOk && item.commDbm < -75) ? "var(--err)" : null },
            ].map((s) => (
              <div
                key={s.l}
                style={{
                  padding: "12px 14px", borderRadius: 10,
                  background: "var(--bg-sunk)", border: "1px solid var(--line-soft)",
                }}
              >
                <div style={{ fontSize: 10, color: "var(--ink-3)" }}>{s.l}</div>
                <div className="mono num" style={{ fontSize: 20, fontWeight: 700, marginTop: 2, color: s.a || "var(--ink)" }}>{s.v}</div>
              </div>
            ))}
          </div>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "var(--ink-3)" }}>방식전위 트렌드 (6시간)</div>
            <div style={{ display: "flex", gap: 10, fontSize: 9, color: "var(--ink-4)" }}>
              {[
                { c: "rgba(239,68,68,0.4)",   l: "부족 (> -850)" },
                { c: "rgba(16,185,129,0.4)",  l: "정상" },
                { c: "rgba(245,158,11,0.4)",  l: "과방식 (< -1200)" },
              ].map(({ c, l }) => (
                <span key={l} style={{ display: "flex", alignItems: "center", gap: 3 }}>
                  <span style={{ width: 8, height: 8, borderRadius: 2, background: c, flexShrink: 0 }} />{l}
                </span>
              ))}
            </div>
          </div>
          <div style={{
            padding: "8px 4px 4px", borderRadius: 10,
            background: "var(--bg-sunk)", border: "1px solid var(--line-soft)",
            height: 170, marginBottom: 20,
          }}>
            <VoltTrendChart item={item} />
          </div>
          <button style={{
            width: "100%", padding: "12px", borderRadius: 10,
            background: "var(--brand)", color: "#fff",
            fontSize: 13, fontWeight: 700,
          }}>
            상세 리포트 생성
          </button>
        </div>
      </div>
    </div>
  );
}

export function Dashboard({ onAnalyze, mapStyle, setMapStyle, theme, autoPlay = true, equipment = [], markers = [], anomalies = [], watch = [], insights = [], aiEvents = [] }) {
  const [activeKpi, setActiveKpi] = useState(null);
  const [drawer, setDrawer] = useState(null);
  const [focused, setFocused] = useState(null); // {lat, lng, node, ts}
  const [fitTrigger, setFitTrigger] = useState(0); // 카운터: 변할 때마다 지도 fit

  const counts = useMemo(() => {
    const c = { all: equipment.length, normal: 0, critical: 0, anomaly: 0, warn: 0, offline: 0 };
    equipment.forEach((e) => { if (c[e.status] !== undefined) c[e.status]++; });
    return c;
  }, [equipment]);

  const tableData = useMemo(() => {
    if (!activeKpi || activeKpi === "all") return equipment;
    return equipment.filter((e) => e.status === activeKpi);
  }, [activeKpi, equipment]);

  const filteredMarkers = useMemo(() => {
    if (!activeKpi || activeKpi === "all") return markers;
    const filtered = equipment.filter((e) => e.status === activeKpi);
    return devicesToMarkers(filtered);
  }, [activeKpi, equipment, markers]);

  // 노드 ID 로 지도 포커싱 (장비 lat/lng 우선, markers fallback)
  const focusByNode = (node) => {
    if (!node) return;
    const eq = equipment.find((e) => e.deviceId === node);
    if (eq && eq.lat != null && eq.lng != null) {
      setFocused({ lat: eq.lat, lng: eq.lng, node, ts: Date.now() });
      return;
    }
    const mk = markers.find((m) => m.node === node);
    if (mk && mk.lat != null && mk.lng != null) {
      setFocused({ lat: mk.lat, lng: mk.lng, node, ts: Date.now() });
      return;
    }
    // 둘 다 없으면 콘솔 경고 (개발 편의)
    console.warn(`[focusByNode] 노드 위치를 찾을 수 없습니다: ${node}`);
  };

  // 표 row 클릭: 드로어 열기 + 지도 포커스
  const handleRowClick = (eq) => {
    setDrawer(eq);
    if (eq && eq.lat != null && eq.lng != null) {
      setFocused({ lat: eq.lat, lng: eq.lng, node: eq.deviceId, ts: Date.now() });
    }
  };

  // AI 탐지 카드 클릭: 분석 모달 + 지도 포커스
  const handleAnalyze = (item) => {
    onAnalyze && onAnalyze(item);
    if (item && item.node) focusByNode(item.node);
  };

  // KPI 카드 클릭: 활성 토글 + 지도 fit (필터된 마커 모두 보이게)
  const handleKpiClick = (newActive) => {
    setActiveKpi(newActive);
    setFitTrigger(Date.now());
  };

  const lines = useLogStream(aiEvents);

  return (
    <>
      {/*
        통합 3-row grid:
          row 1 (auto)             — 좌: KPI                / 우: AI 탐지(span 1-2)
          row 2 (1.2fr, min 360)   — 좌: 지도               / 우: AI 탐지(이어짐)
          row 3 (1fr, min 280)     — 좌: 표+로그            / 우: AI 조치 권고
        AI 탐지가 row 1+2 를 span 하여 우측 분기점이 좌측의 지도/표 분기점과 정확히 일치.
      */}
      <div style={{
        position: "absolute", left: 0, right: 0, top: 0, bottom: 0,
        padding: 24,
        display: "grid",
        gridTemplateColumns: "1fr 520px",
        gridTemplateRows: "auto minmax(360px, 1.2fr) minmax(280px, 1fr)",
        gap: 16,
        minHeight: 0,
      }}>
        {/* (col 1, row 1) — KPI */}
        <div style={{ gridColumn: 1, gridRow: 1, minHeight: 0 }}>
          <KPIRow active={activeKpi} setActive={handleKpiClick} counts={counts} />
        </div>

        {/* (col 2, row 1+2) — AI 탐지 (span) */}
        <div style={{ gridColumn: 2, gridRow: "1 / span 2", minHeight: 0 }}>
          <AIPanels anomalies={anomalies} watch={watch} onAnalyze={handleAnalyze} />
        </div>

        {/* (col 1, row 2) — 지도 */}
        <div style={{ gridColumn: 1, gridRow: 2, minHeight: 0 }}>
          <MapPanelWrap
            markers={filteredMarkers}
            onMarker={() => {}}
            mapStyle={mapStyle}
            setMapStyle={setMapStyle}
            focus={focused}
            fitTrigger={fitTrigger}
          />
        </div>

        {/* (col 1, row 3) — 표 + 로그 */}
        <div style={{
          gridColumn: 1, gridRow: 3,
          display: "grid",
          gridTemplateColumns: "minmax(520px, 1fr) minmax(360px, 0.65fr)",
          gap: 16, minHeight: 0,
        }}>
          <TableSummary data={tableData} onRowClick={handleRowClick} />
          <ChatPanel equipment={equipment} />
        </div>

        {/* (col 2, row 3) — AI 조치 권고 */}
        <div style={{ gridColumn: 2, gridRow: 3, minHeight: 0 }}>
          <AIAdvicePanel insights={insights} onClickInsight={focusByNode} />
        </div>
      </div>
      <DashboardEquipmentDrawer item={drawer} onClose={() => setDrawer(null)} />
    </>
  );
}
