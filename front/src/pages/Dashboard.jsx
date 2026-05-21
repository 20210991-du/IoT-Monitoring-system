import { useState, useEffect, useRef, useMemo } from "react";
import { Icons } from "../components/Icons.jsx";
import { MapPanel } from "../components/MapPanel.jsx";
import { devicesToMarkers } from "../api/client.js";
import { useWeather } from "../lib/weather.js";

const statusChip = (status) => {
  const map = {
    normal:   { ko: "정상", fg: "#047857", bg: "rgba(16,185,129,0.14)", bd: "rgba(16,185,129,0.3)" },
    critical: { ko: "위험", fg: "#fff",     bg: "#dc2626",                bd: "#991b1b" },
    anomaly:  { ko: "이상", fg: "#b91c1c", bg: "rgba(239,68,68,0.12)",   bd: "rgba(239,68,68,0.3)" },
    warn:     { ko: "이상", fg: "#b45309", bg: "rgba(245,158,11,0.14)",  bd: "rgba(245,158,11,0.3)" },
    offline:  { ko: "장애", fg: "#475569", bg: "rgba(100,116,139,0.14)", bd: "rgba(100,116,139,0.3)" },
  };
  return map[status] || map.normal;
};

function Kpi({ label, value, accent, icon, delta, active, onClick, danger }) {
  // 위험(danger=true) 카드는 0건이 아닐 때 추가 강조 (bar 펄스 + 빨강 그림자).
  // 모든 카드는 자신의 status accent 색을 숫자에 적용 → 시각적 일관성.
  const alarming = danger && value > 0;
  const valueFg  = value > 0 ? accent : "var(--ink-3)";
  const iconCol  = accent;
  return (
    <button
      onClick={onClick}
      style={{
        position: "relative", flex: 1, minWidth: 0, height: 112, textAlign: "left",
        background: "var(--bg-elev)", borderRadius: 16,
        border: `1px solid ${active ? accent : "var(--line)"}`,
        padding: "18px 20px",
        boxShadow: alarming
          ? `0 0 0 1px rgba(220,38,38,0.18), 0 6px 16px -6px rgba(220,38,38,0.25), var(--shadow-card)`
          : (active ? `0 0 0 3px ${accent}22, var(--shadow-card)` : "var(--shadow-card)"),
        cursor: "pointer", transition: "all 180ms",
        overflow: "hidden",
      }}
    >
      <div style={{
        position: "absolute", left: 0, top: 0, bottom: 0,
        width: alarming ? 5 : 4,
        background: accent,
        animation: alarming ? "danger-bar-pulse 1.4s ease-in-out infinite" : "none",
      }} />
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: "var(--ink-3)", letterSpacing: "0.02em" }}>{label}</div>
        <div style={{ color: iconCol, opacity: 0.95 }}>{icon}</div>
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <div className="num" style={{ fontSize: 44, fontWeight: 700, lineHeight: 1, color: valueFg }}>{value}</div>
        <div style={{ fontSize: 13, color: "var(--ink-3)", fontWeight: 500 }}>대</div>
        {delta && (
          <div className="mono" style={{
            marginLeft: "auto", fontSize: 11, fontWeight: 700,
            color: delta.startsWith("+") ? "var(--err)" : "var(--ok)",
          }}>{delta}</div>
        )}
      </div>
      <svg viewBox="0 0 200 30" style={{ position: "absolute", right: 14, bottom: 10, width: 130, height: 26, opacity: 0.5 }}>
        <polyline
          fill="none"
          stroke={accent}
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
    { k: "warn",     label: "이상 의심", value: counts.warn,     accent: "var(--warn)",  icon: <Icons.eye size={18} /> },
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
          <div className="mono" style={{ fontSize: 14, fontWeight: 700, color, lineHeight: 1 }}>{item.mse.toFixed(3)}</div>
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
              이상 의심 {watch.length}건
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
        <Metric label="상태" value={m.status === "critical" ? "위험" : "이상 의심"} color={color} />
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

function MapPanelWrap({ markers, onMarker, mapStyle, setMapStyle, focus, fitTrigger, boundsRequest, showNormal, setShowNormal, autoKpiSec = 0, onCancelAutoKpi }) {
  return (
    <Panel style={{ position: "relative", height: "100%", isolation: "isolate" }}>
      <MapPanel markers={markers} onMarker={onMarker} mapStyle={mapStyle} focus={focus} fitTrigger={fitTrigger} boundsRequest={boundsRequest} />

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
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#10b981" }} />정상
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#991b1b" }} />위험
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--warn)" }} />이상 의심
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#64748b" }} />통신 장애
          </span>
        </div>
      </div>

      {/* 정상 핀 토글 (우상단) */}
      <button
        onClick={() => setShowNormal && setShowNormal(!showNormal)}
        title={showNormal ? "정상 핀 숨기기" : "정상 핀 표시"}
        style={{
          position: "absolute", right: 16, top: 16, zIndex: 1000,
          display: "flex", alignItems: "center", gap: 8,
          padding: "7px 14px",
          borderRadius: 999,
          background: showNormal ? "rgba(16,185,129,0.12)" : "var(--bg-elev)",
          border: `1px solid ${showNormal ? "rgba(16,185,129,0.4)" : "var(--line)"}`,
          boxShadow: "0 8px 24px -10px rgba(0,0,0,0.2)",
          fontSize: 11, fontWeight: 700,
          color: showNormal ? "#047857" : "var(--ink-3)",
          cursor: "pointer",
          transition: "all 160ms ease",
          backdropFilter: "blur(10px)",
        }}
      >
        <span style={{
          width: 8, height: 8, borderRadius: "50%",
          background: showNormal ? "#10b981" : "var(--ink-4)",
          transition: "background 160ms ease",
        }} />
        정상 {showNormal ? "표시" : "숨김"}
      </button>

      {/* AI 자동 보기 카운트다운 칩 (지도 상단 중앙) */}
      {autoKpiSec > 0 && (
        <button
          onClick={onCancelAutoKpi}
          title="클릭 시 즉시 전체 보기로 복귀"
          style={{
            position: "absolute", left: "50%", top: 16,
            transform: "translateX(-50%)",
            zIndex: 1000,
            display: "flex", alignItems: "center", gap: 5,
            padding: "4px 10px",
            borderRadius: 999,
            background: "linear-gradient(135deg, #4f46e5, #8b83ff)",
            color: "#fff", fontSize: 10, fontWeight: 700,
            border: "none", cursor: "pointer",
            boxShadow: "0 4px 12px -3px rgba(79,70,229,0.45)",
            animation: "slide-in-up 200ms ease both",
            backdropFilter: "blur(10px)",
          }}
        >
          <Icons.sparkle size={10} color="#fff" />
          <span>AI 자동 보기</span>
          <span style={{
            padding: "0 5px", borderRadius: 999,
            background: "rgba(255,255,255,0.22)",
            fontFamily: "ui-monospace, Menlo, monospace",
          }}>{autoKpiSec}s</span>
          <span style={{ opacity: 0.85 }}>· 해제</span>
        </button>
      )}

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

// KPI 활성 상태별 헤더 라벨/색상
const TABLE_HEADER_BY_KPI = {
  all:      { ko: "전체",      bar: "var(--brand)", chipBg: "var(--brand-wash)",         chipFg: "var(--brand)" },
  normal:   { ko: "정상",      bar: "#10b981",       chipBg: "rgba(16,185,129,0.12)",     chipFg: "#047857"      },
  critical: { ko: "위험",      bar: "#dc2626",       chipBg: "rgba(220,38,38,0.12)",      chipFg: "#991b1b"      },
  warn:     { ko: "이상 의심", bar: "var(--warn)",   chipBg: "rgba(245,158,11,0.14)",     chipFg: "#b45309"      },
  offline:  { ko: "통신 장애", bar: "#64748b",       chipBg: "rgba(100,116,139,0.14)",    chipFg: "#475569"      },
};

function TableSummary({ data, onRowClick, activeKpi }) {
  const [query, setQuery] = useState("");
  const filtered = data.filter(
    (r) =>
      !query ||
      r.facilityId.toLowerCase().includes(query.toLowerCase()) ||
      r.deviceId.toLowerCase().includes(query.toLowerCase()) ||
      r.location.includes(query)
  );
  const meta = TABLE_HEADER_BY_KPI[activeKpi] || TABLE_HEADER_BY_KPI.all;
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
          <span style={{
            width: 4, height: 18, background: meta.bar, borderRadius: 2,
            transition: "background 200ms ease",
          }} />
          <div style={{ fontSize: 14, fontWeight: 700 }}>
            {meta.ko} 장비 현황 요약
          </div>
          <span style={{
            fontSize: 10, fontWeight: 700, padding: "2px 8px",
            background: meta.chipBg, color: meta.chipFg, borderRadius: 999,
            transition: "background 200ms ease, color 200ms ease",
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

// ── AI 챗봇 (mock) ─────────────────────────────────────────
//   현재 LLM 미연동 — 키워드/노드 ID 매칭 기반 응답.
//   실제 백엔드 연결 시 mockAIResponse → fetch("/api/chat") 으로 교체 예정.

const STATUS_KO_BY_KEY = { normal: "정상", critical: "위험", warn: "이상 의심", offline: "통신 장애" };

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

  // 2) 위험/이상 의심 키워드 → 현재 목록
  if (/위험|critical/.test(lower)) {
    const c = equipment.filter((e) => e.status === "critical");
    if (c.length === 0) return "현재 위험 단계 장비가 없습니다.";
    return `🚨 위험 ${c.length}건:\n${c.map((e) => `• ${e.deviceId} · ${e.zone} — ${e.label}`).join("\n")}`;
  }
  if (/이상|의심|관찰|anomaly|watch|warn/.test(lower)) {
    const w = equipment.filter((e) => e.status === "warn");
    if (w.length === 0) return "현재 이상 의심 장비가 없습니다.";
    return `⚠️ 이상 의심 ${w.length}건:\n${w.slice(0, 6).map((e) => `• ${e.deviceId} · ${e.zone} — ${e.label}`).join("\n")}`;
  }
  if (/장애|offline|통신/.test(lower)) {
    const o = equipment.filter((e) => e.status === "offline");
    if (o.length === 0) return "현재 통신 장애 장비가 없습니다.";
    return `📵 통신 장애 ${o.length}건:\n${o.map((e) => `• ${e.deviceId} · ${e.zone}`).join("\n")}`;
  }
  if (/요약|상태|summary|현황/.test(lower)) {
    const c = { critical: 0, warn: 0, normal: 0, offline: 0 };
    equipment.forEach((e) => { if (c[e.status] !== undefined) c[e.status]++; });
    return `📊 전체 ${equipment.length}대\n• 위험 ${c.critical}대 · 이상 의심 ${c.warn}대\n• 통신장애 ${c.offline}대 · 정상 ${c.normal}대`;
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
    return "MSE 임계값(현재 0.409) 초과 시 이상으로 분류. 0.85 이상은 위험, 0.28~0.85 이상 의심. 임계는 모델 학습 시 결정.";
  }

  // 4) 도움말
  if (/도움|help|\?$|메뉴/.test(lower)) {
    return "사용 예시:\n• 'TB24-5JN042' 특정 장비 조회\n• '위험' / '이상 의심' / '장애' 현재 목록\n• '요약' 전체 상태\n• '방식전위' / '희생전류' / 'AC유입' 도메인 설명";
  }

  // 5) fallback
  return `"${text}" — 아직 LLM 미연동 상태라 일반 응답이 어렵습니다.\n노드 ID(예: TB24-5JN042) 또는 도메인 키워드로 질문해 주세요. '도움'을 입력하면 사용법을 안내합니다.`;
}

// 컨텍스트 추출 (equipment + weather → LLM 시스템 프롬프트용)
function buildChatContext(equipment, weather) {
  const counts = { all: equipment.length, normal: 0, critical: 0, warn: 0, offline: 0 };
  const criticalNodes = [];
  const warnNodes = [];
  const offlineNodes = [];       // 단순 ID 리스트 (기존 호환)
  const offlineDetails = [];     // 신규: 마지막 측정 시각 + 두절 시간
  const trends = []; // 위험·이상 의심 노드의 12시간 MSE 추이
  equipment.forEach((e) => {
    if (counts[e.status] !== undefined) counts[e.status]++;
    if (e.status === "critical") criticalNodes.push(e.deviceId);
    else if (e.status === "warn") warnNodes.push(e.deviceId);
    else if (e.status === "offline") {
      offlineNodes.push(e.deviceId);
      offlineDetails.push({
        deviceId:     e.deviceId,
        zone:         e.zone,
        location:     e.location,
        facilityId:   e.facilityId,
        updatedAt:    e.updatedAt,     // 마지막 측정 시각 (ISO)
        hoursSilent:  e.hoursSilent,   // 몇 시간 끊겼는지
      });
    }
    // 위험·이상 의심 노드만 trend 포함 (토큰 절약)
    if ((e.status === "critical" || e.status === "warn") && Array.isArray(e.mseHistory)) {
      trends.push({
        deviceId: e.deviceId,
        zone: e.zone,
        label: e.label,
        status: e.status,
        mse: e.mse,
        mseHistory: e.mseHistory,
      });
    }
  });
  // 시각 정보
  const now = new Date();
  const nowText = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")} ${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
  // 날씨 (있을 때만)
  const weatherCtx = weather && !weather.stale
    ? { temp: weather.temp, ko: weather.ko, code: weather.code, time: weather.time }
    : null;
  return { counts, criticalNodes, warnNodes, offlineNodes, offlineDetails, trends, nowText, weather: weatherCtx };
}

async function callLLM(message, context, history) {
  const ctrl = new AbortController();
  const timeout = setTimeout(() => ctrl.abort(), 60_000);
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, context, history }),
      signal: ctrl.signal,
    });
    clearTimeout(timeout);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "unknown");
    return { ok: true, reply: data.reply, model: data.model };
  } catch (err) {
    clearTimeout(timeout);
    return { ok: false, error: err.message };
  }
}

// SSE 스트리밍 호출
//   onDelta(piece, acc)              — 토큰 단위 콜백
//   onTool({round,name,args})        — 서버가 도구 호출했을 때 알림 (function calling)
//   onSession({sessionId})           — 서버가 부여한 chat_sessions.id
//   onDone(payload)                  — 종료 콜백
//   onError(err)                     — 에러
async function callLLMStream(message, context, history, sessionId, demoMode, { onDelta, onTool, onSession, onDone, onError, signal }) {
  let acc = "";
  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, context, history, sessionId: sessionId || undefined, demo: !!demoMode }),
      signal,
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      // SSE block: event: X\ndata: Y\n\n
      let sep;
      while ((sep = buf.indexOf("\n\n")) !== -1) {
        const block = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        const lines = block.split("\n");
        let event = "message";
        let data  = "";
        for (const ln of lines) {
          if (ln.startsWith("event:")) event = ln.slice(6).trim();
          else if (ln.startsWith("data:")) data += ln.slice(5).trim();
        }
        if (!data) continue;
        let payload;
        try { payload = JSON.parse(data); } catch { continue; }
        if (event === "delta" && payload.text) {
          acc += payload.text;
          onDelta && onDelta(payload.text, acc);
        } else if (event === "tool") {
          // 서버가 DB 도구 호출 시작 (function calling)
          onTool && onTool(payload);
        } else if (event === "session") {
          // 서버가 부여한 chat_sessions.id
          onSession && onSession(payload);
        } else if (event === "done") {
          onDone && onDone(payload);
          return { ok: true, reply: payload.reply || acc.trim(), sessionId: payload.sessionId };
        } else if (event === "error") {
          throw new Error(payload.message || "stream error");
        }
      }
    }
    // 스트림이 자연 종료 (done 이벤트 없이) — 누적된 acc 반환
    onDone && onDone({ reply: acc.trim() });
    return { ok: true, reply: acc.trim() };
  } catch (err) {
    onError && onError(err);
    return { ok: false, error: err.message };
  }
}

// 챗봇 응답에서 단일 status 키워드 감지 (정확히 1개일 때만 반환)
function detectKpiFromReply(text) {
  if (!text) return null;
  const flags = {
    critical: /위험/.test(text),
    warn:     /이상\s*의심|이상의심/.test(text),
    offline:  /통신\s*장애|통신\s*두절/.test(text),
    normal:   /정상\b|정상\s/.test(text),
  };
  const hits = Object.entries(flags).filter(([_, v]) => v).map(([k]) => k);
  return hits.length === 1 ? hits[0] : null;
}

// 채팅 히스토리 localStorage 키 + 한도
const CHAT_STORAGE_KEY = "siwon.chat.history";
const CHAT_SESSION_KEY = "siwon.chat.session_id";
const CHAT_MAX_KEEP = 60; // 최근 60개 메시지만 보관

function loadChatHistory() {
  try {
    const raw = localStorage.getItem(CHAT_STORAGE_KEY);
    if (!raw) return null;
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr) || arr.length === 0) return null;
    // 형식 검증
    return arr.filter((m) => m && typeof m.text === "string" && (m.role === "ai" || m.role === "user"));
  } catch { return null; }
}

function saveChatHistory(messages) {
  try {
    const trimmed = messages.slice(-CHAT_MAX_KEEP);
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(trimmed));
  } catch { /* ignore quota */ }
}

// 세션 목록 fetch (헤더 드롭다운용)
async function fetchChatSessions() {
  try {
    const r = await fetch("/api/chat/sessions");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    return d.ok ? (d.sessions || []) : [];
  } catch { return []; }
}
async function fetchChatSession(id) {
  try {
    const r = await fetch(`/api/chat/sessions/${id}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    return d.ok ? d : null;
  } catch { return null; }
}
async function deleteChatSession(id) {
  try {
    const r = await fetch(`/api/chat/sessions/${id}`, { method: "DELETE" });
    return r.ok;
  } catch { return false; }
}

function ChatPanel({ equipment = [], weather = null, onBotReply, onAutoKpi, demoMode = false }) {
  const initialTime = (() => { const d = new Date(); return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`; })();
  const greeting = { role: "ai", text: "안녕하세요. AI 챗봇입니다.\n노드 ID 또는 키워드(위험/이상/방식전위 등)로 질문해 주세요.", time: initialTime };
  const [messages, setMessages] = useState(() => loadChatHistory() || [greeting]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [llmActive, setLlmActive] = useState(null); // null=미확인, true=LLM, false=mock
  const [sessionId, setSessionId] = useState(() => {
    try { const v = localStorage.getItem(CHAT_SESSION_KEY); return v ? Number(v) : null; } catch { return null; }
  });
  const [showSessions, setShowSessions] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  // 서버·DB 연결 상태 (헤더 배지) — null=확인 중, true=OK, false=끊김
  const [dbActive, setDbActive] = useState(null);
  const [dbInfo,   setDbInfo]   = useState(null);   // { rows, model }
  const listRef = useRef(null);

  // 마운트 시 /api/health 1회 호출 → DB·Ollama 상태 파악
  useEffect(() => {
    let aborted = false;
    fetch("/api/health", { signal: AbortSignal.timeout(6000) })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (aborted) return;
        const ok = d?.db?.ok === true;
        setDbActive(ok);
        setDbInfo({
          rows:  d?.db?.sensor_data_rows ?? null,
          model: d?.model ?? null,
          ollama: !!d?.ollama,
        });
      })
      .catch(() => { if (!aborted) setDbActive(false); });
    return () => { aborted = true; };
  }, []);

  // 드롭다운 열기 → 세션 목록 로드
  const openSessionsList = async () => {
    setShowSessions((s) => !s);
    if (!showSessions) {
      setSessionsLoading(true);
      const list = await fetchChatSessions();
      setSessions(list);
      setSessionsLoading(false);
    }
  };

  // 세션 클릭 → 메시지 로드
  const loadSession = async (sid) => {
    setShowSessions(false);
    if (sending) return;
    const d = await fetchChatSession(sid);
    if (!d) return;
    const msgs = (d.messages || []).map((m) => {
      const t = m.createdAt ? new Date(m.createdAt) : new Date();
      return {
        role: m.role,
        text: m.text,
        time: `${String(t.getHours()).padStart(2,"0")}:${String(t.getMinutes()).padStart(2,"0")}`,
      };
    });
    if (msgs.length === 0) return;
    setMessages(msgs);
    setSessionId(sid);
    try { localStorage.setItem(CHAT_SESSION_KEY, String(sid)); } catch {}
    setLlmActive(true);   // 영구 저장된 세션은 LLM 기록
  };

  // 세션 삭제
  const removeSession = async (sid, e) => {
    if (e) { e.stopPropagation(); e.preventDefault(); }
    if (!confirm(`세션 #${sid} 를 삭제하시겠습니까? (메시지 모두 삭제)`)) return;
    const ok = await deleteChatSession(sid);
    if (ok) {
      setSessions((s) => s.filter((x) => x.id !== sid));
      // 현재 세션이 삭제됐으면 새 세션 시작
      if (sid === sessionId) {
        setMessages([greeting]);
        setSessionId(null);
        try { localStorage.removeItem(CHAT_SESSION_KEY); } catch {}
      }
    }
  };

  // 메시지 변할 때마다 저장 + 자동 스크롤
  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
    saveChatHistory(messages);
  }, [messages, sending]);

  const send = async (e) => {
    e && e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || sending) return;
    const now = new Date();
    const time = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
    const newUserMsg = { role: "user", text: trimmed, time };
    const r = new Date();
    const rtime = `${String(r.getHours()).padStart(2, "0")}:${String(r.getMinutes()).padStart(2, "0")}`;
    // 사용자 메시지 + 빈 AI 메시지(스트리밍 채워질 자리) 동시 추가
    setMessages((m) => [...m, newUserMsg, { role: "ai", text: "", time: rtime, streaming: true }]);
    setInput("");
    setSending(true);

    const ctx = buildChatContext(equipment, weather);
    const historyForLLM = [...messages, newUserMsg].slice(-12);

    // LLM 스트리밍 시도
    let finalReply = "";
    let usedLLM = false;
    let donePayload = null;
    const t0 = Date.now();

    const stream = await callLLMStream(trimmed, ctx, historyForLLM, sessionId, demoMode, {
      onSession: (info) => {
        // 서버가 새 세션 발급 또는 기존 세션 확인 — localStorage 에 저장
        if (info?.sessionId && info.sessionId !== sessionId) {
          setSessionId(info.sessionId);
          try { localStorage.setItem(CHAT_SESSION_KEY, String(info.sessionId)); } catch {}
        }
      },
      onDelta: (_piece, acc) => {
        // AI 메시지(마지막)의 text 누적 갱신
        setMessages((m) => {
          const arr = m.slice();
          const last = arr[arr.length - 1];
          if (last && last.role === "ai" && last.streaming) {
            arr[arr.length - 1] = { ...last, text: acc };
          }
          return arr;
        });
      },
      onTool: (info) => {
        // 서버가 DB 도구 호출 — "🔧 list_devices 조회 중..." UI 표시
        setMessages((m) => {
          const arr = m.slice();
          const last = arr[arr.length - 1];
          if (last && last.role === "ai" && last.streaming) {
            const toolCalls = [...(last.toolCalls || []), info];
            arr[arr.length - 1] = { ...last, toolCalls };
          }
          return arr;
        });
      },
      onDone: (payload) => {
        finalReply = (payload && payload.reply) || "";
        donePayload = payload || null;
      },
    });

    if (stream.ok) {
      finalReply = stream.reply || finalReply;
      usedLLM = true;
    } else {
      // 스트리밍 실패 → mock fallback
      finalReply = mockAIResponse(trimmed, { equipment });
    }

    // 마지막 메시지를 최종 결과로 확정 (streaming 플래그 해제, toolCalls 보존, meta 부착)
    const elapsedMs = Date.now() - t0;
    setMessages((m) => {
      const arr = m.slice();
      const last = arr[arr.length - 1];
      if (last && last.role === "ai") {
        arr[arr.length - 1] = {
          role: "ai",
          text: finalReply,
          time: rtime,
          toolCalls: last.toolCalls || [],
          meta: stream.ok ? {
            elapsedMs,
            rounds: donePayload?.rounds,
            tokens: donePayload?.tokens,
            model:  donePayload?.model,
          } : { elapsedMs, fallback: "mock" },
        };
      }
      return arr;
    });
    setLlmActive(usedLLM);
    setSending(false);

    // 응답에서 노드 ID 추출 → 지도 자동 zoom
    const matches = (finalReply || "").match(/TB24-5JN\d+/g) || [];
    const nodes = [...new Set(matches)];
    if (nodes.length > 0 && onBotReply) onBotReply(nodes);

    // 응답에서 단일 status 추출 → 자동 KPI 필터 (30초)
    if (onAutoKpi) {
      const kpi = detectKpiFromReply(finalReply);
      onAutoKpi(kpi);
    }
  };

  return (
    <Panel style={{ height: "100%", display: "flex", flexDirection: "column", minHeight: 0 }}>
      <PanelHeader
        right={(() => {
          // 모드: null = 아직 호출 X, true = LLM 연결됨, false = mock fallback
          const isLlm  = llmActive === true;
          const isMock = llmActive === false;
          const bg  = isLlm  ? "rgba(16,185,129,0.10)" : isMock ? "rgba(245,158,11,0.10)" : "rgba(79,70,229,0.10)";
          const fg  = isLlm  ? "#047857"               : isMock ? "#b45309"               : "var(--brand)";
          const dot = isLlm  ? "#10b981"                : isMock ? "#f59e0b"               : "var(--brand)";
          const lbl = isLlm  ? "LLM 연결됨"             : isMock ? "mock fallback"         : "대기";
          return (
            <div style={{ display: "flex", alignItems: "center", gap: 6, position: "relative" }}>
              {/* 세션 목록 드롭다운 토글 */}
              <button
                onClick={openSessionsList}
                title="이전 대화 세션 목록"
                style={{
                  display: "grid", placeItems: "center",
                  width: 22, height: 22, borderRadius: 6,
                  background: showSessions ? "var(--bg-elev)" : "transparent",
                  border: "1px solid var(--line)",
                  color: "var(--ink-3)", cursor: "pointer",
                }}
              >
                <Icons.list size={11} />
              </button>
              {/* 새 대화 (초기화) */}
              <button
                onClick={() => {
                  if (sending) return;
                  setMessages([greeting]);
                  setLlmActive(null);
                  setSessionId(null);
                  setShowSessions(false);
                  try {
                    localStorage.removeItem(CHAT_STORAGE_KEY);
                    localStorage.removeItem(CHAT_SESSION_KEY);
                  } catch {}
                }}
                title="대화 초기화 (새 세션 시작)"
                style={{
                  display: "grid", placeItems: "center",
                  width: 22, height: 22, borderRadius: 6,
                  background: "transparent", border: "1px solid var(--line)",
                  color: "var(--ink-3)", cursor: sending ? "not-allowed" : "pointer",
                  opacity: sending ? 0.4 : 1,
                }}
              >
                <Icons.refresh size={11} />
              </button>
              <div style={{
                display: "flex", alignItems: "center", gap: 6,
                padding: "2px 10px", borderRadius: 999,
                background: bg, color: fg,
                fontSize: 10, fontWeight: 700,
              }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: dot, animation: "pulse-dot 1.2s infinite" }} />
                {lbl}
              </div>
              {/* DB 연결 상태 — null=확인 중 / true=OK / false=끊김 */}
              {(() => {
                const isDbOk   = dbActive === true;
                const isDbBad  = dbActive === false;
                const dbBg  = isDbOk ? "rgba(16,185,129,0.10)" : isDbBad ? "rgba(239,68,68,0.10)" : "rgba(79,70,229,0.10)";
                const dbFg  = isDbOk ? "#047857"               : isDbBad ? "#b91c1c"               : "var(--brand)";
                const dbDot = isDbOk ? "#10b981"                : isDbBad ? "#dc2626"               : "var(--brand)";
                const dbLbl = isDbOk ? "DB 연결됨"             : isDbBad ? "DB 끊김"               : "DB 확인 중";
                const tip   = dbInfo && isDbOk
                  ? `siwon MySQL · 시계열 ${dbInfo.rows?.toLocaleString() || "?"} row${dbInfo.ollama ? " · Ollama OK" : ""}`
                  : isDbBad ? "/api/health 실패 — 서버·DB 점검 필요" : "/api/health 확인 중";
                return (
                  <div title={tip} style={{
                    display: "flex", alignItems: "center", gap: 6,
                    padding: "2px 10px", borderRadius: 999,
                    background: dbBg, color: dbFg,
                    fontSize: 10, fontWeight: 700,
                  }}>
                    <span style={{ width: 6, height: 6, borderRadius: "50%", background: dbDot, animation: "pulse-dot 1.2s infinite" }} />
                    {dbLbl}
                  </div>
                );
              })()}

              {/* 세션 목록 드롭다운 */}
              {showSessions && (
                <div style={{
                  position: "absolute", top: 30, right: 0, zIndex: 50,
                  width: 280, maxHeight: 360, overflow: "auto",
                  background: "var(--bg)",
                  border: "1px solid var(--line)", borderRadius: 8,
                  boxShadow: "0 8px 24px -6px rgba(0,0,0,0.18)",
                }} className="scroll">
                  <div style={{
                    padding: "8px 10px", borderBottom: "1px solid var(--line)",
                    fontSize: 11, fontWeight: 700, color: "var(--ink-3)",
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                  }}>
                    <span>저장된 대화 세션</span>
                    <button
                      onClick={() => setShowSessions(false)}
                      style={{ background: "transparent", border: "none", color: "var(--ink-4)", cursor: "pointer", padding: 0 }}
                      title="닫기"
                    ><Icons.close size={11} /></button>
                  </div>
                  {sessionsLoading && (
                    <div style={{ padding: 12, fontSize: 11, color: "var(--ink-4)" }}>불러오는 중...</div>
                  )}
                  {!sessionsLoading && sessions.length === 0 && (
                    <div style={{ padding: 12, fontSize: 11, color: "var(--ink-4)" }}>저장된 세션 없음</div>
                  )}
                  {!sessionsLoading && sessions.map((s) => {
                    const isActive = s.id === sessionId;
                    const dt = s.updated_at ? new Date(s.updated_at) : null;
                    const dtLabel = dt ? `${dt.getMonth()+1}/${dt.getDate()} ${String(dt.getHours()).padStart(2,"0")}:${String(dt.getMinutes()).padStart(2,"0")}` : "";
                    return (
                      <div
                        key={s.id}
                        onClick={() => loadSession(s.id)}
                        style={{
                          padding: "8px 10px",
                          borderBottom: "1px solid var(--line)",
                          background: isActive ? "rgba(79,70,229,0.08)" : "transparent",
                          cursor: "pointer",
                          display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8,
                        }}
                        onMouseOver={(e) => { if (!isActive) e.currentTarget.style.background = "var(--bg-elev)"; }}
                        onMouseOut={(e)  => { if (!isActive) e.currentTarget.style.background = "transparent"; }}
                      >
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{
                            fontSize: 12, fontWeight: isActive ? 700 : 500,
                            color: isActive ? "var(--brand)" : "var(--ink)",
                            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                          }}>
                            {s.title || `세션 #${s.id}`}
                          </div>
                          <div style={{ fontSize: 9, color: "var(--ink-4)", marginTop: 2 }}>
                            {dtLabel} · 메시지 {s.messageCount || 0}
                          </div>
                        </div>
                        <button
                          onClick={(e) => removeSession(s.id, e)}
                          title="세션 삭제"
                          style={{
                            background: "transparent", border: "none",
                            color: "var(--ink-4)", cursor: "pointer", padding: 2,
                            opacity: 0.5,
                          }}
                          onMouseOver={(e) => { e.currentTarget.style.opacity = 1; e.currentTarget.style.color = "#dc2626"; }}
                          onMouseOut={(e)  => { e.currentTarget.style.opacity = 0.5; e.currentTarget.style.color = "var(--ink-4)"; }}
                        ><Icons.close size={10} /></button>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })()}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Icons.sparkle size={16} color="var(--brand)" />
          <div style={{ fontSize: 13, fontWeight: 700 }}>AI 챗봇</div>
        </div>
      </PanelHeader>

      <div ref={listRef} className="scroll" style={{
        flex: 1, overflow: "auto",
        padding: "12px 12px 6px",
        background: "var(--bg-sunk)",
        display: "flex", flexDirection: "column", gap: 8,
      }}>
        {messages.map((m, i) => <ChatMessage key={i} message={m} />)}
        {/* 스트리밍 중엔 마지막 AI 메시지의 깜빡 커서가 visual feedback 역할 — 별도 typing indicator 불필요 */}
        {sending && messages[messages.length - 1]?.role !== "ai" && <ChatTyping />}
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

// 간단 inline 마크다운 파서: **굵게**, `코드`, [텍스트](URL) 정도만
//  - 외부 의존성 없이 React 노드 배열 반환
//  - LLM 이 자주 쓰는 굵게 강조 (** **) 만 잘 처리하면 충분
function renderInlineMD(text) {
  if (!text) return null;
  const tokens = [];
  // 패턴: **굵게**  |  `코드`
  const re = /(\*\*([^*]+)\*\*|`([^`]+)`)/g;
  let last = 0, m, key = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) tokens.push(text.slice(last, m.index));
    if (m[2] != null) {
      tokens.push(<strong key={key++} style={{ fontWeight: 800 }}>{m[2]}</strong>);
    } else if (m[3] != null) {
      tokens.push(
        <code key={key++} style={{
          fontFamily: "JetBrains Mono, ui-monospace, monospace",
          fontSize: "0.92em",
          padding: "1px 5px",
          borderRadius: 4,
          background: "rgba(0,0,0,0.06)",
        }}>{m[3]}</code>
      );
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) tokens.push(text.slice(last));
  return tokens;
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
          {/* 도구 호출 칩 (function calling) — 스트리밍 중이거나 호출이력 있을 때 표시 */}
          {isAi && Array.isArray(message.toolCalls) && message.toolCalls.length > 0 && (
            <div style={{
              display: "flex", flexWrap: "wrap", gap: 4,
              marginBottom: message.text ? 6 : 0,
            }}>
              {message.toolCalls.map((tc, idx) => (
                <span key={idx} style={{
                  display: "inline-flex", alignItems: "center", gap: 4,
                  padding: "2px 7px",
                  fontSize: 10.5, lineHeight: 1.3,
                  borderRadius: 999,
                  background: message.streaming ? "rgba(79,70,229,0.12)" : "rgba(0,0,0,0.04)",
                  color: message.streaming ? "var(--brand)" : "var(--ink-4)",
                  border: `1px solid ${message.streaming ? "rgba(79,70,229,0.25)" : "var(--line)"}`,
                  whiteSpace: "nowrap",
                  animation: message.streaming && !message.text ? "pulse-dot 1.4s ease-in-out infinite" : "none",
                }}>
                  <span style={{ fontSize: 9 }}>🔧</span>
                  <span style={{ fontFamily: "JetBrains Mono, ui-monospace, monospace" }}>{tc.name}</span>
                  {tc.args && Object.keys(tc.args).length > 0 && (
                    <span style={{ opacity: 0.7, fontSize: 9.5 }}>
                      ({Object.entries(tc.args).map(([k,v]) => `${k}:${String(v).slice(0,16)}`).join(", ")})
                    </span>
                  )}
                </span>
              ))}
            </div>
          )}
          {renderInlineMD(message.text)}
          {message.streaming && (
            <span style={{
              display: "inline-block",
              width: 7, height: 13, marginLeft: 2,
              verticalAlign: "text-bottom",
              background: "var(--brand)",
              animation: "blink 0.9s step-start infinite",
            }} />
          )}
        </div>
        <div style={{
          fontSize: 9, color: "var(--ink-4)",
          textAlign: isAi ? "left" : "right",
          paddingLeft: isAi ? 4 : 0,
          paddingRight: isAi ? 0 : 4,
          display: "flex",
          justifyContent: isAi ? "flex-start" : "flex-end",
          gap: 8,
        }}>
          <span>{message.time}</span>
          {/* 응답 메타 (AI 메시지만, 스트리밍 끝난 후) */}
          {isAi && message.meta && !message.streaming && (
            <span style={{ opacity: 0.7 }}>
              {message.meta.elapsedMs != null && `${(message.meta.elapsedMs / 1000).toFixed(1)}s`}
              {message.meta.rounds && ` · ${message.meta.rounds}R`}
              {message.meta.tokens?.completion != null && ` · ${message.meta.tokens.completion}tok`}
              {message.meta.fallback && ` · ${message.meta.fallback}`}
            </span>
          )}
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
    { label: "판정",         value: isAnomaly ? "위험" : "이상 의심", accent: color },
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
                  { c: "rgba(245,158,11,0.35)", l: `이상 의심 ≥${thW.toFixed(3)}` },
                  { c: "rgba(239,68,68,0.35)",  l: `위험 ≥${thA.toFixed(3)}` },
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
                <text x="4" y={yA + 4} fontSize="9" fill="var(--err)"   fontFamily="JetBrains Mono" fontWeight="700">{thA.toFixed(3)} ── 위험</text>
                <text x="4" y={yW + 4} fontSize="9" fill="var(--warn)"  fontFamily="JetBrains Mono" fontWeight="700">{thW.toFixed(3)} ── 이상 의심</text>
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
                      <span className="mono" style={{ fontSize: 12, fontWeight: 700, color: i === 0 ? "var(--err)" : "var(--ink-2)" }}>{c.pct}%</span>
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
                <div className="mono" style={{ fontSize: 20, fontWeight: 700, marginTop: 2, color: s.a || "var(--ink)" }}>{s.v}</div>
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

export function Dashboard({ onAnalyze, mapStyle, setMapStyle, theme, autoPlay = true, equipment = [], markers = [], anomalies = [], watch = [], aiEvents = [], demoMode = false }) {
  const [activeKpi, setActiveKpi] = useState(null);
  const [drawer, setDrawer] = useState(null);
  const [focused, setFocused] = useState(null); // {lat, lng, node, ts}
  const [fitTrigger, setFitTrigger] = useState(0); // 카운터: 변할 때마다 지도 fit
  const [boundsRequest, setBoundsRequest] = useState(null); // { coords, ts } — 챗봇이 여러 노드 언급 시
  const [showNormal, setShowNormal] = useState(true);        // 지도 위 정상 핀 토글
  const weather = useWeather();                              // 챗봇 컨텍스트용 날씨
  const [autoKpiSec, setAutoKpiSec] = useState(0);           // AI 자동 필터 잔여 초 (0 = 비활성)
  const autoKpiTimer = useRef(null);                         // setTimeout 핸들
  const autoKpiTick  = useRef(null);                         // setInterval 핸들 (1초)
  const [logOpen, setLogOpen]   = useState(false);           // 시스템 로그 드로어 토글 (옴니 5/22 피드백 반영 — 챗봇 영역 확장)

  // ESC 로 로그 드로어 닫기
  useEffect(() => {
    if (!logOpen) return;
    const onKey = (e) => { if (e.key === "Escape") setLogOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [logOpen]);

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
    let base = markers;
    if (activeKpi && activeKpi !== "all") {
      base = devicesToMarkers(equipment.filter((e) => e.status === activeKpi));
    }
    // 정상 핀 토글 (활성 KPI 가 'normal' 일 때는 토글 무시 — 명시적 선택)
    if (!showNormal && activeKpi !== "normal") {
      base = base.filter((m) => m.status !== "normal");
    }
    return base;
  }, [activeKpi, equipment, markers, showNormal]);

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

  // 표 row 클릭: 지도 포커스만 (드로어 열지 않음 — 대시보드 표는 요약용)
  const handleRowClick = (eq) => {
    if (eq && eq.lat != null && eq.lng != null) {
      setFocused({ lat: eq.lat, lng: eq.lng, node: eq.deviceId, ts: Date.now() });
    }
  };

  // AI 탐지 카드 클릭: 분석 모달 + 지도 포커스
  const handleAnalyze = (item) => {
    onAnalyze && onAnalyze(item);
    if (item && item.node) focusByNode(item.node);
  };

  // 사용자가 KPI 직접 클릭: 활성 토글 + 지도 fit + AI 자동 타이머 취소
  const handleKpiClick = (newActive) => {
    cancelAutoKpi();
    setActiveKpi(newActive);
    setFitTrigger(Date.now());
  };

  // AI 자동 필터 취소 (타이머 + 카운트다운 정리)
  const cancelAutoKpi = () => {
    if (autoKpiTimer.current) { clearTimeout(autoKpiTimer.current); autoKpiTimer.current = null; }
    if (autoKpiTick.current)  { clearInterval(autoKpiTick.current);  autoKpiTick.current  = null; }
    setAutoKpiSec(0);
  };

  // 챗봇 응답에서 status 추출 → 자동 KPI 활성 (30초 후 자동 해제)
  const handleAutoKpi = (kpi) => {
    cancelAutoKpi();
    setActiveKpi(kpi);
    setFitTrigger(Date.now());
    if (!kpi) return; // 총 장비(전체) 복귀는 타이머 X
    const TOTAL = 30;
    setAutoKpiSec(TOTAL);
    // 1초마다 카운트다운
    autoKpiTick.current = setInterval(() => {
      setAutoKpiSec((s) => Math.max(0, s - 1));
    }, 1000);
    // 30초 후 전체 복귀
    autoKpiTimer.current = setTimeout(() => {
      setActiveKpi(null);
      setFitTrigger(Date.now());
      cancelAutoKpi();
    }, TOTAL * 1000);
  };

  // 언마운트 시 타이머 정리
  useEffect(() => () => cancelAutoKpi(), []);

  // 챗봇 응답에서 노드 ID 들이 언급되면 지도 자동 zoom
  //  - 1개: flyTo + 팝업
  //  - 2개+: fitBounds (모두 한 화면에)
  const fitToNodes = (nodes) => {
    if (!nodes || nodes.length === 0) return;
    const coords = nodes
      .map((n) => equipment.find((e) => e.deviceId === n))
      .filter((eq) => eq && eq.lat != null && eq.lng != null)
      .map((eq) => [eq.lat, eq.lng, eq.deviceId]);
    if (coords.length === 0) return;
    if (coords.length === 1) {
      setFocused({ lat: coords[0][0], lng: coords[0][1], node: coords[0][2], ts: Date.now() });
    } else {
      setBoundsRequest({ coords: coords.map(([lat, lng]) => [lat, lng]), ts: Date.now() });
    }
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
      {/*
        2026-05-22 옴니솔루션 피드백 반영 — 챗봇 영역 확장 (사이즈 작다는 의견):
          row 1 (112px 고정)       — 좌: KPI                / 우: AI 챗봇 (span 1-3, 거대)
          row 2 (1.2fr, min 300)   — 좌: 지도               /     AI 챗봇 이어짐
          row 3 (1fr, min 240)     — 좌: 표 + AI 탐지       /     AI 챗봇 이어짐
        실시간 시스템 로그 → 우상단 floating 버튼 토글로 분리 (드로어).
        AI 탐지 → 옛 LogPanel 자리(좌측 row 3 우측 영역) 로 이동.
        AI 챗봇 → 우측 전체 column 차지 (옛 AIPanels + 옛 ChatPanel 자리 합쳐서).
      */}
      <div style={{
        position: "absolute", left: 0, right: 0, top: 0, bottom: 0,
        padding: 24,
        display: "grid",
        gridTemplateColumns: "1fr 460px",
        gridTemplateRows: "112px minmax(300px, 1.2fr) minmax(240px, 1fr)",
        gap: 16,
        minHeight: 0,
        overflow: "auto",
      }}>
        {/* (col 1, row 1) — KPI */}
        <div style={{ gridColumn: 1, gridRow: 1, minHeight: 0 }}>
          <KPIRow active={activeKpi} setActive={handleKpiClick} counts={counts} />
        </div>

        {/* (col 2, row 1~3) — AI 챗봇 (전체 우측 column · 옴니 5/22 확장) */}
        <div style={{ gridColumn: 2, gridRow: "1 / span 3", minHeight: 0 }}>
          <ChatPanel equipment={equipment} weather={weather} onBotReply={fitToNodes} onAutoKpi={handleAutoKpi} demoMode={demoMode} />
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
            boundsRequest={boundsRequest}
            showNormal={showNormal}
            setShowNormal={setShowNormal}
            autoKpiSec={autoKpiSec}
            onCancelAutoKpi={cancelAutoKpi}
          />
        </div>

        {/* (col 1, row 3) — 표 + AI 탐지 (옛 LogPanel 자리 → AIPanels 이동) */}
        <div style={{
          gridColumn: 1, gridRow: 3,
          display: "grid",
          gridTemplateColumns: "minmax(440px, 1fr) minmax(280px, 0.6fr)",
          gap: 16, minHeight: 0,
        }}>
          <TableSummary data={tableData} onRowClick={handleRowClick} activeKpi={activeKpi} />
          <AIPanels anomalies={anomalies} watch={watch} onAnalyze={handleAnalyze} />
        </div>
      </div>

      {/* 우상단 floating 토글 — 시스템 로그 드로어 */}
      <button
        onClick={() => setLogOpen((v) => !v)}
        title={logOpen ? "시스템 로그 닫기 (ESC)" : "실시간 시스템 로그 열기"}
        style={{
          position: "fixed", top: 168, right: logOpen ? 432 : 24, zIndex: 60,
          display: "inline-flex", alignItems: "center", gap: 6,
          padding: "8px 14px", borderRadius: 999,
          background: logOpen ? "var(--brand)" : "var(--bg-elev)",
          color: logOpen ? "#fff" : "var(--ink)",
          border: `1px solid ${logOpen ? "var(--brand)" : "var(--line)"}`,
          fontSize: 12, fontWeight: 700, letterSpacing: "0.02em",
          cursor: "pointer", boxShadow: "var(--shadow-card)",
          transition: "right 220ms cubic-bezier(.4,0,.2,1), background 160ms, color 160ms",
        }}
      >
        <span style={{
          width: 7, height: 7, borderRadius: "50%",
          background: logOpen ? "#fff" : "var(--ok)",
          animation: "pulse-dot 1.2s infinite",
        }} />
        {logOpen ? "로그 닫기" : "시스템 로그"}
        <span className="mono" style={{ opacity: 0.7, fontSize: 10, marginLeft: 4 }}>{lines.length}</span>
      </button>

      {/* 우측 슬라이드 드로어 — 시스템 로그 */}
      <div style={{
        position: "fixed", top: 0, right: 0, bottom: 0, width: 400, zIndex: 55,
        background: "var(--bg)",
        borderLeft: "1px solid var(--line)",
        boxShadow: logOpen ? "-10px 0 30px -10px rgba(0,0,0,0.4)" : "none",
        transform: logOpen ? "translateX(0)" : "translateX(100%)",
        transition: "transform 260ms cubic-bezier(.4,0,.2,1)",
        display: "flex", flexDirection: "column",
        paddingTop: 144,    // header + subnav + emergency banner 만큼 비움
      }}>
        <div style={{ padding: 16, flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
          <LogPanel lines={lines} />
        </div>
      </div>

      <DashboardEquipmentDrawer item={drawer} onClose={() => setDrawer(null)} />
    </>
  );
}
