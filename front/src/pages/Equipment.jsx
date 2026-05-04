import { useState, useMemo } from "react";
import { Icons } from "../components/Icons.jsx";

const statusChip = (status) => {
  const map = {
    normal:   { ko: "정상",     fg: "#047857", bg: "rgba(16,185,129,0.14)", bd: "rgba(16,185,129,0.3)" },
    critical: { ko: "위험",     fg: "#fff",     bg: "#dc2626",                bd: "#991b1b" },
    anomaly:  { ko: "이상",     fg: "#b91c1c", bg: "rgba(239,68,68,0.12)",   bd: "rgba(239,68,68,0.3)" },
    warn:     { ko: "관찰",     fg: "#b45309", bg: "rgba(245,158,11,0.14)",  bd: "rgba(245,158,11,0.3)" },
    offline:  { ko: "통신장애", fg: "#475569", bg: "rgba(100,116,139,0.14)", bd: "rgba(100,116,139,0.3)" },
  };
  return map[status] || map.normal;
};

function FilterPill({ active, onClick, color, children, count }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: "flex", alignItems: "center", gap: 8,
        padding: "8px 16px", borderRadius: 999,
        background: active ? color : "var(--bg-elev)",
        border: `1px solid ${active ? color : "var(--line)"}`,
        color: active ? "#fff" : "var(--ink-2)",
        fontSize: 13, fontWeight: 600,
        transition: "all 160ms",
      }}
    >
      {children}
      {count !== undefined && (
        <span style={{
          fontSize: 11, fontWeight: 700,
          padding: "1px 8px", borderRadius: 999,
          background: active ? "rgba(255,255,255,0.25)" : "var(--bg-sunk)",
          color: active ? "#fff" : "var(--ink-3)",
        }}>
          {count}
        </span>
      )}
    </button>
  );
}

const pageBtn = (disabled) => ({
  minWidth: 32, height: 32, padding: "0 10px",
  borderRadius: 8, fontSize: 12, fontWeight: 500,
  color: disabled ? "var(--ink-4)" : "var(--ink-2)",
  background: "transparent",
  cursor: disabled ? "not-allowed" : "pointer",
});

export function Equipment({ onOpen, equipment = [] }) {
  const [filter, setFilter] = useState("all");
  const [zone, setZone] = useState("all");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState({ col: "deviceId", dir: "asc" });
  const [page, setPage] = useState(1);
  const pageSize = 14;

  const counts = useMemo(() => {
    const c = { all: equipment.length, normal: 0, critical: 0, anomaly: 0, warn: 0, offline: 0 };
    equipment.forEach((e) => { if (c[e.status] !== undefined) c[e.status]++; });
    return c;
  }, [equipment]);

  const zones = useMemo(() => {
    const s = new Set(equipment.map((e) => e.zone));
    return ["all", ...Array.from(s)];
  }, [equipment]);

  const filtered = useMemo(() => {
    let list = equipment.slice();
    if (filter !== "all") list = list.filter((e) => e.status === filter);
    if (zone !== "all") list = list.filter((e) => e.zone === zone);
    if (query) {
      const q = query.toLowerCase();
      list = list.filter(
        (e) =>
          e.deviceId.toLowerCase().includes(q) ||
          e.facilityId.toLowerCase().includes(q) ||
          e.location.includes(query)
      );
    }
    list.sort((a, b) => {
      const av = a[sort.col], bv = b[sort.col];
      if (av === bv) return 0;
      const r = av > bv ? 1 : -1;
      return sort.dir === "asc" ? r : -r;
    });
    return list;
  }, [filter, zone, query, sort, equipment]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const pageData = filtered.slice((page - 1) * pageSize, page * pageSize);

  const toggleSort = (col) => {
    setSort((s) => (s.col === col ? { col, dir: s.dir === "asc" ? "desc" : "asc" } : { col, dir: "asc" }));
  };

  const columns = [
    { k: "facilityId", label: "시설명", align: "left", w: 120 },
    { k: "deviceId",   label: "장비명", align: "left", w: 130 },
    { k: "location",   label: "설치위치", align: "left", w: 320 },
    { k: "zone",       label: "구역", align: "left", w: 110 },
    { k: "status",     label: "상태", align: "center", w: 100 },
    { k: "volt",        label: "방식전위(mV)", align: "right", w: 130 },
    { k: "ac",          label: "AC유입(mV)", align: "right", w: 120 },
    { k: "sacrificial", label: "희생전류(mA)", align: "right", w: 120 },
    { k: "temp",        label: "온도(°C)", align: "right", w: 90 },
    { k: "hum",         label: "습도(%)", align: "right", w: 90 },
    { k: "commDbm",     label: "통신품질(dBm)", align: "right", w: 120 },
    { k: "updatedAt",  label: "최근 갱신", align: "left", w: 140 },
  ];

  return (
    <div style={{
      position: "absolute", left: 0, right: 0, top: 0, bottom: 0,
      padding: 24, display: "flex", flexDirection: "column", gap: 16, overflow: "hidden",
    }}>
      {/* Header (제목만) */}
      <div>
        <div style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em" }}>전체 장비 현황</div>
        <div style={{ fontSize: 13, color: "var(--ink-3)", marginTop: 4 }}>
          전국 {equipment.length}개 감시 노드 · 1시간 주기 수집
        </div>
      </div>

      {/* Filter pills + 검색 */}
      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <FilterPill active={filter === "all"} onClick={() => { setFilter("all"); setPage(1); }} color="var(--brand)" count={counts.all}>전체</FilterPill>
        <FilterPill active={filter === "normal"} onClick={() => { setFilter("normal"); setPage(1); }} color="var(--ok)" count={counts.normal}>
          <Icons.check size={12} /> 정상
        </FilterPill>
        <FilterPill active={filter === "critical"} onClick={() => { setFilter("critical"); setPage(1); }} color="#dc2626" count={counts.critical}>
          <Icons.alert size={12} /> 위험
        </FilterPill>
        <FilterPill active={filter === "anomaly"} onClick={() => { setFilter("anomaly"); setPage(1); }} color="var(--err)" count={counts.anomaly}>
          <Icons.alert size={12} /> 이상
        </FilterPill>
        <FilterPill active={filter === "warn"} onClick={() => { setFilter("warn"); setPage(1); }} color="var(--warn)" count={counts.warn}>
          <Icons.eye size={12} /> 관찰
        </FilterPill>
        <FilterPill active={filter === "offline"} onClick={() => { setFilter("offline"); setPage(1); }} color="var(--ink-3)" count={counts.offline}>
          <Icons.wifi_off size={12} /> 통신장애
        </FilterPill>
        <div style={{ width: 1, height: 24, background: "var(--line)", margin: "0 4px" }} />
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {zones.map((z) => (
            <button
              key={z}
              onClick={() => { setZone(z); setPage(1); }}
              style={{
                padding: "6px 12px", borderRadius: 8,
                fontSize: 12, fontWeight: 600,
                background: zone === z ? "var(--brand-wash)" : "transparent",
                color: zone === z ? "var(--brand)" : "var(--ink-3)",
                border: `1px solid ${zone === z ? "rgba(79,70,229,0.25)" : "var(--line)"}`,
              }}
            >
              {z === "all" ? "전체 구역" : z}
            </button>
          ))}
        </div>
        <div style={{ flex: 1 }} />
        <div style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: "8px 14px", borderRadius: 12,
          background: "var(--bg-elev)", border: "1px solid var(--line)",
          width: 320,
        }}>
          <Icons.search size={14} color="var(--ink-4)" />
          <input
            placeholder="장비명, 시설명, 위치로 검색..."
            value={query}
            onChange={(e) => { setQuery(e.target.value); setPage(1); }}
            style={{
              flex: 1, border: "none", outline: "none", background: "transparent",
              fontSize: 13,
            }}
          />
          {query && (
            <button onClick={() => setQuery("")} style={{ color: "var(--ink-4)" }}>
              <Icons.close size={12} />
            </button>
          )}
        </div>
      </div>

      {/* Table */}
      <div style={{
        flex: 1, minHeight: 0,
        background: "var(--bg-elev)", borderRadius: 16,
        border: "1px solid var(--line)",
        boxShadow: "var(--shadow-card)",
        display: "flex", flexDirection: "column", overflow: "hidden",
      }}>
        <div className="scroll" style={{ flex: 1, overflow: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, tableLayout: "fixed" }}>
            <colgroup>
              {columns.map((c) => <col key={c.k} style={{ width: c.w }} />)}
            </colgroup>
            <thead>
              <tr>
                {columns.map((c) => (
                  <th
                    key={c.k}
                    onClick={() => toggleSort(c.k)}
                    style={{
                      position: "sticky", top: 0, zIndex: 2,
                      background: "var(--bg-elev)",
                      textAlign: c.align,
                      verticalAlign: "middle",
                      padding: "14px 14px",
                      fontWeight: 700, fontSize: 12, color: "var(--ink-3)",
                      borderBottom: "1px solid var(--line)",
                      cursor: "pointer", userSelect: "none",
                      whiteSpace: "nowrap",
                      letterSpacing: "-0.01em",
                      height: 46,
                    }}
                  >
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                      {c.label}
                      {sort.col === c.k && (
                        sort.dir === "asc"
                          ? <Icons.arrow_up size={10} color="var(--brand)" />
                          : <Icons.arrow_down size={10} color="var(--brand)" />
                      )}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {pageData.map((r, idx) => {
                const c = statusChip(r.status);
                const zebraBg = idx % 2 === 1 ? "var(--bg-sunk)" : "transparent";
                return (
                  <tr
                    key={r.id}
                    onClick={() => onOpen(r)}
                    style={{
                      borderBottom: "1px solid var(--line-soft)", cursor: "pointer",
                      transition: "background 120ms",
                      background: zebraBg,
                      height: 52,
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = "var(--brand-wash)")}
                    onMouseLeave={(e) => (e.currentTarget.style.background = zebraBg)}
                  >
                    <td className="mono" style={{ padding: "14px 14px", fontWeight: 700, color: "var(--ink)", verticalAlign: "middle" }}>{r.facilityId}</td>
                    <td className="mono" style={{ padding: "14px 14px", color: "var(--ink-2)", fontWeight: 500, verticalAlign: "middle" }}>{r.deviceId}</td>
                    <td style={{ padding: "14px 14px", color: "var(--ink-2)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", verticalAlign: "middle" }}>{r.location}</td>
                    <td style={{ padding: "14px 14px", color: "var(--ink-3)", fontSize: 12, verticalAlign: "middle" }}>{r.zone}</td>
                    <td style={{ padding: "14px 14px", textAlign: "center", verticalAlign: "middle" }}>
                      <span style={{
                        display: "inline-block", padding: "4px 12px", borderRadius: 999,
                        fontSize: 12, fontWeight: 700,
                        background: c.bg, color: c.fg, border: `1px solid ${c.bd}`,
                        minWidth: 48,
                      }}>
                        {c.ko}
                      </span>
                    </td>
                    <td className="mono" style={{ padding: "14px 14px", textAlign: "right", verticalAlign: "middle", color: r.status === "anomaly" ? "var(--err)" : "var(--ink)", fontWeight: r.status === "anomaly" ? 700 : 500 }}>{r.volt}</td>
                    <td className="mono" style={{ padding: "14px 14px", textAlign: "right", verticalAlign: "middle", color: r.status === "anomaly" && r.label === "AC 유입 과다" ? "var(--err)" : "var(--ink-2)" }}>{r.ac.toLocaleString()}</td>
                    <td className="mono" style={{ padding: "14px 14px", textAlign: "right", verticalAlign: "middle", color: r.sacrificial !== 0 && r.sacrificial < 1 ? "var(--warn)" : "var(--ink-2)" }}>{r.sacrificial}</td>
                    <td className="mono" style={{ padding: "14px 14px", textAlign: "right", verticalAlign: "middle", color: "var(--ink-2)" }}>{r.temp}</td>
                    <td className="mono" style={{ padding: "14px 14px", textAlign: "right", verticalAlign: "middle", color: "var(--ink-2)" }}>{r.hum}</td>
                    <td className="mono" style={{ padding: "14px 14px", textAlign: "right", verticalAlign: "middle", color: !r.commOk ? "var(--err)" : r.commDbm < -75 ? "var(--warn)" : "var(--ink-2)", fontWeight: !r.commOk ? 700 : 500 }}>{r.commOk ? r.commDbm : "—"}</td>
                    <td className="mono" style={{ padding: "14px 14px", color: "var(--ink-3)", whiteSpace: "nowrap", fontSize: 12, verticalAlign: "middle" }}>{r.updatedAt}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div style={{
          padding: "12px 20px",
          borderTop: "1px solid var(--line-soft)",
          display: "flex", justifyContent: "space-between", alignItems: "center",
          background: "var(--bg-sunk)",
        }}>
          <div style={{ fontSize: 12, color: "var(--ink-3)" }}>
            총 <span style={{ color: "var(--ink)", fontWeight: 700 }}>{filtered.length}</span>건 중{" "}
            <span style={{ fontWeight: 700, color: "var(--ink)" }}>
              {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, filtered.length)}
            </span>{" "}
            표시
          </div>
          <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <button onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page === 1} style={pageBtn(page === 1)}>이전</button>
            {Array.from({ length: Math.min(7, totalPages) }, (_, i) => {
              let p;
              if (totalPages <= 7) p = i + 1;
              else if (page <= 4) p = i + 1;
              else if (page >= totalPages - 3) p = totalPages - 6 + i;
              else p = page - 3 + i;
              return (
                <button
                  key={p}
                  onClick={() => setPage(p)}
                  style={{
                    ...pageBtn(false),
                    background: page === p ? "var(--brand)" : "transparent",
                    color: page === p ? "#fff" : "var(--ink-2)",
                    fontWeight: page === p ? 700 : 500,
                  }}
                >
                  {p}
                </button>
              );
            })}
            <button onClick={() => setPage((p) => Math.min(totalPages, p + 1))} disabled={page === totalPages} style={pageBtn(page === totalPages)}>다음</button>
          </div>
        </div>
      </div>
    </div>
  );
}
