import { useState, useEffect, useCallback } from "react";
import {
  Header,
  SubNav,
  EmergencyBanner,
} from "./components/chrome.jsx";
import { TweaksPanel } from "./components/TweaksPanel.jsx";
import { Icons } from "./components/Icons.jsx";
import { Login } from "./pages/Login.jsx";
import { SignUp } from "./pages/SignUp.jsx";
import { Dashboard, AnalysisModal } from "./pages/Dashboard.jsx";
import { Equipment } from "./pages/Equipment.jsx";
import { Admin } from "./pages/Admin.jsx";
import { currentSession, signOut, listPending } from "./lib/authMock.js";
import {
  EQUIPMENT  as MOCK_EQUIPMENT,
  MAP_MARKERS as MOCK_MARKERS,
  AI_ANOMALIES as MOCK_ANOMALIES,
  AI_WATCH    as MOCK_WATCH,
  AI_INSIGHTS as MOCK_INSIGHTS,
} from "./data/mockData.js";
import { fetchDevices, fetchAnomalies, fetchInsights, devicesToMarkers } from "./api/client.js";

const TWEAK_DEFAULTS = { theme: "light", mapStyle: "light", autoPlay: true };
const POLL_MS = 60_000;

// Drawer (used by the equipment tab)
function EquipmentDrawer({ item, onClose }) {
  if (!item) return null;
  const statusMap = {
    normal:  { ko: "정상", fg: "#047857", bg: "rgba(16,185,129,0.14)", bd: "rgba(16,185,129,0.3)" },
    anomaly: { ko: "이상", fg: "#b91c1c", bg: "rgba(239,68,68,0.12)",   bd: "rgba(239,68,68,0.3)" },
    warn:    { ko: "관찰", fg: "#b45309", bg: "rgba(245,158,11,0.14)",  bd: "rgba(245,158,11,0.3)" },
    offline: { ko: "장애", fg: "#475569", bg: "rgba(100,116,139,0.14)", bd: "rgba(100,116,139,0.3)" },
  };
  const c = statusMap[item.status] || statusMap.normal;
  return (
    <div style={{ position: "absolute", inset: 0, zIndex: 90, pointerEvents: "none" }}>
      <div
        onClick={onClose}
        style={{
          position: "absolute", inset: 0,
          background: "rgba(10,15,30,0.3)", backdropFilter: "blur(2px)",
          pointerEvents: "auto", animation: "slide-in-up 180ms ease",
        }}
      />
      <div style={{
        position: "absolute", right: 0, top: 0, bottom: 0, width: 520,
        background: "var(--bg-elev)", borderLeft: "1px solid var(--line)",
        boxShadow: "var(--shadow-lg)", pointerEvents: "auto",
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
          <button onClick={onClose} style={{ color: "var(--ink-3)" }}>
            <Icons.close size={18} />
          </button>
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
          <div style={{ fontSize: 12, fontWeight: 700, color: "var(--ink-3)", marginBottom: 10 }}>방식전위 트렌드 (6시간)</div>
          <div style={{
            padding: 12, borderRadius: 10,
            background: "var(--bg-sunk)", border: "1px solid var(--line-soft)",
            height: 120, marginBottom: 20,
          }}>
            <svg viewBox="0 0 440 96" style={{ width: "100%", height: "100%" }}>
              <polyline fill="none" stroke="var(--brand)" strokeWidth="2"
                        points="0,70 40,68 80,72 120,65 160,68 200,60 240,55 280,50 320,45 360,40 400,38 440,42" />
            </svg>
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

export function App() {
  const [screen, setScreen] = useState(() => {
    // 세션이 살아있으면 바로 app, 아니면 login
    const saved = localStorage.getItem("screen");
    if (saved === "signup") return saved;
    return currentSession() ? "app" : "login";
  });
  const [user, setUser]     = useState(() => currentSession());
  const [pendingId, setPendingId] = useState("");  // 가입 직후 로그인 화면에 prefill
  const [tab, setTab]       = useState(() => localStorage.getItem("tab") || "dashboard");
  const [tweakState, setTweakState] = useState(() => {
    const el = typeof document !== "undefined" && document.getElementById("tweaks-defaults");
    if (!el) return TWEAK_DEFAULTS;
    try {
      return {
        ...TWEAK_DEFAULTS,
        ...JSON.parse(el.textContent.replace(/\/\*EDITMODE-(BEGIN|END)\*\//g, "")),
      };
    } catch {
      return TWEAK_DEFAULTS;
    }
  });
  const [tweaksOn,   setTweaksOn]   = useState(false);
  const [bannerOpen, setBannerOpen] = useState(true);
  const [analysis,   setAnalysis]   = useState(null);
  const [drawer,     setDrawer]     = useState(null);

  // ── API 데이터 상태 (초기값 = 목업) ──────────────────────────
  const [equipment, setEquipment] = useState(MOCK_EQUIPMENT);
  const [markers,   setMarkers]   = useState(() => devicesToMarkers(MOCK_EQUIPMENT));
  const [anomalies, setAnomalies] = useState(MOCK_ANOMALIES);
  const [watch,     setWatch]     = useState(MOCK_WATCH);
  const [insights,  setInsights]  = useState(MOCK_INSIGHTS);
  const [apiStatus, setApiStatus] = useState("mock"); // "mock" | "loading" | "ok" | "error"
  const [aiEvents,  setAiEvents]  = useState([]);     // 실시간 로그로 전달되는 AI 이벤트

  // ── AI 이벤트 생성 헬퍼 ────────────────────────────────────
  function makeAiLogEvents(devRes, anoRes) {
    const now = new Date();
    const t   = now.toTimeString().slice(0, 8); // HH:MM:SS
    const base = now.getTime();
    const events = [];

    events.push({
      id: base, kind: "ai", time: t,
      text: `AI: LSTM 배치 추론 완료 · ${devRes.devices.length}개 장비 분석 (${devRes.last_updated || t})`,
    });

    anoRes.anomalies.slice(0, 5).forEach((a, i) => {
      events.push({
        id: base + 1 + i, kind: "alert", time: t,
        text: `ALERT: MSE ${a.mse.toFixed(4)} > TH ${a.threshold.toFixed(4)} @ ${a.node} [${a.label}]`,
      });
    });

    anoRes.watch.slice(0, 3).forEach((w, i) => {
      events.push({
        id: base + 10 + i, kind: "warn", time: t,
        text: `WARN: 관찰 대상 ${w.node} · MSE=${w.mse.toFixed(4)} [${w.label}]`,
      });
    });

    const offline = devRes.devices.filter((d) => d.status === "offline");
    if (offline.length > 0) {
      events.push({
        id: base + 20, kind: "warn", time: t,
        text: `WARN: 통신고장 ${offline.length}건 · ${offline.slice(0, 2).map((d) => d.deviceId).join(", ")}`,
      });
    }

    const normal = devRes.devices.filter((d) => d.status === "normal").length;
    events.push({
      id: base + 30, kind: "ok", time: t,
      text: `SYS: 정상 ${normal}건 · 이상 ${anoRes.anomalies.length}건 · 관찰 ${anoRes.watch.length}건`,
      tail: "OK",
    });

    return events;
  }

  // ── 백엔드에서 데이터 로드 ─────────────────────────────────
  const loadData = useCallback(async () => {
    setApiStatus((s) => s === "mock" ? "loading" : s);
    try {
      const [devRes, anoRes, insRes] = await Promise.all([
        fetchDevices(),
        fetchAnomalies(),
        fetchInsights(),
      ]);
      setEquipment(devRes.devices);
      setMarkers(devicesToMarkers(devRes.devices));
      setAnomalies(anoRes.anomalies);
      setWatch(anoRes.watch);
      setInsights(insRes.insights);
      setAiEvents(makeAiLogEvents(devRes, anoRes));
      setApiStatus("ok");
    } catch (err) {
      console.warn("[API] 백엔드 연결 실패 — 목업 데이터로 동작합니다.", err.message);
      setApiStatus((s) => s === "loading" ? "error" : s);
    }
  }, []);

  // 마운트 시 1회 + 60초 폴링
  useEffect(() => {
    loadData();
    const id = setInterval(loadData, POLL_MS);
    return () => clearInterval(id);
  }, [loadData]);

  useEffect(() => { localStorage.setItem("screen", screen); }, [screen]);
  useEffect(() => { localStorage.setItem("tab",    tab);    }, [tab]);

  // admin 권한 사라지면 users 탭에서 자동 빠져나감 (탭 보호)
  useEffect(() => {
    if (tab === "users" && user?.role !== "admin") setTab("dashboard");
  }, [tab, user]);

  // 테마 적용
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", tweakState.theme || "light");
  }, [tweakState.theme]);

  // Design-mode protocol (claude.ai/design iframe 연동)
  useEffect(() => {
    const onMsg = (e) => {
      if (!e.data || !e.data.type) return;
      if (e.data.type === "__activate_edit_mode")   setTweaksOn(true);
      if (e.data.type === "__deactivate_edit_mode") setTweaksOn(false);
    };
    window.addEventListener("message", onMsg);
    try { window.parent.postMessage({ type: "__edit_mode_available" }, "*"); } catch { }
    return () => window.removeEventListener("message", onMsg);
  }, []);

  const setMapStyle = (mapStyle) => {
    setTweakState((s) => ({ ...s, mapStyle }));
    try { window.parent.postMessage({ type: "__edit_mode_set_keys", edits: { mapStyle } }, "*"); } catch { }
  };

  if (screen === "login") {
    return (
      <Login
        prefillId={pendingId}
        onLogin={(u) => { setUser(u); setPendingId(""); setScreen("app"); }}
        onSignUp={() => setScreen("signup")}
      />
    );
  }
  if (screen === "signup") {
    return (
      <SignUp
        onSuccess={(newId) => { setPendingId(newId); setScreen("login"); }}
        onBackToLogin={() => setScreen("login")}
      />
    );
  }

  return (
    <>
      <Header
        onLogout={() => { signOut(); setUser(null); setScreen("login"); }}
        apiStatus={apiStatus}
        user={user}
        setUser={setUser}
        theme={tweakState.theme}
        setTheme={(t) => setTweakState((s) => ({ ...s, theme: t }))}
        mapStyle={tweakState.mapStyle}
        setMapStyle={setMapStyle}
      />
      <SubNav
        tab={tab}
        setTab={(k) => {
          // 비-admin 이 users 탭 진입 차단 (URL/state 보호)
          if (k === "users" && user?.role !== "admin") return;
          setTab(k);
        }}
        apiStatus={apiStatus}
        user={user}
        pendingCount={user?.role === "admin" ? (listPending(user).users?.length || 0) : 0}
      />
      {bannerOpen && tab === "dashboard" && (() => {
        const criticalDevices = (equipment || []).filter((e) => e.status === "critical");
        if (criticalDevices.length === 0) return null;
        return (
          <EmergencyBanner
            criticalDevices={criticalDevices}
            onDismiss={() => setBannerOpen(false)}
            onOpen={() => setAnalysis(anomalies.find((a) => a.node === criticalDevices[0].deviceId) || anomalies[0] || null)}
          />
        );
      })()}
      {(() => {
        const hasCritical = tab === "dashboard" && bannerOpen && (equipment || []).some((e) => e.status === "critical");
        return (
          <div style={{
            position: "absolute", left: 0, right: 0,
            top: hasCritical ? 144 : 104,
            bottom: 0,
            transition: "top 220ms",
          }}>
        {tab === "dashboard" && (
          <Dashboard
            onAnalyze={setAnalysis}
            mapStyle={tweakState.mapStyle}
            setMapStyle={setMapStyle}
            theme={tweakState.theme}
            autoPlay={tweakState.autoPlay}
            equipment={equipment}
            markers={markers}
            anomalies={anomalies}
            watch={watch}
            insights={insights}
            aiEvents={aiEvents}
          />
        )}
        {tab === "equipment" && <Equipment onOpen={setDrawer} equipment={equipment} />}
        {tab === "users" && (
          <Admin
            user={user}
            equipment={equipment}
            anomalies={anomalies}
            watch={watch}
            apiStatus={apiStatus}
          />
        )}
      </div>
        );
      })()}
      <AnalysisModal item={analysis} onClose={() => setAnalysis(null)} />
      <EquipmentDrawer item={drawer} onClose={() => setDrawer(null)} />
      <TweaksPanel state={tweakState} setState={setTweakState} show={tweaksOn} />
    </>
  );
}
