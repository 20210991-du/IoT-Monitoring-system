/**
 * userPrefs.js — 사용자 환경설정 (개인용)
 *
 *  - 테마는 App.jsx 의 tweakState 에서 관리 (기존 시스템) → 여기선 비프음만
 *  - 비프음: 이상/관찰 알림 발생 시 시스템 비프 사운드
 *  - 추후 글자크기, 색약 모드 등 추가 가능
 *
 *  키: localStorage["siwon.prefs.<항목>"]
 */

const KEY_BEEP   = "siwon.prefs.beep";
const KEY_VOLUME = "siwon.prefs.beepVol";

// ── 비프음 ─────────────────────────────────────────────
export function getBeep() {
  return localStorage.getItem(KEY_BEEP) === "on";
}
export function setBeep(on) {
  localStorage.setItem(KEY_BEEP, on ? "on" : "off");
}

export function getBeepVolume() {
  const v = parseFloat(localStorage.getItem(KEY_VOLUME));
  return Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : 0.5;
}
export function setBeepVolume(v) {
  localStorage.setItem(KEY_VOLUME, String(Math.max(0, Math.min(1, v))));
}

/**
 * 비프 한 번 재생 (Web Audio API).
 *  kind: "alert" | "warn" | "ok"
 *  비프 OFF 면 no-op.
 */
let _ctx = null;
export function playBeep(kind = "alert") {
  if (!getBeep()) return;
  try {
    if (!_ctx) _ctx = new (window.AudioContext || window.webkitAudioContext)();
    const ctx = _ctx;
    if (ctx.state === "suspended") ctx.resume();

    const freqMap = { alert: 880, warn: 660, ok: 440 };
    const dur = kind === "alert" ? 0.18 : 0.12;
    const freq = freqMap[kind] || 660;
    const vol = getBeepVolume();

    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.type = "sine";
    o.frequency.value = freq;
    g.gain.setValueAtTime(0, ctx.currentTime);
    g.gain.linearRampToValueAtTime(vol, ctx.currentTime + 0.01);
    g.gain.linearRampToValueAtTime(0, ctx.currentTime + dur);
    o.connect(g).connect(ctx.destination);
    o.start();
    o.stop(ctx.currentTime + dur);

    // alert 는 두 번 (강조)
    if (kind === "alert") {
      const o2 = ctx.createOscillator();
      const g2 = ctx.createGain();
      o2.type = "sine";
      o2.frequency.value = freq;
      const t0 = ctx.currentTime + 0.22;
      g2.gain.setValueAtTime(0, t0);
      g2.gain.linearRampToValueAtTime(vol, t0 + 0.01);
      g2.gain.linearRampToValueAtTime(0, t0 + dur);
      o2.connect(g2).connect(ctx.destination);
      o2.start(t0);
      o2.stop(t0 + dur);
    }
  } catch {
    /* 무시 */
  }
}
