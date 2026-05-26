/**
 * weather.js — Open-Meteo (무료·키 X·CORS OK) 로 군산 현재 날씨 fetch
 *
 *  - 1시간 TTL, localStorage 캐시
 *  - WMO weather code → 한국어 + 이모지 매핑
 *  - 실패 시 기본값(흐림 12°C) 반환
 *  - 5/26: 서울 → 군산 변경 (군산도시가스 = 우리 SITE_ID=2 대상 지역)
 */

import { useEffect, useState } from "react";

// 군산시청 좌표 (군산 시 중심부)
const GUNSAN_LAT = 35.9678;
const GUNSAN_LON = 126.7369;
const CACHE_KEY = "siwon.weather";
const TTL_MS    = 60 * 60 * 1000; // 1h

// WMO Weather interpretation code (https://open-meteo.com/en/docs)
const WMO = {
  0:  { ko: "맑음",      icon: "☀️" },
  1:  { ko: "대체로 맑음", icon: "🌤" },
  2:  { ko: "부분 흐림",   icon: "⛅" },
  3:  { ko: "흐림",       icon: "☁️" },
  45: { ko: "안개",       icon: "🌫" },
  48: { ko: "안개",       icon: "🌫" },
  51: { ko: "약한 이슬비",  icon: "🌦" },
  53: { ko: "이슬비",      icon: "🌦" },
  55: { ko: "강한 이슬비",  icon: "🌦" },
  61: { ko: "약한 비",     icon: "🌧" },
  63: { ko: "비",         icon: "🌧" },
  65: { ko: "강한 비",     icon: "🌧" },
  71: { ko: "약한 눈",     icon: "🌨" },
  73: { ko: "눈",         icon: "🌨" },
  75: { ko: "강한 눈",     icon: "🌨" },
  77: { ko: "눈알갱이",    icon: "🌨" },
  80: { ko: "소나기",      icon: "🌦" },
  81: { ko: "소나기",      icon: "🌦" },
  82: { ko: "강한 소나기",  icon: "🌦" },
  85: { ko: "눈 소나기",   icon: "🌨" },
  86: { ko: "눈 소나기",   icon: "🌨" },
  95: { ko: "뇌우",       icon: "⛈" },
  96: { ko: "뇌우(우박)",   icon: "⛈" },
  99: { ko: "강한 뇌우",    icon: "⛈" },
};

const FALLBACK = { temp: 12, ko: "흐림", icon: "⛅", stale: true };

function readCache() {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const d = JSON.parse(raw);
    if (!d || typeof d.ts !== "number" || !d.data) return null;
    if (Date.now() - d.ts > TTL_MS) return null;
    return d.data;
  } catch { return null; }
}

function writeCache(data) {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify({ ts: Date.now(), data }));
  } catch { /* ignore */ }
}

export function useWeather() {
  const [w, setW] = useState(() => readCache() || FALLBACK);

  useEffect(() => {
    let alive = true;
    const fetchWeather = async () => {
      try {
        const url = `https://api.open-meteo.com/v1/forecast?latitude=${GUNSAN_LAT}&longitude=${GUNSAN_LON}&current=temperature_2m,weather_code,precipitation,relative_humidity_2m&timezone=Asia%2FSeoul`;
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), 8000);
        const res = await fetch(url, { signal: ctrl.signal });
        clearTimeout(t);
        if (!res.ok) return;
        const j = await res.json();
        if (!j.current) return;
        const code = j.current.weather_code;
        const meta = WMO[code] || { ko: "-", icon: "🌍" };
        const data = {
          temp: Math.round(j.current.temperature_2m),
          ko:   meta.ko,
          icon: meta.icon,
          code,
          precip: j.current.precipitation != null ? Number(j.current.precipitation) : null,    // mm
          humidity: j.current.relative_humidity_2m != null ? Math.round(j.current.relative_humidity_2m) : null,
          time: j.current.time,
          stale: false,
        };
        if (alive) setW(data);
        writeCache(data);
      } catch { /* network/abort error → keep cached or fallback */ }
    };
    // 캐시가 stale 거나 없으면 즉시 fetch, 그리고 1h 마다
    if (!w || w.stale) fetchWeather();
    const id = setInterval(fetchWeather, TTL_MS);
    return () => { alive = false; clearInterval(id); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return w;
}
