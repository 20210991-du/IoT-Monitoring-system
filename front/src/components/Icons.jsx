// Icon set — hand-tuned 1.5px SVGs (Lucide-ish feel)

const ic = (paths, size = 20) => (props) => (
  <svg
    width={props.size || size}
    height={props.size || size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.8"
    strokeLinecap="round"
    strokeLinejoin="round"
    {...props}
  >
    {paths}
  </svg>
);

export const Icons = {
  radio: ic(<>
    <path d="M4.9 19.1A10 10 0 0 1 4.9 4.9" />
    <path d="M8.5 15.5a5 5 0 0 1 0-7" />
    <circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none" />
    <path d="M15.5 8.5a5 5 0 0 1 0 7" />
    <path d="M19.1 4.9a10 10 0 0 1 0 14.2" />
  </>),
  bell: ic(<>
    <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
    <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
  </>),
  settings: ic(<>
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3h0a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8v0a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z" />
  </>),
  alert: ic(<>
    <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
    <path d="M12 9v4" />
    <circle cx="12" cy="17" r="0.5" fill="currentColor" />
  </>),
  eye: ic(<>
    <path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z" />
    <circle cx="12" cy="12" r="3" />
  </>),
  wifi_off: ic(<>
    <path d="M2 2l20 20" />
    <path d="M8.5 16.5a5 5 0 0 1 7 0" />
    <path d="M2 8.8A17 17 0 0 1 7 6" />
    <path d="M22 8.8A17 17 0 0 0 16 5.8" />
    <circle cx="12" cy="20" r="1" fill="currentColor" />
  </>),
  check: ic(<><path d="M5 12.5 10 17l9-11" /></>),
  box: ic(<>
    <path d="M21 8v13a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8" />
    <path d="M1 3h22v5H1z" />
    <path d="M10 12h4" />
  </>),
  search: ic(<>
    <circle cx="11" cy="11" r="7" />
    <path d="m21 21-4.3-4.3" />
  </>),
  refresh: ic(<>
    <path d="M21 12a9 9 0 1 1-3-6.7L21 8" />
    <path d="M21 3v5h-5" />
  </>),
  user: ic(<>
    <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
    <circle cx="12" cy="7" r="4" />
  </>),
  lock: ic(<>
    <rect x="4" y="10.5" width="16" height="11" rx="2" />
    <path d="M8 10.5V7a4 4 0 1 1 8 0v3.5" />
  </>),
  sparkle: ic(<>
    <path d="M12 3v4M12 17v4M4.2 4.2l2.8 2.8M17 17l2.8 2.8M3 12h4M17 12h4M4.2 19.8 7 17M17 7l2.8-2.8" />
    <circle cx="12" cy="12" r="3" fill="currentColor" stroke="none" />
  </>),
  trend: ic(<>
    <path d="M3 17l6-6 4 4 8-9" />
    <path d="M14 6h7v7" />
  </>),
  maximize: ic(<>
    <path d="M4 9V5a1 1 0 0 1 1-1h4" />
    <path d="M20 9V5a1 1 0 0 0-1-1h-4" />
    <path d="M4 15v4a1 1 0 0 0 1 1h4" />
    <path d="M20 15v4a1 1 0 0 1-1 1h-4" />
  </>),
  close: ic(<><path d="M6 6l12 12M18 6 6 18" /></>),
  chevron_right: ic(<><path d="m9 6 6 6-6 6" /></>),
  arrow_up: ic(<><path d="M12 20V4M5 11l7-7 7 7" /></>),
  arrow_down: ic(<><path d="M12 4v16M19 13l-7 7-7-7" /></>),
  crosshair: ic(<>
    <circle cx="12" cy="12" r="8" />
    <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
  </>),
  plus: ic(<><path d="M12 5v14M5 12h14" /></>),
  minus: ic(<><path d="M5 12h14" /></>),
  pin: ic(<>
    <path d="M12 2a7 7 0 0 1 7 7c0 5-7 13-7 13S5 14 5 9a7 7 0 0 1 7-7z" />
    <circle cx="12" cy="9" r="2.5" />
  </>),
  layers: ic(<>
    <path d="M12 2 2 7l10 5 10-5-10-5z" />
    <path d="M2 17l10 5 10-5" />
    <path d="M2 12l10 5 10-5" />
  </>),
  activity: ic(<><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></>),
  cpu: ic(<>
    <rect x="4" y="4" width="16" height="16" rx="2" />
    <rect x="9" y="9" width="6" height="6" />
    <path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3" />
  </>),
  filter: ic(<><path d="M3 4h18l-7 9v7l-4-2v-5L3 4z" /></>),
  sun: ic(<>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </>),
  moon: ic(<><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></>),
  satellite: ic(<>
    <path d="m4 10 4-4 5 5-4 4z" />
    <path d="m10 4 4-4 5 5-4 4" />
    <path d="M19 15a4 4 0 1 1-8 0" />
    <path d="M21 10a8 8 0 0 1-8 8" />
  </>),
  clock: ic(<>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v5l3 2" />
  </>),
  zap: ic(<><path d="M13 2 3 14h7l-1 8 10-12h-7l1-8z" /></>),
  database: ic(<>
    <ellipse cx="12" cy="5" rx="9" ry="3" />
    <path d="M3 5v14a9 3 0 0 0 18 0V5" />
    <path d="M3 12a9 3 0 0 0 18 0" />
  </>),
  dot: ic(<circle cx="12" cy="12" r="5" fill="currentColor" stroke="none" />),
  mail: ic(<>
    <rect x="3" y="5" width="18" height="14" rx="2" />
    <path d="M3 7l9 6 9-6" />
  </>),
  briefcase: ic(<>
    <rect x="3" y="7" width="18" height="13" rx="2" />
    <path d="M9 7V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2" />
    <path d="M3 13h18" />
  </>),
  id_card: ic(<>
    <rect x="3" y="5" width="18" height="14" rx="2" />
    <circle cx="9" cy="12" r="2.2" />
    <path d="M14 11h5" />
    <path d="M14 14h4" />
  </>),
};
