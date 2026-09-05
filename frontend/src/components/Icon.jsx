import React from 'react'

/* A small inline icon set, 16px, 1.5px stroke - the same stroke the peak
   markers use. Replaces the "↑ √ ▣ ‣" glyphs the first build used as icons,
   which the design system rejects on sight. */
const PATHS = {
  up: 'M8 13V3M3.5 7.5 8 3l4.5 4.5',
  folder: 'M2 4.5A1.5 1.5 0 0 1 3.5 3h3l1.5 1.5h4.5A1.5 1.5 0 0 1 14 6v6a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 2 12z',
  campaign: 'M2 4.5A1.5 1.5 0 0 1 3.5 3h3l1.5 1.5h4.5A1.5 1.5 0 0 1 14 6v6a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 2 12zM5 10c1 0 1.5-3 2.5-3S9 10 10 10s1-2 1-2',
  sqrt: 'M2 9h2l2 4 3-10h5',
  chev: 'M6 3.5 10.5 8 6 12.5',
  chevd: 'M3.5 6 8 10.5 12.5 6',
  x: 'M4 4l8 8M12 4l-8 8',
  check: 'M3 8.5l3 3 7-7',
  plus: 'M8 3v10M3 8h10',
  print: 'M4 6V2.5h8V6M4 11H2.5V7h11v4H12M4.5 9.5h7v4h-7z',
  sun: 'M8 4.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7zM8 1v1.5M8 13.5V15M1 8h1.5M13.5 8H15M3 3l1 1M12 12l1 1M3 13l1-1M12 4l1-1',
  moon: 'M13 9.5A5.5 5.5 0 0 1 6.5 3 5.5 5.5 0 1 0 13 9.5z',
  system: 'M2.5 3.5h11v7h-11zM6 13.5h4M8 10.5v3',
  info: 'M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13zM8 7v4M8 5v.5',
  warn: 'M8 2.5 14 13H2zM8 6.5v3M8 11v.5',
  book: 'M2.5 3.5h4.5A1.5 1.5 0 0 1 8.5 5v8.5A1.5 1.5 0 0 0 7 12H2.5zM13.5 3.5H9A1.5 1.5 0 0 0 7.5 5v8.5A1.5 1.5 0 0 1 9 12h4.5z',
  external: 'M9 3h4v4M13 3 7 9M11 9v4H3V5h4',
  drag: 'M2 8h12M5 5 2 8l3 3M11 5l3 3-3 3',
  refresh: 'M13 8a5 5 0 1 1-1.5-3.5M13 2.5v3h-3',
  undo: 'M3 6.5h7a3 3 0 0 1 0 6H7M3 6.5 5.5 4M3 6.5 5.5 9',
  download: 'M8 2v8M4.5 6.5 8 10l3.5-3.5M3 13h10',
  upload: 'M8 10V2M4.5 5.5 8 2l3.5 3.5M3 13h10',
  flask: 'M6 2h4M6.5 2v4L3 12.5A1 1 0 0 0 4 14h8a1 1 0 0 0 1-1.5L9.5 6V2',
  lock: 'M4 7V5a4 4 0 0 1 8 0v2M3.5 7h9v6.5h-9z',
}

export default function Icon({ name, size = 16, className = '', label }) {
  const d = PATHS[name] || PATHS.info
  return (
    <svg className={`icon ${className}`} width={size} height={size} viewBox="0 0 16 16"
         fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"
         strokeLinejoin="round" aria-hidden={label ? undefined : 'true'}
         role={label ? 'img' : undefined} aria-label={label} focusable="false">
      <path d={d} />
    </svg>
  )
}
