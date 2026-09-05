import React from 'react'

/* The peak mark: a resolved peak with an unresolved shoulder on its tail -
   the exact thing a campaign exists to pull apart. Drawn from tokens, so it
   inverts with the theme and never drifts from the trace palette. */
export default function Mark({ height = 28, decorative = false }) {
  return (
    <svg className="mark" viewBox="0 0 160 112" height={height}
         width={(height * 160) / 112}
         role={decorative ? undefined : 'img'}
         aria-hidden={decorative ? 'true' : undefined}
         aria-label={decorative ? undefined : 'Gradient Bench'}
         focusable="false">
      <rect x="1" y="1" width="158" height="110" rx="14"
            fill="var(--surface)" stroke="var(--rule)" strokeWidth="2" />
      <g fill="none" strokeLinecap="round">
        <path d="M28 84C42 84 47 81.5 52.5 60C57.5 40 60.5 29 64 29C67.5 29 70.5 40 75.5 60C79.5 79 87 84 116 84"
              stroke="var(--ink)" strokeOpacity="0.78" strokeWidth="6" />
        <path d="M85 84C89.5 84 92.5 76 96.5 60C100.5 76 103.5 84 108.5 84"
              stroke="var(--shoulder)" strokeWidth="5" />
      </g>
    </svg>
  )
}
