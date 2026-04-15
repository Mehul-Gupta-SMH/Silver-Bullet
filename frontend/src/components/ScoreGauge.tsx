interface Props {
  probability: number;
  prediction: number;
}

export function ScoreGauge({ probability, prediction }: Props) {
  const isHigh = probability >= 0.7;
  const isMid  = probability >= 0.4;

  const scoreColor = isHigh ? '#34D399' : isMid ? '#F59E0B' : '#F87171';
  const glowColor  = isHigh ? 'rgba(52,211,153,0.18)' : isMid ? 'rgba(245,158,11,0.18)' : 'rgba(248,113,113,0.18)';
  const dimColor   = isHigh ? 'rgba(52,211,153,0.07)'  : isMid ? 'rgba(245,158,11,0.07)'  : 'rgba(248,113,113,0.07)';

  // SVG arc parameters
  const r = 52;
  const cx = 70;
  const cy = 70;
  const circumference = 2 * Math.PI * r;     // 326.7
  const arcFraction   = 0.75;               // 270° sweep
  const arcLength     = circumference * arcFraction;
  const gapLength     = circumference * (1 - arcFraction);

  // Fill length based on probability
  const fillLength = arcLength * probability;

  // The arc starts at -225deg (bottom-left, going clockwise).
  // We achieve this by rotating the circle -225deg around its center.
  const rotateOffset = -225;

  return (
    <div style={{
      background: 'var(--bg-2)',
      border: '1px solid var(--border)',
      borderRadius: 12,
      padding: '20px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 12,
    }}>
      <div style={{
        fontFamily: 'var(--font-mono)',
        fontSize: 10,
        color: 'var(--text-3)',
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        alignSelf: 'flex-start',
      }}>
        Faithfulness Score
      </div>

      {/* Arc gauge */}
      <div style={{ position: 'relative', width: 140, height: 140 }}>
        <svg width="140" height="140" viewBox="0 0 140 140" style={{ overflow: 'visible' }}>
          <defs>
            <filter id="arc-glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Track arc */}
          <circle
            cx={cx} cy={cy} r={r}
            fill="none"
            stroke={dimColor}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={`${arcLength} ${gapLength}`}
            strokeDashoffset={0}
            transform={`rotate(${rotateOffset} ${cx} ${cy})`}
          />

          {/* Fill arc */}
          <circle
            cx={cx} cy={cy} r={r}
            fill="none"
            stroke={scoreColor}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={`${fillLength} ${circumference - fillLength}`}
            strokeDashoffset={0}
            transform={`rotate(${rotateOffset} ${cx} ${cy})`}
            filter="url(#arc-glow)"
            style={{
              transition: 'stroke-dasharray 0.9s cubic-bezier(0.16, 1, 0.3, 1), stroke 0.3s',
              filter: `drop-shadow(0 0 4px ${glowColor})`,
            }}
          />

          {/* Tick marks at 0, 0.5, 1.0 */}
          {[0, 0.5, 1].map((tick) => {
            const angleDeg = rotateOffset + tick * 270;
            const rad = (angleDeg * Math.PI) / 180;
            const x1 = cx + (r - 14) * Math.cos(rad);
            const y1 = cy + (r - 14) * Math.sin(rad);
            const x2 = cx + (r - 8)  * Math.cos(rad);
            const y2 = cy + (r - 8)  * Math.sin(rad);
            return (
              <line key={tick} x1={x1} y1={y1} x2={x2} y2={y2}
                stroke={tick === 0 || tick === 1 ? 'var(--text-3)' : 'var(--border-2)'}
                strokeWidth="1.5" strokeLinecap="round"
              />
            );
          })}

          {/* Center: score number */}
          <text
            x={cx} y={cy - 4}
            textAnchor="middle" dominantBaseline="middle"
            fill={scoreColor}
            fontFamily="var(--font-mono)"
            fontSize="26"
            fontWeight="400"
            letterSpacing="-1"
            style={{ transition: 'fill 0.3s' }}
          >
            {probability.toFixed(3)}
          </text>

          {/* Center: label */}
          <text
            x={cx} y={cy + 16}
            textAnchor="middle" dominantBaseline="middle"
            fill="var(--text-3)"
            fontFamily="var(--font-mono)"
            fontSize="9"
            letterSpacing="1.5"
            textDecoration=""
          >
            {(probability * 100).toFixed(1)}%
          </text>
        </svg>
      </div>

      {/* Prediction badge */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '5px 14px',
        borderRadius: 99,
        background: prediction === 1 ? 'rgba(52,211,153,0.12)' : 'rgba(248,113,113,0.12)',
        border: `1px solid ${prediction === 1 ? 'rgba(52,211,153,0.3)' : 'rgba(248,113,113,0.3)'}`,
        fontFamily: 'var(--font-mono)',
        fontSize: 11,
        fontWeight: 500,
        color: prediction === 1 ? '#34D399' : '#F87171',
        letterSpacing: '0.08em',
        textTransform: 'uppercase' as const,
      }}>
        <span style={{
          width: 6, height: 6, borderRadius: '50%',
          background: prediction === 1 ? '#34D399' : '#F87171',
          display: 'inline-block',
        }} />
        {prediction === 1 ? 'Similar' : 'Different'}
      </div>

      {/* Tick labels */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        width: '100%',
        paddingInline: 4,
      }}>
        {['0', '0.5', '1.0'].map((v) => (
          <span key={v} style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 9,
            color: 'var(--text-3)',
          }}>
            {v}
          </span>
        ))}
      </div>
    </div>
  );
}
