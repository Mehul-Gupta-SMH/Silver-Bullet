import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ScoreGauge } from './ScoreGauge';

describe('ScoreGauge', () => {
  it('shows green bar for high probability (>= 0.7)', () => {
    const { container } = render(<ScoreGauge probability={0.85} prediction={1} />);
    expect(container.querySelector('.bg-emerald-500')).toBeInTheDocument();
  });

  it('shows yellow bar for medium probability (0.4–0.69)', () => {
    const { container } = render(<ScoreGauge probability={0.55} prediction={0} />);
    expect(container.querySelector('.bg-amber-400')).toBeInTheDocument();
  });

  it('shows red bar for low probability (< 0.4)', () => {
    const { container } = render(<ScoreGauge probability={0.2} prediction={0} />);
    expect(container.querySelector('.bg-red-500')).toBeInTheDocument();
  });

  it('shows "Similar" badge when prediction is 1', () => {
    render(<ScoreGauge probability={0.85} prediction={1} />);
    expect(screen.getByText('Similar')).toBeInTheDocument();
  });

  it('shows "Different" badge when prediction is 0', () => {
    render(<ScoreGauge probability={0.2} prediction={0} />);
    expect(screen.getByText('Different')).toBeInTheDocument();
  });
});
