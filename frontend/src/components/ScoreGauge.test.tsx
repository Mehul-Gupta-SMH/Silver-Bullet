import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ScoreGauge } from './ScoreGauge';

describe('ScoreGauge', () => {
  it('shows green arc for high probability (>= 0.7)', () => {
    const { container } = render(<ScoreGauge probability={0.85} prediction={1} />);
    const fillArc = container.querySelectorAll('circle')[1];
    expect(fillArc).toBeTruthy();
    expect(fillArc.getAttribute('stroke')).toBe('#34D399');
  });

  it('shows yellow arc for medium probability (0.4–0.69)', () => {
    const { container } = render(<ScoreGauge probability={0.55} prediction={0} />);
    const fillArc = container.querySelectorAll('circle')[1];
    expect(fillArc).toBeTruthy();
    expect(fillArc.getAttribute('stroke')).toBe('#F59E0B');
  });

  it('shows red arc for low probability (< 0.4)', () => {
    const { container } = render(<ScoreGauge probability={0.2} prediction={0} />);
    const fillArc = container.querySelectorAll('circle')[1];
    expect(fillArc).toBeTruthy();
    expect(fillArc.getAttribute('stroke')).toBe('#F87171');
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
