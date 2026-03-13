import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { PairScorer } from './PairScorer';

vi.mock('../services/api', () => ({
  predictPair: vi.fn(),
}));

const DEFAULT_MODE = 'reference-vs-generated' as const;
const noop = () => {};

describe('PairScorer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('Analyse button is disabled when both textareas are empty', () => {
    render(<PairScorer mode={DEFAULT_MODE} onSave={noop} />);
    expect(screen.getByRole('button', { name: /analyse pair/i })).toBeDisabled();
  });

  it('Analyse button is disabled when only text1 is filled', () => {
    render(<PairScorer mode={DEFAULT_MODE} onSave={noop} />);
    const textareas = screen.getAllByRole('textbox');
    fireEvent.change(textareas[0], { target: { value: 'Hello world' } });
    expect(screen.getByRole('button', { name: /analyse pair/i })).toBeDisabled();
  });

  it('Analyse button is disabled when only text2 is filled', () => {
    render(<PairScorer mode={DEFAULT_MODE} onSave={noop} />);
    const textareas = screen.getAllByRole('textbox');
    fireEvent.change(textareas[1], { target: { value: 'Hi there' } });
    expect(screen.getByRole('button', { name: /analyse pair/i })).toBeDisabled();
  });

  it('Analyse button is enabled when both textareas have content', () => {
    render(<PairScorer mode={DEFAULT_MODE} onSave={noop} />);
    const textareas = screen.getAllByRole('textbox');
    fireEvent.change(textareas[0], { target: { value: 'Hello world' } });
    fireEvent.change(textareas[1], { target: { value: 'Hi there' } });
    expect(screen.getByRole('button', { name: /analyse pair/i })).toBeEnabled();
  });

  it('shows loading state on submit', async () => {
    const { predictPair } = await import('../services/api');
    vi.mocked(predictPair).mockImplementation(() => new Promise(() => {}));

    render(<PairScorer mode={DEFAULT_MODE} onSave={noop} />);
    const textareas = screen.getAllByRole('textbox');
    fireEvent.change(textareas[0], { target: { value: 'Hello world' } });
    fireEvent.change(textareas[1], { target: { value: 'Hi there' } });

    fireEvent.click(screen.getByRole('button', { name: /analyse pair/i }));
    expect(screen.getByRole('button', { name: /analysing/i })).toBeDisabled();
  });
});
