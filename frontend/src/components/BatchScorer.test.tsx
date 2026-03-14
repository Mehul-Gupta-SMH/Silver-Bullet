import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { BatchScorer } from './BatchScorer';

vi.mock('../services/api', () => ({
  predictBatch: vi.fn(),
}));

const noop = () => {};

describe('BatchScorer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('keeps Analyse Batch disabled until pairs are loaded from examples', async () => {
    render(<BatchScorer mode="reference-vs-generated" onSave={noop} />);

    const actionButton = screen.getByRole('button', { name: /analyse batch/i });
    expect(actionButton).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: /example test cases/i }));
    fireEvent.click(screen.getAllByRole('button', { name: /load example/i })[0]);

    await waitFor(() =>
      expect(screen.getByRole('button', { name: /analyse batch/i })).toBeEnabled(),
    );
  });

  it('calls predictBatch and shows results on success', async () => {
    const { predictBatch } = await import('../services/api');
    vi.mocked(predictBatch).mockResolvedValue({
      results: Array.from({ length: 8 }, (_, i) => ({
        prediction: i % 2,
        probability: 0.5,
      })),
    });

    render(<BatchScorer mode="reference-vs-generated" onSave={noop} />);
    fireEvent.click(screen.getByRole('button', { name: /example test cases/i }));
    fireEvent.click(screen.getAllByRole('button', { name: /load example/i })[0]);

    const actionButton = await screen.findByRole('button', { name: /analyse batch/i });
    fireEvent.click(actionButton);

    await screen.findByText(/all results/i);
    await waitFor(() => expect(predictBatch).toHaveBeenCalledTimes(1));
    const [payload] = vi.mocked(predictBatch).mock.calls[0];
    expect(Array.isArray(payload)).toBe(true);
    expect(payload).toHaveLength(8);
  });

  it('surfaces JSON parse errors from uploaded files', async () => {
    const badJson = '[{"text1":"only one field"}]';

    class MockFileReader {
      result: string | ArrayBuffer | null = null;
      onload: ((event: ProgressEvent<FileReader>) => void) | null = null;
      readAsText() {
        this.result = badJson;
        this.onload?.({ target: { result: badJson } } as unknown as ProgressEvent<FileReader>);
      }
    }

    vi.stubGlobal('FileReader', MockFileReader as unknown as typeof FileReader);

    const { container } = render(<BatchScorer mode="reference-vs-generated" onSave={noop} />);
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File([badJson], 'bad.json', { type: 'application/json' });

    fireEvent.change(input, { target: { files: [file] } });

    expect(await screen.findByText(/missing text1 or text2/i)).toBeInTheDocument();
  });
});
