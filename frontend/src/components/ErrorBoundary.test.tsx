import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ErrorBoundary } from './ErrorBoundary';

const ThrowingChild = () => {
  throw new Error('boom');
};

const SafeChild = () => <div>safe content</div>;

describe('ErrorBoundary', () => {
  const originalError = console.error;

  beforeEach(() => {
    console.error = vi.fn();
  });

  afterEach(() => {
    console.error = originalError;
  });

  it('renders fallback UI when a child throws', () => {
    render(
      <ErrorBoundary>
        <ThrowingChild />
      </ErrorBoundary>,
    );

    expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
    expect(screen.getByText('boom')).toBeInTheDocument();
  });

  it('recovers when retry is clicked after replacing the child', () => {
    const { rerender } = render(
      <ErrorBoundary>
        <ThrowingChild />
      </ErrorBoundary>,
    );

    rerender(
      <ErrorBoundary>
        <SafeChild />
      </ErrorBoundary>,
    );

    fireEvent.click(screen.getByRole('button', { name: /try again/i }));
    expect(screen.getByText('safe content')).toBeInTheDocument();
  });
});
