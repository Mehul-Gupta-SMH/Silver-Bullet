import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { PairScorer } from './PairScorer'

// Mock the api service so tests don't require a running server
vi.mock('../services/api', () => ({
  predictPair: vi.fn(),
}))

describe('PairScorer', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('Score button is disabled when both textareas are empty', () => {
    render(<PairScorer />)
    const button = screen.getByRole('button', { name: /score/i })
    expect(button).toBeDisabled()
  })

  it('Score button is disabled when only text1 is filled', () => {
    render(<PairScorer />)
    const textareas = screen.getAllByRole('textbox')
    fireEvent.change(textareas[0], { target: { value: 'Hello world' } })
    expect(screen.getByRole('button', { name: /score/i })).toBeDisabled()
  })

  it('Score button is disabled when only text2 is filled', () => {
    render(<PairScorer />)
    const textareas = screen.getAllByRole('textbox')
    fireEvent.change(textareas[1], { target: { value: 'Hi there' } })
    expect(screen.getByRole('button', { name: /score/i })).toBeDisabled()
  })

  it('Score button is enabled when both textareas have content', () => {
    render(<PairScorer />)
    const textareas = screen.getAllByRole('textbox')
    fireEvent.change(textareas[0], { target: { value: 'Hello world' } })
    fireEvent.change(textareas[1], { target: { value: 'Hi there' } })
    expect(screen.getByRole('button', { name: /score/i })).toBeEnabled()
  })

  it('shows loading state on submit', async () => {
    const { predictPair } = await import('../services/api')
    // Return a promise that never resolves so we can observe loading state
    vi.mocked(predictPair).mockImplementation(() => new Promise(() => {}))

    render(<PairScorer />)
    const textareas = screen.getAllByRole('textbox')
    fireEvent.change(textareas[0], { target: { value: 'Hello world' } })
    fireEvent.change(textareas[1], { target: { value: 'Hi there' } })

    const button = screen.getByRole('button', { name: /score/i })
    fireEvent.click(button)

    // Button should now show loading text and be disabled
    expect(screen.getByRole('button', { name: /scoring/i })).toBeDisabled()
  })
})
