import { useState } from 'react'
import './App.css'

function App() {
  const [inputText, setInputText] = useState('')
  const [summary, setSummary] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSummarize = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text to summarize')
      return
    }

    if (inputText.length < 50) {
      setError('Please enter at least 50 characters for meaningful summarization')
      return
    }

    setIsLoading(true)
    setError('')
    setSummary('')

    try {
      const response = await fetch('http://127.0.0.1:5000/summarize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'text': inputText })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setSummary(data.summary || data.result || 'Summary generated successfully')
    } catch (err) {
      setError('Failed to generate summary. Please try again.')
      console.error('Summarization error:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleClear = () => {
    setInputText('')
    setSummary('')
    setError('')
  }

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>AI Text Summarizer</h1>
        </header>

        <div className="main-content">
          <div className="input-section">
            <div className="input-header">
              <label htmlFor="input-text">Enter your text</label>
              <span className="char-count">
                {inputText.length} characters
              </span>
            </div>
            <textarea
              id="input-text"
              className="text-input"
              placeholder="Paste your text here to generate a summary."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              disabled={isLoading}
            />
            
            <div className="button-group">
              <button
                className="summarize-btn"
                onClick={handleSummarize}
                disabled={isLoading || !inputText.trim()}
              >
                {isLoading ? (
                  <>
                    <div className="loading-spinner"></div>
                    Generating Summary...
                  </>
                ) : (
                  'Summarize Text'
                )}
              </button>
              
              <button
                className="clear-btn"
                onClick={handleClear}
                disabled={isLoading}
              >
                Clear All
              </button>
            </div>
          </div>

          {error && (
            <div className="error-message">
              <div className="error-icon">⚠️</div>
              <span>{error}</span>
            </div>
          )}

          {summary && (
            <div className="output-section">
              <div className="output-header">
                <h3>Summary</h3>
              </div>
              <div className="summary-output">
                {summary}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App