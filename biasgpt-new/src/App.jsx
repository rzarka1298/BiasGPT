import { useState } from 'react'

function App() {
  const [inputText, setInputText] = useState("")
  const [queryResult, setQueryResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleQuery = async (e) => {
    e.preventDefault()
    setLoading(true)
    setQueryResult(null)
    try {
      const response = await fetch("http://127.0.0.1:8000/swap_and_analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: inputText, do_gpt: false }),
      })
      if (!response.ok) {
        throw new Error("Failed to fetch response from backend")
      }
      const data = await response.json()
      setQueryResult(data)
    } catch (error) {
      setQueryResult({ error: error.message })
    }
    setLoading(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900 text-gray-100 flex flex-col items-center">
      <div className="w-full max-w-4xl px-4 py-16">
        {/* Header */}
        <header className="text-center mb-16">
          <div className="flex flex-col items-center space-y-6">
            <div className="w-24 h-24 bg-gradient-to-br from-white to-gray-100 rounded-full flex items-center justify-center shadow-2xl transform hover:scale-105 transition-all duration-500">
              <span className="text-5xl font-bold bg-gradient-to-r from-purple-500 to-pink-500 bg-clip-text text-transparent">B</span>
            </div>
            <h1 className="text-6xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">BiasGPT</h1>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">Smartly detect and mitigate bias with AI-driven demographic swaps</p>
          </div>
        </header>

        {/* Main Content */}
        <main className="space-y-8">
          {/* Input Section */}
          <div className="bg-white/5 backdrop-blur-lg rounded-3xl p-8 border border-white/10 shadow-2xl">
            <form onSubmit={handleQuery} className="space-y-6">
              <label htmlFor="query" className="block text-xl font-medium text-gray-200 text-center">
                Enter your prompt:
              </label>
              <div className="max-w-2xl mx-auto">
                <textarea
                  id="query"
                  rows="4"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  className="w-full p-4 bg-white/5 border border-white/10 rounded-2xl focus:outline-none focus:ring-2 focus:ring-purple-500 text-white placeholder-gray-400 resize-none"
                  placeholder="e.g., Tell a short story about a doctor named John."
                  required
                />
              </div>
              <div className="max-w-2xl mx-auto">
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full py-3 px-6 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-2xl font-medium hover:from-purple-700 hover:to-pink-700 transition-all duration-300 disabled:opacity-50"
                >
                  {loading ? 'Analyzing...' : 'Analyze'}
                </button>
              </div>
            </form>
          </div>

          {/* Results Section */}
          {queryResult && (
            <div className="bg-white/5 backdrop-blur-lg rounded-3xl p-8 border border-white/10 shadow-2xl">
              {queryResult.error ? (
                <p className="text-red-400 text-center">{queryResult.error}</p>
              ) : (
                <div className="space-y-6">
                  <h2 className="text-2xl font-bold text-white text-center">Analysis Results</h2>
                  <div className="space-y-4 max-w-2xl mx-auto">
                    <div className="bg-white/5 p-4 rounded-xl">
                      <h3 className="text-lg font-medium text-gray-300">Original Text</h3>
                      <p className="mt-2 text-white">{queryResult.original_text}</p>
                    </div>
                    <div className="bg-white/5 p-4 rounded-xl">
                      <h3 className="text-lg font-medium text-gray-300">Swapped Text</h3>
                      <p className="mt-2 text-white">{queryResult.swapped_text}</p>
                    </div>
                    <div className="bg-white/5 p-4 rounded-xl">
                      <h3 className="text-lg font-medium text-gray-300">Prediction</h3>
                      <p className="mt-2 text-white">{queryResult.prediction}</p>
                    </div>
                    {queryResult.gpt_output && (
                      <div className="bg-white/5 p-4 rounded-xl">
                        <h3 className="text-lg font-medium text-gray-300">GPT Output</h3>
                        <p className="mt-2 text-white">{queryResult.gpt_output}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </main>

        {/* Footer */}
        <footer className="mt-16 text-center text-gray-400">
          <p>© 2025 BiasGPT. All Rights Reserved.</p>
        </footer>
      </div>
    </div>
  )
}

export default App
