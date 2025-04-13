// src/App.jsx
import React, { useState } from 'react';

function App() {
  const [inputText, setInputText] = useState("");
  const [queryResult, setQueryResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Function to send the query to the backend API.
  const handleQuery = async (e) => {
    e.preventDefault();
    setLoading(true);
    setQueryResult(null);
    try {
      const response = await fetch("http://127.0.0.1:8000/swap_and_analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: inputText, do_gpt: false }),
      });
      if (!response.ok) {
        throw new Error("Failed to fetch response from backend");
      }
      const data = await response.json();
      setQueryResult(data);
    } catch (error) {
      setQueryResult({ error: error.message });
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-800 to-pink-900 text-gray-100 flex flex-col items-center">
      {/* Header with Logo */}
      <header className="py-16 text-center w-full max-w-4xl px-4">
        <div className="flex flex-col items-center space-y-6">
          <div className="w-32 h-32 bg-gradient-to-br from-white to-gray-100 rounded-full flex items-center justify-center shadow-2xl transform hover:scale-105 transition-all duration-500 hover:shadow-purple-500/50">
            <span className="text-6xl font-bold bg-gradient-to-r from-indigo-500 to-pink-500 bg-clip-text text-transparent">B</span>
          </div>
          <h1 className="text-8xl font-bold bg-gradient-to-r from-white via-gray-200 to-gray-300 bg-clip-text text-transparent">BiasGPT</h1>
          <p className="text-2xl text-gray-300 font-light tracking-wide max-w-2xl">Smartly detect and mitigate bias with AI-driven demographic swaps</p>
        </div>
      </header>
      
      <main className="w-full max-w-2xl px-4 flex-1 flex flex-col items-center">
        {/* Query Section */}
        <div className="w-full bg-white/5 backdrop-blur-xl rounded-[2.5rem] shadow-2xl p-10 mb-10 border border-white/10 transform hover:scale-[1.01] transition-all duration-500 hover:shadow-purple-500/20">
          <form onSubmit={handleQuery} className="space-y-8 flex flex-col items-center">
            <label htmlFor="query" className="block text-2xl font-light text-white text-center tracking-wide">
              Enter your prompt:
            </label>
            <div className="w-full max-w-md">
              <textarea
                id="query"
                rows="4"
                placeholder="e.g., Tell a short story about a doctor named John."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="w-full p-8 bg-white/5 border border-white/10 rounded-[2.5rem] focus:outline-none focus:ring-2 focus:ring-purple-500/50 text-white placeholder-gray-400/70 resize-none text-lg font-light tracking-wide backdrop-blur-sm"
                required
              />
            </div>
            <div className="w-full max-w-md">
              <button
                type="submit"
                className="w-full bg-gradient-to-r from-purple-600/90 to-pink-600/90 hover:from-purple-700/90 hover:to-pink-700/90 text-white font-light tracking-wide text-lg py-5 rounded-[2.5rem] transition-all duration-500 shadow-xl hover:shadow-2xl disabled:opacity-50 transform hover:scale-[1.02] hover:shadow-purple-500/30"
                disabled={loading}
              >
                {loading ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Analyzing...
                  </span>
                ) : "Analyze"}
              </button>
            </div>
          </form>
        </div>

        {/* Result Section */}
        {queryResult && (
          <div className="w-full bg-white/5 backdrop-blur-xl rounded-[2.5rem] shadow-2xl p-10 border border-white/10 transform hover:scale-[1.01] transition-all duration-500 hover:shadow-purple-500/20">
            {queryResult.error ? (
              <p className="text-red-400 font-light text-center tracking-wide">Error: {queryResult.error}</p>
            ) : (
              <>
                <h2 className="text-3xl font-light mb-8 text-white text-center tracking-wide">Analysis Result</h2>
                <div className="space-y-8 flex flex-col items-center">
                  <div className="w-full max-w-md bg-white/5 p-8 rounded-[2.5rem] backdrop-blur-sm">
                    <span className="font-light text-gray-300 tracking-wide">Original: </span>
                    <p className="mt-3 text-white text-lg font-light tracking-wide">{queryResult.original_text}</p>
                  </div>
                  <div className="w-full max-w-md bg-white/5 p-8 rounded-[2.5rem] backdrop-blur-sm">
                    <span className="font-light text-gray-300 tracking-wide">Swapped: </span>
                    <p className="mt-3 text-white text-lg font-light tracking-wide">{queryResult.swapped_text}</p>
                  </div>
                  <div className="w-full max-w-md bg-white/5 p-8 rounded-[2.5rem] backdrop-blur-sm">
                    <span className="font-light text-gray-300 tracking-wide">Prediction: </span>
                    <p className="mt-3 text-white text-lg font-light tracking-wide">{queryResult.prediction}</p>
                  </div>
                  {queryResult.gpt_output && (
                    <div className="w-full max-w-md bg-white/5 p-8 rounded-[2.5rem] backdrop-blur-sm">
                      <span className="font-light text-gray-300 tracking-wide">GPT Output: </span>
                      <p className="mt-3 text-white text-lg font-light tracking-wide">{queryResult.gpt_output}</p>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        )}
      </main>
      
      <footer className="mt-16 text-center py-6 w-full">
        <p className="text-gray-400/70 text-sm font-light tracking-wide">© 2025 BiasGPT. All Rights Reserved.</p>
      </footer>
    </div>
  );
}

export default App;
