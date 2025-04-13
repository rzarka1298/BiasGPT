// src/App.jsx
import React, { useState } from 'react';

function App() {
  // State variables for input text, result data, and loading indicator.
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Function to handle the form submission.
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      // Replace the URL with your backend's URL if it's different.
      const response = await fetch("http://127.0.0.1:8000/swap_and_analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: inputText, do_gpt: false })
      });

      if (!response.ok) {
        throw new Error("Failed to connect to the backend");
      }
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      setResult({ error: error.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <h1 className="text-3xl font-bold mb-6">BiasGPT Analysis</h1>
      
      <form onSubmit={handleSubmit} className="w-full max-w-lg bg-white shadow p-6 rounded">
        <label htmlFor="inputText" className="block text-gray-700 font-medium mb-2">
          Enter Text:
        </label>
        <textarea
          id="inputText"
          className="w-full p-2 border border-gray-300 rounded mb-4"
          rows={4}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Type your text here..."
          required
        />
        <button
          type="submit"
          className="w-full bg-blue-500 text-white py-2 rounded hover:bg-blue-600 transition"
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </form>

      {result && (
        <div className="mt-6 w-full max-w-lg bg-white shadow p-6 rounded">
          {result.error ? (
            <p className="text-red-500">Error: {result.error}</p>
          ) : (
            <>
              <h2 className="text-xl font-semibold mb-4">Analysis Result</h2>
              <p>
                <strong>Original:</strong> {result.original_text}
              </p>
              <p>
                <strong>Swapped:</strong> {result.swapped_text}
              </p>
              <p>
                <strong>Prediction:</strong> {result.prediction}
              </p>
              {result.gpt_output && (
                <p>
                  <strong>GPT Output:</strong> {result.gpt_output}
                </p>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
