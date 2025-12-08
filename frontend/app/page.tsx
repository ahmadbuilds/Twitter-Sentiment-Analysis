'use client';
import React, { useState } from 'react';
import { Send, TrendingUp, TrendingDown } from 'lucide-react';

type SentimentResult = {
  sentiment: 'positive' | 'negative';
  confidence: number;
};

export default function SentimentAnalyzer() {
  const [tweet, setTweet] = useState('');
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const analyzeSentiment = async () => {
    if (!tweet.trim()) {
      setError('Please enter a post to analyze');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch(process.env.NEXT_PUBLIC_BACKEND_SERVER_URL+'/tweet', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: tweet }),
      });

      if (!response.ok) {
        throw new Error('Failed to analyze sentiment');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Unable to connect to the API. Make sure the backend is running.'+err);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      analyzeSentiment();
    }
  };

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-6">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-white rounded-xl mb-4">
            <svg viewBox="0 0 24 24" className="w-10 h-10" fill="black">
              <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
            </svg>
          </div>
          <h1 className="text-4xl font-bold text-white mb-2">
            𝕏 Sentiment Analysis
          </h1>
          <p className="text-gray-400">Analyze the sentiment of any post instantly</p>
        </div>

        {/* Main Card */}
        <div className="bg-neutral-950 rounded-2xl border border-neutral-800 p-8">
          {/* Input Area */}
          <div className="mb-6">
            <label className="block text-gray-300 text-sm font-medium mb-3">
              Post Text
            </label>
            <textarea
              value={tweet}
              onChange={(e) => setTweet(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="What's happening?"
              className="w-full bg-black text-white rounded-xl p-4 border border-neutral-800 focus:border-white focus:ring-1 focus:ring-white outline-none resize-none transition-colors placeholder-gray-600"
              rows={6}
            />
            <div className="flex justify-between items-center mt-2 text-sm">
              <span className="text-gray-500">{tweet.split(/\s+/).filter(word => word.length > 0).length}/128 words</span>
              {error && <span className="text-red-400">{error}</span>}
            </div>
          </div>

          {/* Analyze Button */}
          <button
            onClick={analyzeSentiment}
            disabled={loading}
            className="w-full bg-white hover:bg-gray-200 text-black font-semibold py-3 px-6 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <div className="w-5 h-5 border-2 border-black border-t-transparent rounded-full animate-spin"></div>
                Analyzing...
              </>
            ) : (
              <>
                <Send className="w-5 h-5" />
                Analyze Sentiment
              </>
            )}
          </button>

          {/* Results */}
          {result && (
            <div className="mt-6">
              <div className={`rounded-xl p-6 border ${
                result.sentiment === 'positive' 
                  ? 'bg-emerald-950/30 border-emerald-700' 
                  : 'bg-rose-950/30 border-rose-700'
              }`}>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    {result.sentiment === 'positive' ? (
                      <TrendingUp className="w-8 h-8 text-emerald-400" />
                    ) : (
                      <TrendingDown className="w-8 h-8 text-rose-400" />
                    )}
                    <h3 className={`text-2xl font-bold ${
                      result.sentiment == 'positive' ? 'text-emerald-400' : 'text-rose-400'
                    }`}>
                      {result.sentiment}
                    </h3>
                  </div>
                  <div className="text-right">
                    <div className="text-3xl font-bold text-white">
                      {(result.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="text-gray-400 text-sm">Probability</div>
                  </div>
                </div>

                {/* Confidence Bar */}
                <div className="w-full bg-neutral-900 rounded-full h-2">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${
                      result.sentiment == 'positive' 
                        ? 'bg-emerald-500' 
                        : 'bg-rose-500'
                    }`}
                    style={{ width: `${result.confidence * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="text-center mt-6 text-gray-600 text-sm">
          Powered by AI
        </div>
      </div>
    </div>
  );
}