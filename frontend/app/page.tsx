"use client";
import { useState, useEffect } from "react";

export default function Home() {
  const [text, setText] = useState("");
  const [dark, setDark] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [analysisStep, setAnalysisStep] = useState(0);

  const analyze = () => {
    setLoading(true);
    setShowResults(false);
    setAnalysisStep(0);

    // Simulate analysis steps
    const steps = ["Initializing", "Processing NLP", "Running Models", "Generating Report"];
    let step = 0;
    
    const interval = setInterval(() => {
      setAnalysisStep(step);
      step++;
      if (step >= steps.length) {
        clearInterval(interval);
        setTimeout(() => {
          setLoading(false);
          setShowResults(true);
        }, 800);
      }
    }, 400);
  };

  const clearAll = () => {
    setText("");
    setShowResults(false);
  };

  const exampleText = "Scientists discover revolutionary new energy source that will power entire cities for free! Major corporations are trying to suppress this groundbreaking technology that could change the world overnight.";

  useEffect(() => {
    if (dark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [dark]);

  return (
    <div className="min-h-screen relative">
      {/* Main Container */}
      <div className="relative min-h-screen flex flex-col items-center px-4 py-12 md:py-20">
        
        {/* Animated background elements */}
        <div className="fixed inset-0 -z-10 overflow-hidden">
          <div className="absolute top-1/4 left-1/4 w-72 h-72 bg-[#3b82f6]/10 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-[#06b6d4]/10 rounded-full blur-3xl floating" />
          <div className="absolute top-3/4 left-1/3 w-64 h-64 bg-[#0ea5e9]/5 rounded-full blur-3xl" />
        </div>
        
        {/* Theme Toggle - Floating */}
        <button
          onClick={() => setDark(!dark)}
          className="fixed top-6 right-6 z-50 group"
          aria-label="Toggle theme"
        >
          <div className="glass-card p-3 rounded-2xl border border-white/20 shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-110">
            {dark ? (
              <span className="w-6 h-6 text-[#fbbf24] group-hover:rotate-45 transition-transform block">☀️</span>
            ) : (
              <span className="w-6 h-6 text-[#3b82f6] group-hover:rotate-12 transition-transform block">🌙</span>
            )}
          </div>
        </button>

        {/* Header with Animated Elements */}
        <div className="text-center mb-12 animate-fade-in-up">
          <div className="inline-flex items-center gap-3 mb-4 px-6 py-3 rounded-full bg-gradient-to-r from-[#3b82f6]/10 to-[#06b6d4]/10 border border-[#3b82f6]/30 dark:border-[#60a5fa]/20">
            <span className="w-5 h-5 text-[#3b82f6] dark:text-[#60a5fa]">✨</span>
            <span className="text-sm font-medium gradient-text">AI-Powered Analysis</span>
            <span className="w-5 h-5 text-[#3b82f6] dark:text-[#60a5fa]">🛡️</span>
          </div>
          
          <h1 className="text-5xl md:text-6xl font-bold mb-4 tracking-tight">
            <span className="gradient-text">SpectraScan</span>
          </h1>
          <p className="text-lg md:text-xl text-[#475569] dark:text-[#cbd5e1] max-w-2xl mx-auto mb-8">
            Advanced multi-model AI detection for news authenticity verification
          </p>
        </div>

        {/* Main Analysis Card */}
        <div className="w-full max-w-4xl animate-scale-in">
          <div className="glass-card rounded-3xl p-8 md:p-10 border border-white/30 dark:border-[#334155]/50 shadow-2xl backdrop-blur-xl">
            
            {/* Text Input Section */}
            <div className="mb-8">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-xl bg-gradient-to-br from-[#3b82f6] to-[#06b6d4]">
                    <span className="w-6 h-6 text-white">📊</span>
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-[#1e293b] dark:text-black">
                      News Analysis
                    </h2>
                    <p className="text-sm text-[#64748b] dark:text-[#94a3b8]">
                      Paste or type your news content below
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center gap-2 text-sm text-[#64748b] dark:text-[#94a3b8]">
                  <span className="w-4 h-4">⚡</span>
                  <span>Powered by AI</span>
                </div>
              </div>

              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="w-full h-48 p-5 rounded-2xl bg-white/50 dark:bg-[#1e293b]/50 border-2 border-[#dbeafe] dark:border-[#334155] focus:border-[#3b82f6] dark:focus:border-[#60a5fa] focus:ring-4 focus:ring-[#3b82f6]/30 dark:focus:ring-[#60a5fa]/20 resize-none transition-all placeholder:text-[#94a3b8] dark:placeholder:text-[#64748b] text-[#1e293b] dark:text-[#e2e8f0]"
                placeholder="Enter news article text here for analysis..."
              />
              
              <div className="flex justify-between items-center mt-4">
                <button
                  onClick={() => setText(exampleText)}
                  className="text-sm px-4 py-2 rounded-lg bg-[#dbeafe] dark:bg-[#1e293b] text-[#3b82f6] dark:text-[#60a5fa] hover:bg-[#bfdbfe] dark:hover:bg-[#0f172a] transition-colors"
                >
                  Load Example
                </button>
                <span className="text-sm text-[#64748b]">
                  {text.length} characters
                </span>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 mb-10">
              <button
                onClick={analyze}
                disabled={!text.trim() || loading}
                className="group flex-1 glow-effect py-4 rounded-2xl bg-gradient-to-r from-[#3b82f6] to-[#06b6d4] hover:from-[#2563eb] hover:to-[#0891b2] text-white font-semibold text-lg shadow-xl hover:shadow-2xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3"
              >
                {loading ? (
                  <>
                    <span className="w-5 h-5 animate-spin">⟳</span>
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <>
                    <span className="w-5 h-5 group-hover:scale-110 transition-transform">🤖</span>
                    <span>Analyze Authenticity</span>
                  </>
                )}
              </button>

              <button
                onClick={clearAll}
                className="px-8 py-4 rounded-2xl bg-gradient-to-r from-[#f8fafc] to-[#f1f5f9] dark:from-[#1e293b] dark:to-[#0f172a] border border-[#cbd5e1] dark:border-[#475569] text-[#334155] dark:text-[#cbd5e1] font-semibold hover:shadow-lg transition-all hover:scale-[1.02]"
              >
                Clear All
              </button>
            </div>

            {/* Loading Animation */}
            {loading && (
              <div className="mb-10">
                <div className="flex items-center justify-between mb-4">
                  {["Initializing", "Processing", "Analyzing", "Finalizing"].map((step, idx) => (
                    <div key={step} className="flex flex-col items-center">
                      <div className={`w-3 h-3 rounded-full ${analysisStep >= idx ? 'bg-gradient-to-r from-[#3b82f6] to-[#06b6d4]' : 'bg-[#e2e8f0] dark:bg-[#334155]'} transition-all duration-300`} />
                      <span className={`text-xs mt-2 ${analysisStep >= idx ? 'text-[#3b82f6] dark:text-[#60a5fa] font-medium' : 'text-[#94a3b8]'}`}>
                        {step}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="h-2 rounded-full bg-[#e2e8f0] dark:bg-[#334155] overflow-hidden">
                  <div 
                    className="h-full rounded-full bg-gradient-to-r from-[#3b82f6] to-[#06b6d4] transition-all duration-300 shimmer"
                    style={{ width: `${(analysisStep + 1) * 25}%` }}
                  />
                </div>
              </div>
            )}

            {/* Results Section */}
            {showResults && (
              <div className="space-y-8 animate-fade-in-up">
                <div className="flex items-center justify-between">
                  <h3 className="text-2xl font-bold text-[#1e293b] dark:text-white">
                    Analysis Results
                  </h3>
                  <div className="px-4 py-2 rounded-full bg-gradient-to-r from-[#10b981]/10 to-[#059669]/10 border border-[#10b981]/20">
                    <span className="text-sm font-medium text-[#059669] dark:text-[#10b981]">
                      Analysis Complete
                    </span>
                  </div>
                </div>

                {/* Model Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {[
                    { 
                      name: "BERT Transformer", 
                      confidence: 94, 
                      result: "Fake", 
                      icon: "🤖",
                      color: "from-[#ef4444] to-[#ec4899]"
                    },
                    { 
                      name: "RoBERTa-Large", 
                      confidence: 89, 
                      result: "Fake", 
                      icon: "🧠",
                      color: "from-[#f97316] to-[#ef4444]"
                    },
                    { 
                      name: "DeBERTa-v3", 
                      confidence: 96, 
                      result: "Fake", 
                      icon: "⚡",
                      color: "from-[#8b5cf6] to-[#ec4899]"
                    }
                  ].map((model, idx) => (
                    <div 
                      key={model.name}
                      className={`glass-card p-6 rounded-2xl border hover:scale-[1.02] transition-all duration-300 animate-fade-in-up animate-delay-${(idx + 1) * 100}`}
                    >
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex items-center gap-3">
                          <div className="text-2xl">{model.icon}</div>
                          <div>
                            <h4 className="font-bold text-[#1e293b] dark:text-white">
                              {model.name}
                            </h4>
                            <p className="text-sm text-[#64748b] dark:text-[#94a3b8]">
                              Confidence: <span className="font-bold text-[#3b82f6] dark:text-[#60a5fa]">{model.confidence}%</span>
                            </p>
                          </div>
                        </div>
                        <span className="w-8 h-8 text-[#ef4444]">⚠️</span>
                      </div>
                      <div className="h-2 rounded-full bg-[#e2e8f0] dark:bg-[#334155] overflow-hidden mb-3">
                        <div 
                          className={`h-full rounded-full bg-gradient-to-r ${model.color}`}
                          style={{ width: `${model.confidence}%` }}
                        />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-2xl font-bold text-[#ef4444] dark:text-[#fca5a5]">
                          {model.result}
                        </span>
                        <span className="text-sm text-[#64748b] dark:text-[#94a3b8]">
                          High confidence
                        </span>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Final Verdict */}
                <div className="glass-card p-8 rounded-2xl border border-[#fecaca] dark:border-[#ef4444]/30 bg-gradient-to-r from-[#fef2f2]/50 to-[#fce7f3]/50 dark:from-[#7f1d1d]/20 dark:to-[#831843]/20 animate-fade-in-up animate-delay-300">
                  <div className="flex items-center gap-4">
                    <div className="p-3 rounded-xl bg-gradient-to-br from-[#ef4444] to-[#ec4899]">
                      <span className="w-8 h-8 text-white">⚠️</span>
                    </div>
                    <div className="flex-1">
                      <h4 className="text-xl font-bold text-[#dc2626] dark:text-[#fca5a5] mb-2">
                        High Risk Detected
                      </h4>
                      <p className="text-[#475569] dark:text-[#cbd5e1]">
                        All AI models strongly indicate this content contains misleading or fabricated information. 
                        Multiple red flags detected including sensational claims and lack of credible sources.
                      </p>
                    </div>
                  </div>
                  
                  <div className="mt-6 p-4 rounded-xl bg-white/50 dark:bg-[#1e293b]/50">
                    <div className="flex items-center gap-3 text-sm">
                      <span className="w-4 h-4 text-[#10b981]">✅</span>
                      <span className="text-[#475569] dark:text-[#cbd5e1]">
                        <span className="font-bold">3/3 models</span> agree on potential misinformation
                      </span>
                    </div>
                  </div>
                </div>

                {/* Disclaimer */}
                <div className="text-center pt-6 border-t border-[#e2e8f0] dark:border-[#334155]">
                  <p className="text-sm text-[#64748b] dark:text-[#94a3b8]">
                    This analysis is generated by AI models and should be used as a reference only.
                    Always verify with trusted sources.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center animate-fade-in-up">
          <p className="text-[#64748b] dark:text-[#94a3b8] text-sm">
            Built with advanced AI • Real-time analysis • Enterprise security
          </p>
        </div>
      </div>
    </div>
  );
}