'use client';

import { useState, useRef } from 'react';

export default function Home() {
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [acousticStats, setAcousticStats] = useState(null);
  const [error, setError] = useState(null);
  
  // Custom API keys inputs for testing fallback
  const [useCustomKeys, setUseCustomKeys] = useState(false);
  const [apiKeys, setApiKeys] = useState({
    gemini: '',
    openai: '',
    deepseek: '',
    minimax: ''
  });

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const fileInputRef = useRef(null);

  // Microphone recording functions
  const startRecording = async () => {
    setError(null);
    setFile(null);
    setAudioUrl(null);
    setResults(null);
    setAcousticStats(null);
    audioChunksRef.current = [];
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mediaRecorder;
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const audioFile = new File([audioBlob], 'recorded_speech.wav', { type: 'audio/wav' });
        
        setFile(audioFile);
        setAudioUrl(URL.createObjectURL(audioBlob));
        
        // Stop stream tracks
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
      
    } catch (err) {
      setError('Permission to access microphone denied or unsupported browser.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(timerRef.current);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setAudioUrl(URL.createObjectURL(selectedFile));
      setError(null);
      setResults(null);
      setAcousticStats(null);
    }
  };

  const triggerFileSelect = () => {
    fileInputRef.current.click();
  };

  // Submit file for evaluation
  const runEvaluation = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    setResults(null);
    setAcousticStats(null);
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('models', 'tensorflow,gemini,openai,deepseek,minimax');
    
    // Pass custom keys if enabled
    if (useCustomKeys) {
      if (apiKeys.gemini) formData.append('key_gemini', apiKeys.gemini);
      if (apiKeys.openai) formData.append('key_openai', apiKeys.openai);
      if (apiKeys.deepseek) formData.append('key_deepseek', apiKeys.deepseek);
      if (apiKeys.minimax) formData.append('key_minimax', apiKeys.minimax);
    }

    try {
      const res = await fetch('/app/api/evaluate', {
        method: 'POST',
        body: formData
      });
      
      const data = await res.json();
      if (res.ok && data.success) {
        setResults(data.results);
        setAcousticStats(data.acousticStats);
      } else {
        setError(data.error || 'Failed to complete evaluation.');
      }
    } catch (err) {
      setError('Connection error contacting serverless handler.');
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (secs) => {
    const m = Math.floor(secs / 60).toString().padStart(2, '0');
    const s = (secs % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
      <header>
        <h1>Voice Guard</h1>
        <p>Acoustic Forensics & LLM Reasoning Audio Deepfake Detector</p>
      </header>

      <div className="dashboard-grid">
        {/* Left Side: Controls & Input */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <div className="card">
            <h2 className="card-title">
              <span className="upload-icon">🎙️</span> Input Speech Source
            </h2>
            <div className="input-section">
              {/* Record block */}
              <div className="recorder-box">
                <button 
                  onClick={isRecording ? stopRecording : startRecording} 
                  className={`record-btn ${isRecording ? 'recording' : ''}`}
                  title={isRecording ? "Stop Recording" : "Start Recording"}
                />
                <span style={{ fontSize: '1.2rem', fontWeight: 600, color: isRecording ? '#ff3366' : '#94a3b8' }}>
                  {isRecording ? `Recording... ${formatTime(recordingTime)}` : 'Record speech sample'}
                </span>
              </div>

              <div style={{ textAlignment: 'center', color: '#64748b', fontSize: '0.9rem' }}>— or —</div>

              {/* Upload block */}
              <div onClick={triggerFileSelect} className="upload-box">
                <span className="upload-icon">📤</span>
                <span style={{ fontWeight: 500, display: 'block', marginBottom: '0.25rem' }}>Click to upload file</span>
                <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>WAV, M4A, MP3, FLAC (max 10MB)</span>
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleFileChange} 
                  accept="audio/*" 
                  style={{ display: 'none' }} 
                />
              </div>

              {file && (
                <div style={{ background: 'rgba(255, 255, 255, 0.02)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <p style={{ fontSize: '0.9rem', color: '#fff', marginBottom: '0.5rem', fontWeight: 500, wordBreak: 'break-all' }}>
                    📎 {file.name}
                  </p>
                  {audioUrl && <audio src={audioUrl} controls style={{ width: '100%', outline: 'none' }} />}
                  <button 
                    onClick={runEvaluation} 
                    disabled={loading}
                    className="btn btn-primary" 
                    style={{ width: '100%', marginTop: '1rem', justifyContent: 'center' }}
                  >
                    {loading ? 'Evaluating Model Outputs...' : 'Scan Audio Authenticity'}
                  </button>
                </div>
              )}
              
              {error && (
                <div style={{ background: 'rgba(255,51,102,0.1)', border: '1px solid rgba(255,51,102,0.2)', padding: '1rem', borderRadius: '8px', color: '#ff3366', fontSize: '0.9rem' }}>
                  ⚠️ {error}
                </div>
              )}
            </div>
          </div>
          
          {/* Custom API Keys Config */}
          <div className="card">
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontWeight: 500 }}>
              <input 
                type="checkbox" 
                checked={useCustomKeys} 
                onChange={(e) => setUseCustomKeys(e.target.checked)} 
              />
              Configure Custom API Keys
            </label>
            
            {useCustomKeys && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '1rem' }}>
                <div>
                  <label style={{ fontSize: '0.8rem', color: '#94a3b8', display: 'block', marginBottom: '0.25rem' }}>Gemini Key</label>
                  <input 
                    type="password" 
                    placeholder="GEMINI_API_KEY"
                    value={apiKeys.gemini}
                    onChange={(e) => setApiKeys({...apiKeys, gemini: e.target.value})}
                    style={{ width: '100%', padding: '0.5rem', background: '#111', border: '1px solid #333', color: '#fff', borderRadius: '4px' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: '0.8rem', color: '#94a3b8', display: 'block', marginBottom: '0.25rem' }}>OpenAI Key</label>
                  <input 
                    type="password" 
                    placeholder="OPENAI_API_KEY"
                    value={apiKeys.openai}
                    onChange={(e) => setApiKeys({...apiKeys, openai: e.target.value})}
                    style={{ width: '100%', padding: '0.5rem', background: '#111', border: '1px solid #333', color: '#fff', borderRadius: '4px' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: '0.8rem', color: '#94a3b8', display: 'block', marginBottom: '0.25rem' }}>DeepSeek Key</label>
                  <input 
                    type="password" 
                    placeholder="DEEPSEEK_API_KEY"
                    value={apiKeys.deepseek}
                    onChange={(e) => setApiKeys({...apiKeys, deepseek: e.target.value})}
                    style={{ width: '100%', padding: '0.5rem', background: '#111', border: '1px solid #333', color: '#fff', borderRadius: '4px' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: '0.8rem', color: '#94a3b8', display: 'block', marginBottom: '0.25rem' }}>MiniMax Key</label>
                  <input 
                    type="password" 
                    placeholder="MINIMAX_API_KEY"
                    value={apiKeys.minimax}
                    onChange={(e) => setApiKeys({...apiKeys, minimax: e.target.value})}
                    style={{ width: '100%', padding: '0.5rem', background: '#111', border: '1px solid #333', color: '#fff', borderRadius: '4px' }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Side: Evaluation Results */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <div className="card" style={{ minHeight: '400px' }}>
            <h2 className="card-title">🔍 Scan Dashboard</h2>
            
            {!results && !loading && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '300px', color: '#64748b' }}>
                <span style={{ fontSize: '3rem', marginBottom: '1rem' }}>📈</span>
                <p>Record or upload speech to run the deepfake scanning algorithm.</p>
              </div>
            )}

            {loading && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '300px' }}>
                <span className="verdict-badge verdict-loading" style={{ padding: '1rem 2rem', fontSize: '1.2rem' }}>
                  🤖 Running Audio Diagnostics...
                </span>
                <p style={{ marginTop: '1rem', color: '#94a3b8', fontSize: '0.9rem' }}>Evaluating neural classifiers and query-reasoning engines</p>
              </div>
            )}

            {results && (
              <div className="model-list">
                {/* 1. TensorFlow Local Classifier */}
                <div className="model-row">
                  <div className="model-meta">
                    <h3>TensorFlow Model</h3>
                    <p>YAMNet + BERT (Local Keras)</p>
                  </div>
                  <div>
                    {results.tensorflow?.error ? (
                      <span className="verdict-badge verdict-fake" style={{ background: 'rgba(255,51,102,0.05)', color: '#ff3366' }}>Offline</span>
                    ) : (
                      <span className={`verdict-badge ${results.tensorflow?.verdict === 'Fake' ? 'verdict-fake' : 'verdict-real'}`}>
                        {results.tensorflow?.verdict} ({Math.round(results.tensorflow?.confidence * 100)}%)
                      </span>
                    )}
                  </div>
                  <p className="reasoning-text">
                    {results.tensorflow?.error || results.tensorflow?.reasoning} (Latency: {results.tensorflow?.latency?.toFixed(2)}s)
                  </p>
                </div>

                {/* 2. Gemini 1.5 Flash */}
                <div className="model-row">
                  <div className="model-meta">
                    <h3>Gemini 1.5 Flash</h3>
                    <p>Google Multimodal</p>
                  </div>
                  <div>
                    {results.gemini?.error ? (
                      <span className="verdict-badge verdict-fake" style={{ background: 'rgba(255,51,102,0.05)', color: '#ff3366' }}>Error</span>
                    ) : (
                      <span className={`verdict-badge ${results.gemini?.verdict === 'Fake' ? 'verdict-fake' : 'verdict-real'}`}>
                        {results.gemini?.verdict} ({Math.round(results.gemini?.confidence * 100)}%)
                      </span>
                    )}
                  </div>
                  <p className="reasoning-text">
                    {results.gemini?.error || results.gemini?.reasoning} (Latency: {results.gemini?.latency?.toFixed(2)}s)
                  </p>
                </div>

                {/* 3. OpenAI GPT-4o */}
                <div className="model-row">
                  <div className="model-meta">
                    <h3>OpenAI GPT-4o</h3>
                    <p>Native Audio API</p>
                  </div>
                  <div>
                    {results.openai?.error ? (
                      <span className="verdict-badge verdict-fake" style={{ background: 'rgba(255,51,102,0.05)', color: '#ff3366' }}>Error</span>
                    ) : (
                      <span className={`verdict-badge ${results.openai?.verdict === 'Fake' ? 'verdict-fake' : 'verdict-real'}`}>
                        {results.openai?.verdict} ({Math.round(results.openai?.confidence * 100)}%)
                      </span>
                    )}
                  </div>
                  <p className="reasoning-text">
                    {results.openai?.error || results.openai?.reasoning} (Latency: {results.openai?.latency?.toFixed(2)}s)
                  </p>
                </div>

                {/* 4. MiniMax M3 */}
                <div className="model-row">
                  <div className="model-meta">
                    <h3>MiniMax M3</h3>
                    <p>Multimodal API</p>
                  </div>
                  <div>
                    {results.minimax?.error ? (
                      <span className="verdict-badge verdict-fake" style={{ background: 'rgba(255,51,102,0.05)', color: '#ff3366' }}>Error</span>
                    ) : (
                      <span className={`verdict-badge ${results.minimax?.verdict === 'Fake' ? 'verdict-fake' : 'verdict-real'}`}>
                        {results.minimax?.verdict} ({Math.round(results.minimax?.confidence * 100)}%)
                      </span>
                    )}
                  </div>
                  <p className="reasoning-text">
                    {results.minimax?.error || results.minimax?.reasoning} (Latency: {results.minimax?.latency?.toFixed(2)}s)
                  </p>
                </div>

                {/* 5. DeepSeek-V4-Pro */}
                <div className="model-row">
                  <div className="model-meta">
                    <h3>DeepSeek-V4-Pro</h3>
                    <p>Acoustic-Prompt Forensic</p>
                  </div>
                  <div>
                    {results.deepseek?.error ? (
                      <span className="verdict-badge verdict-fake" style={{ background: 'rgba(255,51,102,0.05)', color: '#ff3366' }}>Error</span>
                    ) : (
                      <span className={`verdict-badge ${results.deepseek?.verdict === 'Fake' ? 'verdict-fake' : 'verdict-real'}`}>
                        {results.deepseek?.verdict} ({Math.round(results.deepseek?.confidence * 100)}%)
                      </span>
                    )}
                  </div>
                  <p className="reasoning-text">
                    {results.deepseek?.error || results.deepseek?.reasoning} (Latency: {results.deepseek?.latency?.toFixed(2)}s)
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Local Acoustic Stats Panel */}
          {acousticStats && !acousticStats.error && (
            <div className="card">
              <h2 className="card-title">📊 Local Acoustic Statistics</h2>
              <div className="stat-grid">
                <div className="stat-item">
                  <div className="stat-value">{acousticStats.durationSec}s</div>
                  <div className="stat-label">Duration</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value">{acousticStats.sampleRate} Hz</div>
                  <div className="stat-label">Sample Rate</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value">{acousticStats.rms}</div>
                  <div className="stat-label">RMS Energy</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value">{acousticStats.maxAmplitude}</div>
                  <div className="stat-label">Max Amp</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value">{acousticStats.zeroCrossingRate}</div>
                  <div className="stat-label">ZCR Rate</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
