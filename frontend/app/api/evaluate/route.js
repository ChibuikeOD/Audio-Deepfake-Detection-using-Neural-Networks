import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

// Parse basic WAV metrics in JS for DeepSeek reasoning fallback
function parseWavStats(input) {
  try {
    const buffer = Buffer.isBuffer(input)
      ? input.buffer.slice(input.byteOffset, input.byteOffset + input.byteLength)
      : input;
    const view = new DataView(buffer);
    
    // Check RIFF header
    const isRiff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3)) === 'RIFF';
    if (!isRiff) return { error: 'Acoustic stats are only available for WAV/RIFF audio in this evaluator.' };
    
    const sampleRate = view.getUint32(24, true);
    const numChannels = view.getUint16(22, true);
    const bitsPerSample = view.getUint16(34, true);
    
    // Find data chunk
    let offset = 12;
    let dataOffset = -1;
    let dataLength = 0;
    
    while (offset < buffer.byteLength - 8) {
      const chunkId = String.fromCharCode(
        view.getUint8(offset),
        view.getUint8(offset + 1),
        view.getUint8(offset + 2),
        view.getUint8(offset + 3)
      );
      const chunkSize = view.getUint32(offset + 4, true);
      
      if (chunkId === 'data') {
        dataOffset = offset + 8;
        dataLength = chunkSize;
        break;
      }
      offset += 8 + chunkSize;
    }
    
    if (dataOffset === -1) {
      dataOffset = 44; // default header end
      dataLength = buffer.byteLength - 44;
    }
    
    const duration = dataLength / (sampleRate * numChannels * (bitsPerSample / 8));
    
    // Compute simple statistics from a sample window (max 30,000 samples to keep it fast)
    const bytesPerFrame = bitsPerSample / 8;
    const totalFrames = Math.floor(dataLength / (bytesPerFrame * numChannels));
    const step = Math.max(1, Math.floor(totalFrames / 30000));
    
    let sum = 0;
    let squareSum = 0;
    let maxVal = 0;
    let crossCount = 0;
    let lastVal = 0;
    let samplesCount = 0;
    
    for (let i = 0; i < totalFrames; i += step) {
      const byteIndex = dataOffset + (i * bytesPerFrame * numChannels);
      if (byteIndex >= buffer.byteLength) break;
      
      let val = 0;
      if (bitsPerSample === 16) {
        val = view.getInt16(byteIndex, true) / 32768.0;
      } else if (bitsPerSample === 8) {
        val = (view.getUint8(byteIndex) - 128) / 128.0;
      }
      
      sum += val;
      squareSum += val * val;
      maxVal = Math.max(maxVal, Math.abs(val));
      
      // Count sign crossings for zero crossing rate
      if (i > 0) {
        if ((val > 0 && lastVal <= 0) || (val < 0 && lastVal >= 0)) {
          crossCount++;
        }
      }
      lastVal = val;
      samplesCount++;
    }
    
    const mean = sum / (samplesCount || 1);
    const rms = Math.sqrt(squareSum / (samplesCount || 1));
    const zcr = crossCount / (duration || 1);
    
    return {
      durationSec: Math.round(duration * 100) / 100,
      sampleRate: sampleRate,
      rms: Math.round(rms * 1000) / 1000,
      maxAmplitude: Math.round(maxVal * 1000) / 1000,
      meanAmplitude: Math.round(mean * 10000) / 10000,
      zeroCrossingRate: Math.round(zcr * 100) / 100
    };
  } catch (e) {
    return { error: 'Failed to parse stats: ' + e.message };
  }
}

// Helpers to parse LLM structured reports
function parseVerdict(text) {
  let verdict = 'Unknown';
  let confidence = '0.70';
  let reasoning = text;
  
  if (!text) return { verdict, confidence, reasoning };
  
  const lines = text.split('\n');
  for (const line of lines) {
    const clean = line.trim();
    if (clean.toUpperCase().startsWith('VERDICT:')) {
      verdict = clean.split(':', 2)[1].replace(/[\[\]]/g, '').trim();
    } else if (clean.toUpperCase().startsWith('CONFIDENCE:')) {
      confidence = clean.split(':', 2)[1].replace(/[\[\]]/g, '').trim();
    } else if (clean.toUpperCase().startsWith('REASONING:')) {
      reasoning = clean.split(':', 2)[1].trim();
    }
  }
  
  if (reasoning === text && text.includes('REASONING:')) {
    reasoning = text.split('REASONING:', 2)[1].trim();
  }
  
  return { verdict, confidence, reasoning };
}

// =====================================================================
// Evaluator Implementations
// =====================================================================

async function runGemini(audioBase64, key) {
  if (!key) return { error: 'No Gemini API key provided. Enter one in the UI or set GEMINI_API_KEY in Vercel.', latency: 0 };
  
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${key}`;
  const payload = {
    contents: [
      {
        parts: [
          {
            text: (
              "Analyze this speech audio file carefully. Does it sound like a real human voice " +
              "(bonafide) or a synthetic/AI-generated (deepfake/spoof) voice? " +
              "Provide your response in this exact format:\n" +
              "VERDICT: [Real or Fake]\n" +
              "CONFIDENCE: [0.0 to 1.0]\n" +
              "REASONING: [Explain details like vocoder noise, metallic tone, consistency, etc.]"
            )
          },
          {
            inlineData: {
              mimeType: "audio/wav",
              data: audioBase64
            }
          }
        ]
      }
    ]
  };
  
  const start = Date.now();
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const latency = (Date.now() - start) / 1000;
  
  if (res.ok) {
    const data = await res.json();
    const text = data.candidates[0].content.parts[0].text;
    const parsed = parseVerdict(text);
    return { ...parsed, latency };
  } else {
    const text = await res.text();
    return { error: `Gemini API Error: ${res.status} - ${text}`, latency };
  }
}

async function runOpenAI(audioBase64, key) {
  if (!key) return { error: 'No OpenAI API key provided. Enter one in the UI or set OPENAI_API_KEY in Vercel.', latency: 0 };
  
  const url = "https://api.openai.com/v1/chat/completions";
  const payload = {
    model: "gpt-4o-audio-preview",
    modalities: ["text"],
    audio: {
      voice: "alloy",
      format: "wav"
    },
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: (
              "Analyze this speech audio file carefully. Does it sound like a real human voice " +
              "(bonafide) or a synthetic/AI-generated (deepfake/spoof) voice? " +
              "Provide your response in this exact format:\n" +
              "VERDICT: [Real or Fake]\n" +
              "CONFIDENCE: [0.0 to 1.0]\n" +
              "REASONING: [Explain your reasoning details.]"
            )
          },
          {
            type: "input_audio",
            input_audio: {
              data: audioBase64,
              format: "wav"
            }
          }
        ]
      }
    ]
  };
  
  const start = Date.now();
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${key}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });
  const latency = (Date.now() - start) / 1000;
  
  if (res.ok) {
    const data = await res.json();
    const text = data.choices[0].message.content;
    const parsed = parseVerdict(text);
    return { ...parsed, latency };
  } else {
    const text = await res.text();
    return { error: `OpenAI API Error: ${res.status} - ${text}`, latency };
  }
}

async function runDeepSeek(audioFile, stats, deepseekKey, openaiKey) {
  if (!deepseekKey) return { error: 'No DeepSeek API key provided. Enter one in the UI or set DEEPSEEK_API_KEY in Vercel.', latency: 0 };
  
  // Get Whisper Transcription
  let transcript = '[No transcription obtained]';
  if (openaiKey) {
    try {
      const whisperUrl = "https://api.openai.com/v1/audio/transcriptions";
      const whisperForm = new FormData();
      whisperForm.append('file', audioFile, 'audio.wav');
      whisperForm.append('model', 'whisper-1');
      
      const transRes = await fetch(whisperUrl, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${openaiKey}` },
        body: whisperForm
      });
      if (transRes.ok) {
        const transData = await transRes.json();
        transcript = transData.text;
      }
    } catch (e) {
      transcript = `[Transcription Failed: ${e.message}]`;
    }
  }
  
  const url = "https://api.deepseek.com/chat/completions";
  const prompt = (
    `You are a speech forensics expert. You are investigating if the audio recording with the following ` +
    `features is a Real Human voice (bonafide) or a Synthetic/Deepfake spoof:\n\n` +
    `--- Audio Transcription ---\n` +
    `"${transcript}"\n\n` +
    `--- Local Acoustic Statistics ---\n` +
    `Duration: ${stats.durationSec ?? 'N/A'} seconds\n` +
    `Sample Rate: ${stats.sampleRate ?? 'N/A'} Hz\n` +
    `RMS Energy: ${stats.rms ?? 'N/A'}\n` +
    `Maximum Amplitude: ${stats.maxAmplitude ?? 'N/A'}\n` +
    `Zero Crossing Rate: ${stats.zeroCrossingRate ?? 'N/A'} crossings/sec\n\n` +
    `Analyze if these features indicate deepfake traits (e.g. synthetic speech often exhibits ` +
    `abnormally flat amplitude variance, robotic zero crossing stability, or phase vocoder anomalies). ` +
    `Provide your response in this exact format:\n` +
    `VERDICT: [Real or Fake]\n` +
    `CONFIDENCE: [0.0 to 1.0]\n` +
    `REASONING: [Explain your reasoning based on transcript and acoustic metrics.]`
  );
  
  const payload = {
    model: "deepseek-v4-pro",
    messages: [
      { role: "system", content: "You are a helpful and expert assistant in audio forensics." },
      { role: "user", content: prompt }
    ]
  };
  
  const start = Date.now();
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${deepseekKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });
  const latency = (Date.now() - start) / 1000;
  
  if (res.ok) {
    const data = await res.json();
    const text = data.choices[0].message.content;
    const parsed = parseVerdict(text);
    return { ...parsed, latency };
  } else {
    const text = await res.text();
    return { error: `DeepSeek API Error: ${res.status} - ${text}`, latency };
  }
}

async function runMiniMax(audioBase64, key) {
  if (!key) return { error: 'No MiniMax API key provided. Enter one in the UI or set MINIMAX_API_KEY in Vercel.', latency: 0 };
  
  const url = "https://api.minimax.io/v1/chat/completions";
  const payload = {
    model: "MiniMax-M3",
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: (
              "Analyze this speech audio file carefully. Does it sound like a real human voice " +
              "(bonafide) or a synthetic/AI-generated (deepfake/spoof) voice? " +
              "Provide your response in this exact format:\n" +
              "VERDICT: [Real or Fake]\n" +
              "CONFIDENCE: [0.0 to 1.0]\n" +
              "REASONING: [Explain your forensic details.]"
            )
          },
          {
            type: "media",
            media: {
              type: "audio",
              base64: audioBase64
            }
          }
        ]
      }
    ]
  };
  
  const start = Date.now();
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${key}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });
  const latency = (Date.now() - start) / 1000;
  
  if (res.ok) {
    const data = await res.json();
    const text = data.choices[0].message.content;
    const parsed = parseVerdict(text);
    return { ...parsed, latency };
  } else {
    const text = await res.text();
    return { error: `MiniMax API Error: ${res.status} - ${text}`, latency };
  }
}

async function runTensorFlow(audioFile, backendUrl) {
  if (!backendUrl) return { error: 'RENDER_BACKEND_URL is not set on frontend server.' };
  
  const url = `${backendUrl}/predict`;
  const form = new FormData();
  form.append('file', audioFile, 'audio.wav');
  
  const start = Date.now();
  const res = await fetch(url, {
    method: 'POST',
    body: form
  });
  const latency = (Date.now() - start) / 1000;
  
  if (res.ok) {
    const data = await res.json();
    // Map response structure to unified standard
    const verdict = data.verdict === 'Spoof/Fake' ? 'Fake' : 'Real';
    return {
      verdict,
      confidence: data.confidence.toFixed(2),
      reasoning: `Classifier returned spoof probability: ${Math.round(data.spoof_probability * 100)}%. Model loaded from Render.`,
      latency
    };
  } else {
    const text = await res.text();
    return { error: `TensorFlow Backend Error: ${res.status} - ${text}`, latency };
  }
}

// =====================================================================
// Next.js Route POST Handler
// =====================================================================
export async function POST(req) {
  try {
    const formData = await req.formData();
    const file = formData.get('file');
    const modelSelection = formData.get('models'); // comma-separated values, e.g. "tensorflow,gemini"
    
    if (!file) {
      return NextResponse.json({ error: 'No audio file provided' }, { status: 400 });
    }
    
    const buffer = Buffer.from(await file.arrayBuffer());
    const audioBase64 = buffer.toString('base64');
    
    // Parse acoustic features locally for DeepSeek
    const stats = parseWavStats(buffer);
    
    // Prefer keys submitted from the UI, with server env vars as fallback.
    const geminiKey = formData.get('key_gemini') || process.env.GEMINI_API_KEY;
    const openaiKey = formData.get('key_openai') || process.env.OPENAI_API_KEY;
    const deepseekKey = formData.get('key_deepseek') || process.env.DEEPSEEK_API_KEY;
    const minimaxKey = formData.get('key_minimax') || process.env.MINIMAX_API_KEY;
    const renderUrl = process.env.RENDER_BACKEND_URL || (process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000');
    
    const modelsToRun = modelSelection ? modelSelection.split(',') : ['tensorflow', 'gemini', 'openai', 'deepseek', 'minimax'];
    const results = {};
    
    const tasks = modelsToRun.map(async (model) => {
      try {
        if (model === 'tensorflow') {
          results.tensorflow = await runTensorFlow(file, renderUrl);
        } else if (model === 'gemini') {
          results.gemini = await runGemini(audioBase64, geminiKey);
        } else if (model === 'openai') {
          results.openai = await runOpenAI(audioBase64, openaiKey);
        } else if (model === 'deepseek') {
          results.deepseek = await runDeepSeek(file, stats, deepseekKey, openaiKey);
        } else if (model === 'minimax') {
          results.minimax = await runMiniMax(audioBase64, minimaxKey);
        }
      } catch (err) {
        results[model] = { error: err.message, latency: 0 };
      }
    });
    
    await Promise.all(tasks);
    
    return NextResponse.json({
      success: true,
      results,
      acousticStats: stats
    });
    
  } catch (err) {
    return NextResponse.json({ error: 'Evaluation failed: ' + err.message }, { status: 500 });
  }
}
