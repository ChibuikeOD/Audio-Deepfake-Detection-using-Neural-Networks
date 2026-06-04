import os
import sys
import base64
import time
import argparse
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import requests

# Set terminal color codes
class Color:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

# =====================================================================
# 1. Audio Preprocessing & Feature Extraction
# =====================================================================

def convert_m4a_to_wav(m4a_path, wav_path="temp_converted.wav"):
    """
    Converts .m4a audio file to 16kHz mono .wav format using pydub.
    """
    try:
        audio = AudioSegment.from_file(m4a_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print(f"{Color.RED}Failed to convert M4A to WAV: {e}{Color.END}")
        return None

def extract_acoustic_features(wav_path):
    """
    Extracts basic acoustic statistical features from a WAV file locally.
    """
    try:
        sample_rate, data = wavfile.read(wav_path)
        
        # Normalize data to float32 between -1.0 and 1.0 if it is integer PCM
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
            
        duration = len(data) / sample_rate
        
        # Basic amplitude stats
        mean_amp = np.mean(data)
        std_amp = np.std(data)
        max_amp = np.max(data)
        min_amp = np.min(data)
        rms = np.sqrt(np.mean(data**2))
        
        # Zero Crossing Rate (ZCR)
        zero_crossings = np.nonzero(np.diff(data > 0))[0]
        zcr = len(zero_crossings) / duration
        
        # Spectral Centroid (Approximated using FFT)
        fft_vals = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(len(data), 1/sample_rate)
        spectral_centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-10)
        
        return {
            "sample_rate": sample_rate,
            "duration_sec": round(duration, 2),
            "max_amplitude": round(float(max_amp), 4),
            "min_amplitude": round(float(min_amp), 4),
            "mean_amplitude": round(float(mean_amp), 4),
            "amplitude_std": round(float(std_amp), 4),
            "rms_energy": round(float(rms), 4),
            "zero_crossing_rate": round(float(zcr), 2),
            "approx_spectral_centroid_hz": round(float(spectral_centroid), 2)
        }
    except Exception as e:
        print(f"{Color.YELLOW}Could not extract local acoustic features: {e}{Color.END}")
        return {}

def transcribe_audio(wav_path, openai_api_key):
    """
    Transcribes audio using OpenAI's Whisper API.
    """
    if not openai_api_key:
        return "[No OpenAI API key provided for transcription]"
        
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {openai_api_key}"}
    
    try:
        with open(wav_path, "rb") as f:
            files = {
                "file": (os.path.basename(wav_path), f, "audio/wav"),
                "model": (None, "whisper-1")
            }
            response = requests.post(url, headers=headers, files=files)
            if response.status_code == 200:
                return response.json().get("text", "")
            else:
                return f"[Transcription API Error: {response.status_code} - {response.text}]"
    except Exception as e:
        return f"[Transcription Failed: {e}]"

# =====================================================================
# 2. LLM Evaluator Models
# =====================================================================

class LLMEvaluator:
    def __init__(self):
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        self.minimax_key = os.environ.get("MINIMAX_API_KEY")

    def evaluate_gemini(self, wav_path):
        """
        Evaluates audio using Gemini 1.5 Flash API (with native audio upload).
        """
        if not self.gemini_key:
            return {"error": "GEMINI_API_KEY environment variable not set."}
            
        print(f"{Color.CYAN}Sending to Gemini 1.5 Flash...{Color.END}")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
        
        try:
            with open(wav_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")
                
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": (
                                    "Analyze this speech audio file carefully. Does it sound like a real human voice "
                                    "(bonafide) or a synthetic/AI-generated (deepfake/spoof) voice? "
                                    "Provide your response in this exact format:\n"
                                    "VERDICT: [Real or Fake]\n"
                                    "CONFIDENCE: [0.0 to 1.0]\n"
                                    "REASONING: [Explain details like vocoder noise, metallic tone, consistency, etc.]"
                                )
                            },
                            {
                                "inlineData": {
                                    "mimeType": "audio/wav",
                                    "data": audio_base64
                                }
                            }
                        ]
                    }
                ]
            }
            
            headers = {"Content-Type": "application/json"}
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                res_json = response.json()
                text_response = res_json["candidates"][0]["content"]["parts"][0]["text"]
                return {"text": text_response, "latency": latency}
            else:
                return {"error": f"API Error {response.status_code}: {response.text}", "latency": latency}
        except Exception as e:
            return {"error": str(e), "latency": 0}

    def evaluate_openai(self, wav_path):
        """
        Evaluates audio using OpenAI's gpt-4o-audio-preview API.
        """
        if not self.openai_key:
            return {"error": "OPENAI_API_KEY environment variable not set."}
            
        print(f"{Color.CYAN}Sending to OpenAI GPT-4o (Audio)...{Color.END}")
        url = "https://api.openai.com/v1/chat/completions"
        
        try:
            with open(wav_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")
                
            payload = {
                "model": "gpt-4o-audio-preview",
                "modalities": ["text"],
                "audio": {
                    "voice": "alloy",
                    "format": "wav"
                },
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this speech audio file carefully. Does it sound like a real human voice "
                                    "(bonafide) or a synthetic/AI-generated (deepfake/spoof) voice? "
                                    "Provide your response in this exact format:\n"
                                    "VERDICT: [Real or Fake]\n"
                                    "CONFIDENCE: [0.0 to 1.0]\n"
                                    "REASONING: [Explain your reasoning details.]"
                                )
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_base64,
                                    "format": "wav"
                                }
                            }
                        ]
                    }
                ]
            }
            
            headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json"
            }
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                res_json = response.json()
                text_response = res_json["choices"][0]["message"]["content"]
                return {"text": text_response, "latency": latency}
            else:
                return {"error": f"API Error {response.status_code}: {response.text}", "latency": latency}
        except Exception as e:
            return {"error": str(e), "latency": 0}

    def evaluate_deepseek(self, wav_path):
        """
        Evaluates audio using DeepSeek-V4-Pro by transcribing it first
        and providing local acoustic features.
        """
        if not self.deepseek_key:
            return {"error": "DEEPSEEK_API_KEY environment variable not set."}
            
        print(f"{Color.CYAN}Extracting features and transcribing for DeepSeek...{Color.END}")
        
        # Get transcription (Requires OpenAI API key or we show fallback text)
        transcript = transcribe_audio(wav_path, self.openai_key)
        
        # Get local acoustic stats
        stats = extract_acoustic_features(wav_path)
        
        print(f"{Color.CYAN}Sending text + acoustic prompt to DeepSeek-V4-Pro...{Color.END}")
        url = "https://api.deepseek.com/chat/completions"
        
        prompt = (
            f"You are a speech forensics expert. You are investigating if the audio recording with the following "
            f"features is a Real Human voice (bonafide) or a Synthetic/Deepfake spoof:\n\n"
            f"--- Audio Transcription ---\n"
            f"\"{transcript}\"\n\n"
            f"--- Local Acoustic Statistics ---\n"
            f"Duration: {stats.get('duration_sec', 'N/A')} seconds\n"
            f"Sample Rate: {stats.get('sample_rate', 'N/A')} Hz\n"
            f"RMS Energy: {stats.get('rms_energy', 'N/A')}\n"
            f"Amplitude Standard Deviation: {stats.get('amplitude_std', 'N/A')}\n"
            f"Zero Crossing Rate: {stats.get('zero_crossing_rate', 'N/A')} crossings/sec\n"
            f"Approximate Spectral Centroid: {stats.get('approx_spectral_centroid_hz', 'N/A')} Hz\n\n"
            f"Analyze if these features indicate deepfake traits (e.g. synthetic speech often exhibits "
            f"abnormally flat amplitude variance, robotic zero crossing stability, or phase vocoder anomalies). "
            f"Provide your response in this exact format:\n"
            f"VERDICT: [Real or Fake]\n"
            f"CONFIDENCE: [0.0 to 1.0]\n"
            f"REASONING: [Explain your reasoning based on transcript and acoustic metrics.]"
        )
        
        payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful and expert assistant in audio forensics."
                },
                {"role": "user", "content": prompt}
            ]
        }
        
        headers = {
            "Authorization": f"Bearer {self.deepseek_key}",
            "Content-Type": "application/json"
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                res_json = response.json()
                text_response = res_json["choices"][0]["message"]["content"]
                return {"text": text_response, "latency": latency}
            else:
                return {"error": f"API Error {response.status_code}: {response.text}", "latency": latency}
        except Exception as e:
            return {"error": str(e), "latency": 0}

    def evaluate_minimax(self, wav_path):
        """
        Evaluates audio using MiniMax M3 API.
        """
        if not self.minimax_key:
            return {"error": "MINIMAX_API_KEY environment variable not set."}
            
        print(f"{Color.CYAN}Sending to MiniMax M3...{Color.END}")
        url = "https://api.minimax.io/v1/chat/completions"
        
        try:
            with open(wav_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")
                
            payload = {
                "model": "MiniMax-M3",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this speech audio file carefully. Does it sound like a real human voice "
                                    "(bonafide) or a synthetic/AI-generated (deepfake/spoof) voice? "
                                    "Provide your response in this exact format:\n"
                                    "VERDICT: [Real or Fake]\n"
                                    "CONFIDENCE: [0.0 to 1.0]\n"
                                    "REASONING: [Explain your forensic details.]"
                                )
                            },
                            {
                                "type": "media",
                                "media": {
                                    "type": "audio",
                                    "base64": audio_base64
                                }
                            }
                        ]
                    }
                ]
            }
            
            headers = {
                "Authorization": f"Bearer {self.minimax_key}",
                "Content-Type": "application/json"
            }
            
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                res_json = response.json()
                text_response = res_json["choices"][0]["message"]["content"]
                return {"text": text_response, "latency": latency}
            else:
                return {"error": f"API Error {response.status_code}: {response.text}", "latency": latency}
        except Exception as e:
            return {"error": str(e), "latency": 0}

# =====================================================================
# 3. Main Runner
# =====================================================================

def parse_result(result_text):
    """
    Parses the structured response from LLMs.
    """
    verdict = "Unknown"
    confidence = "N/A"
    reasoning = "N/A"
    
    if not isinstance(result_text, str):
        return verdict, confidence, reasoning
        
    for line in result_text.split("\n"):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip()
        elif line.upper().startswith("CONFIDENCE:"):
            confidence = line.split(":", 1)[1].strip()
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
            
    # Fallback to general reasoning search if reasoning isn't single line
    if reasoning == "N/A" and "REASONING:" in result_text:
        reasoning = result_text.split("REASONING:", 1)[1].strip()
        
    return verdict, confidence, reasoning

def print_report(model_name, response_dict):
    """
    Prints a beautiful summary report of the model results.
    """
    print(f"\n{Color.BOLD}{Color.BLUE}=================================================={Color.END}")
    print(f"{Color.BOLD}{Color.BLUE} REPORT: {model_name.upper()} EVALUATION{Color.END}")
    print(f"{Color.BOLD}{Color.BLUE}=================================================={Color.END}")
    
    if "error" in response_dict:
        print(f"{Color.RED}Error: {response_dict['error']}{Color.END}")
        print(f"Latency: {response_dict.get('latency', 0):.2f}s")
        return
        
    raw_text = response_dict.get("text", "")
    verdict, confidence, reasoning = parse_result(raw_text)
    
    # Color code the verdict
    verdict_color = Color.GREEN if "REAL" in verdict.upper() else Color.RED
    
    print(f"Verdict:   {verdict_color}{Color.BOLD}{verdict}{Color.END}")
    print(f"Confidence: {Color.YELLOW}{confidence}{Color.END}")
    print(f"Latency:    {response_dict.get('latency', 0):.2f}s")
    print(f"\nReasoning:\n{reasoning}")
    print(f"{Color.BLUE}=================================================={Color.END}\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate audio files for deepfake detection using proprietary and open LLMs.")
    parser.add_argument("--file", required=True, help="Path to the audio file (.wav or .m4a)")
    parser.add_argument("--model", required=True, choices=["gemini", "openai", "deepseek", "minimax", "all"], help="LLM model to run")
    
    args = parser.parse_args()
    
    audio_path = args.file
    if not os.path.exists(audio_path):
        print(f"{Color.RED}Error: Audio file '{audio_path}' not found.{Color.END}")
        sys.exit(1)
        
    # Check if M4A needs conversion
    temp_wav = None
    if audio_path.lower().endswith(".m4a"):
        print(f"{Color.YELLOW}Detected M4A file. Converting to 16kHz Mono WAV...{Color.END}")
        temp_wav = convert_m4a_to_wav(audio_path)
        if not temp_wav:
            sys.exit(1)
        audio_to_evaluate = temp_wav
    else:
        audio_to_evaluate = audio_path
        
    evaluator = LLMEvaluator()
    
    try:
        if args.model == "gemini" or args.model == "all":
            res = evaluator.evaluate_gemini(audio_to_evaluate)
            print_report("Gemini 1.5 Flash", res)
            
        if args.model == "openai" or args.model == "all":
            res = evaluator.evaluate_openai(audio_to_evaluate)
            print_report("OpenAI GPT-4o Audio", res)
            
        if args.model == "deepseek" or args.model == "all":
            res = evaluator.evaluate_deepseek(audio_to_evaluate)
            print_report("DeepSeek-V4-Pro Forensic Reasoning", res)
            
        if args.model == "minimax" or args.model == "all":
            res = evaluator.evaluate_minimax(audio_to_evaluate)
            print_report("MiniMax M3", res)
            
    finally:
        # Cleanup temporary WAV file if created
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)

if __name__ == "__main__":
    main()
