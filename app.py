import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import shutil
import tempfile
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from scipy.io import wavfile
from pydub import AudioSegment
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFBertModel

app = FastAPI(title="Audio Deepfake Detection TF API", version="1.0")

# Enable CORS for Next.js frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model pointers
yamnet_model = None
tokenizer = None
bert_model = None
fusion_model = None
load_errors = {}

@app.on_event("startup")
def load_models():
    global yamnet_model, tokenizer, bert_model, fusion_model, load_errors
    load_errors = {}
    print("Loading YAMNet, BERT, and Keras Fusion models...")
    
    # Load YAMNet
    try:
        yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    except Exception as e:
        load_errors["yamnet"] = str(e)
        print(f"Error loading YAMNet: {e}")
        
    # Load BERT
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    except Exception as e:
        load_errors["bert"] = str(e)
        print(f"Error loading BERT: {e}")
        
    # Load Keras model
    model_file = "fusion_model_fitted.keras"
    if not os.path.exists(model_file):
        model_file = "fusion_model.keras"
        
    if os.path.exists(model_file):
        print(f"Loading Keras Fusion Model from '{model_file}'...")
        try:
            fusion_model = tf.keras.models.load_model(model_file)
        except Exception as e:
            load_errors["keras_model"] = str(e)
            print(f"Error loading Keras model: {e}")
    else:
        load_errors["keras_model"] = "No Keras model found."
        print("Warning: No Keras model found. /predict will fail.")

# =====================================================================
# Feature Extraction Helpers
# =====================================================================
def extract_yamnet_embedding(wav_path):
    sample_rate, wav_data = wavfile.read(wav_path)
    
    # Convert to mono float32 normalized
    if wav_data.dtype == np.int16:
        wav_data = wav_data.astype(np.float32) / 32768.0
    elif wav_data.dtype == np.int32:
        wav_data = wav_data.astype(np.float32) / 2147483648.0
        
    scores, embeddings, spectrogram = yamnet_model(wav_data)
    mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
    return mean_embedding

def extract_bert_embedding(question):
    tokens = tokenizer(question, return_tensors="tf", truncation=True, padding=True, max_length=32)
    outputs = bert_model(**tokens)
    return outputs.pooler_output[0].numpy()

# =====================================================================
# API Endpoints
# =====================================================================
@app.get("/health")
def health_check():
    status = {
        "status": "healthy",
        "yamnet_loaded": yamnet_model is not None,
        "bert_loaded": bert_model is not None,
        "keras_model_loaded": fusion_model is not None,
        "load_errors": load_errors
    }
    return status

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not fusion_model:
        raise HTTPException(status_code=503, detail="Keras classification model not loaded on server.")
    if not yamnet_model or not bert_model:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Feature extraction models (YAMNet/BERT) not loaded.",
                "yamnet_loaded": yamnet_model is not None,
                "bert_loaded": bert_model is not None,
                "load_errors": load_errors,
            },
        )

    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in [".wav", ".m4a", ".mp3", ".flac"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format. Use WAV, M4A, MP3, or FLAC.")

    # Save uploaded file to temp file
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, file.filename)
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Convert to WAV if not already WAV
        wav_path = os.path.join(temp_dir, "converted.wav")
        if suffix != ".wav":
            try:
                audio = AudioSegment.from_file(input_path)
                audio = audio.set_channels(1)
                audio = audio.set_frame_rate(16000)
                audio.export(wav_path, format="wav")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Audio conversion failed: {str(e)}")
        else:
            wav_path = input_path
            
        # Extract features
        try:
            audio_emb = extract_yamnet_embedding(wav_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Acoustic feature extraction failed: {str(e)}")
            
        question = "Is this audio real human speech or spoofed synthetic speech?"
        text_emb = extract_bert_embedding(question)
        
        # Reshape inputs
        X_audio = np.expand_dims(audio_emb, axis=0)
        X_text = np.expand_dims(text_emb, axis=0)
        
        # Predict
        pred = fusion_model.predict({"audio_input": X_audio, "text_input": X_text})[0][0]
        score = float(pred)
        verdict = "Spoof/Fake" if score > 0.5 else "Bonafide/Real"
        
        return {
            "verdict": verdict,
            "spoof_probability": score,
            "confidence": score if score > 0.5 else (1.0 - score)
        }
        
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    # Render maps port dynamically
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
