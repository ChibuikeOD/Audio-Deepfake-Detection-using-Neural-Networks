import os
import sys
import numpy as np
from scipy.io import wavfile
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFBertModel

# Disable TensorFlow warnings/info logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =====================================================================
# 1. Generate Synthetic Audio
# =====================================================================
def generate_synthetic_audio(filename="synthetic_audio.wav", duration=3.0, sample_rate=16000):
    """
    Generates a synthetic audio file (sine wave beep with some white noise).
    """
    print(f"Generating synthetic audio: {filename}...")
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 440 Hz sine wave (standard A4 note) + some 880 Hz harmonic
    signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)
    
    # Add a bit of white noise
    noise = np.random.normal(0, 0.05, signal.shape)
    signal = signal + noise
    
    # Clip to ensure valid range
    signal = np.clip(signal, -1.0, 1.0)
    
    # Convert to 16-bit PCM WAV
    signal_int16 = (signal * 32767).astype(np.int16)
    wavfile.write(filename, sample_rate, signal_int16)
    print("Synthetic audio generated successfully.")
    return filename

# =====================================================================
# 2. Extract Embeddings (YAMNet + BERT)
# =====================================================================
def extract_yamnet_embedding(wav_path, yamnet_model):
    """
    Loads wav and extracts 1024-D YAMNet embedding.
    """
    sample_rate, wav_data = wavfile.read(wav_path)
    
    # Convert to mono float32 normalized
    if wav_data.dtype == np.int16:
        wav_data = wav_data.astype(np.float32) / 32768.0
    elif wav_data.dtype == np.int32:
        wav_data = wav_data.astype(np.float32) / 2147483648.0
        
    # YAMNet expects float32 mono array
    scores, embeddings, spectrogram = yamnet_model(wav_data)
    
    # Mean pooling over time frames
    mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
    return mean_embedding

def extract_bert_embedding(question, tokenizer, bert_model):
    """
    Extracts 768-D pooler output BERT embedding.
    """
    tokens = tokenizer(question, return_tensors="tf", truncation=True, padding=True, max_length=32)
    outputs = bert_model(**tokens)
    return outputs.pooler_output[0].numpy()

# =====================================================================
# 3. Main Run & Prediction
# =====================================================================
def run_evaluation():
    # 1. Generate Synthetic Audio
    synth_file = generate_synthetic_audio()
    
    # 2. Initialize Models
    print("\nLoading pre-trained YAMNet and BERT models...")
    try:
        yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert = TFBertModel.from_pretrained("bert-base-uncased", from_pt=True, use_safetensors=True)
    except Exception as e:
        print(f"Error loading models: {e}")
        return
        
    # 3. Load Saved Keras Model
    model_file = "fusion_model_fitted.keras"
    if not os.path.exists(model_file):
        model_file = "fusion_model.keras"
        
    if not os.path.exists(model_file):
        print(f"Error: Neither 'fusion_model_fitted.keras' nor 'fusion_model.keras' found in the workspace.")
        return
        
    print(f"Loading Keras Fusion Model from '{model_file}'...")
    try:
        model = tf.keras.models.load_model(model_file)
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return

    # 4. Define Prompt Question
    question = "Is this audio real human speech or spoofed synthetic speech?"
    print(f"Extracting BERT embedding for prompt: '{question}'...")
    text_emb = extract_bert_embedding(question, tokenizer, bert)
    
    # 5. Predict on Synthetic Audio (which should be "Spoof")
    print(f"Extracting YAMNet embedding for: {synth_file}...")
    synth_audio_emb = extract_yamnet_embedding(synth_file, yamnet)
    
    # Reshape for Keras input (batch_size=1)
    X_audio_synth = np.expand_dims(synth_audio_emb, axis=0)
    X_text_synth = np.expand_dims(text_emb, axis=0)
    
    print("\nRunning prediction on synthetic audio...")
    pred_synth = model.predict({"audio_input": X_audio_synth, "text_input": X_text_synth})[0][0]
    verdict_synth = "Spoof/Fake" if pred_synth > 0.5 else "Bonafide/Real"
    
    print(f"\nRESULTS FOR SYNTHETIC AUDIO ({synth_file}):")
    print(f"  Raw Score (Probability of Spoof): {pred_synth:.6f}")
    print(f"  Model Verdict:                    {verdict_synth}")
    print(f"  Correct Prediction?               {'YES (Synthetic wave detected as spoof)' if verdict_synth == 'Spoof/Fake' else 'NO'}")

    # 6. Predict on User's Recording (Recording.wav) if it exists
    real_file = "Recording.wav"
    if os.path.exists(real_file):
        print(f"\nFound local recording: {real_file}. Running evaluation...")
        try:
            real_audio_emb = extract_yamnet_embedding(real_file, yamnet)
            X_audio_real = np.expand_dims(real_audio_emb, axis=0)
            
            pred_real = model.predict({"audio_input": X_audio_real, "text_input": X_text_synth})[0][0]
            verdict_real = "Spoof/Fake" if pred_real > 0.5 else "Bonafide/Real"
            
            print(f"\nRESULTS FOR RECORDING AUDIO ({real_file}):")
            print(f"  Raw Score (Probability of Spoof): {pred_real:.6f}")
            print(f"  Model Verdict:                    {verdict_real}")
        except Exception as e:
            print(f"Could not run prediction on {real_file}: {e}")

if __name__ == "__main__":
    run_evaluation()
