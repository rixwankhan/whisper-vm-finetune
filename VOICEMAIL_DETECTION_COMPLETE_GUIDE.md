# Complete Guide to Voicemail Detection Systems

This comprehensive guide covers two voicemail detection (AMD - Answering Machine Detection) systems:

1. **Whisper-VM-Finetune** - ASR + Text Classification approach
2. **Wav2Vec-VM-Finetune** - End-to-end audio classification approach

Both are designed for integration with VoIP systems like FreeSWITCH, Asterisk, and WebRTC gateways.

---

## Table of Contents

1. [Overview & Architecture Comparison](#overview--architecture-comparison)
2. [System Requirements](#system-requirements)
3. [Data Preparation Guide](#data-preparation-guide)
4. [Whisper-VM-Finetune Complete Guide](#whisper-vm-finetune-complete-guide)
5. [Wav2Vec-VM-Finetune Complete Guide](#wav2vec-vm-finetune-complete-guide)
6. [FreeSWITCH Integration](#freeswitch-integration)
7. [Accuracy Enhancement Strategies](#accuracy-enhancement-strategies)
8. [Production Deployment](#production-deployment)
9. [Troubleshooting](#troubleshooting)

---

## Overview & Architecture Comparison

### Whisper-VM-Finetune

**Architecture:** Two-stage pipeline
1. **Stage 1 - ASR:** Fine-tuned Whisper-small converts audio to text
2. **Stage 2 - Classification:** TF-IDF + Logistic Regression classifies text as human/machine

**Pros:**
- Explainable decisions (you can see the transcript)
- Smaller final classifier model
- Works well with diverse accents (Whisper is multilingual)
- Easy to debug and improve

**Cons:**
- Two-stage pipeline adds latency
- ASR errors propagate to classifier

### Wav2Vec-VM-Finetune

**Architecture:** Single-stage end-to-end
- Fine-tuned wav2vec2-xls-r-300m directly classifies audio as human/machine

**Pros:**
- Single model, simpler pipeline
- No intermediate text step
- Can learn acoustic patterns (tone, beeps, background noise)

**Cons:**
- Larger model (300M parameters vs ~250M for Whisper-small)
- Less explainable (no transcript)
- English-only training

---

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| GPU | None (CPU inference) | NVIDIA GPU 8GB+ VRAM |
| Storage | 10 GB | 50+ GB (for training data) |

### Software Requirements

```bash
# Operating System
Ubuntu 20.04+ / Debian 11+ / CentOS 8+

# Python
Python 3.10+

# GPU Support (Optional but recommended)
CUDA 12.x
cuDNN 9.x

# Audio Libraries
libsndfile1
ffmpeg
```

### Installation Commands

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip \
    libsndfile1 ffmpeg build-essential

# For GPU support (NVIDIA)
# Follow NVIDIA CUDA installation guide for your distribution
```

---

## Data Preparation Guide

### Step 1: Audio Collection

#### Sources of Training Data

1. **Your Own Call Recordings** (Best)
   - Export from your PBX/dialer system
   - Ensure proper consent/compliance

2. **Public Datasets**
   - Common Voice (mozilla.org)
   - LibriSpeech
   - VoxForge

3. **Synthetic Data**
   - Text-to-speech for machine messages
   - Record actual voicemail greetings

#### Audio Requirements

| Parameter | Value |
|-----------|-------|
| Sample Rate | 16kHz (will be resampled if different) |
| Channels | Mono (will be converted if stereo) |
| Bit Depth | 16-bit PCM |
| Duration | 1-3 seconds (2s optimal) |
| Format | WAV, MP3, FLAC, M4A, OGG |

### Step 2: Audio Preprocessing

Create a preprocessing script:

```python
#!/usr/bin/env python3
"""
audio_preprocessor.py - Prepare audio files for training
"""
import os
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path

TARGET_SR = 16000
TARGET_DURATION = 2.0  # seconds

def preprocess_audio(input_path, output_path):
    """Load, resample, convert to mono, and clip/pad to 2 seconds."""
    try:
        # Load audio
        audio, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)

        # Clip or pad to target duration
        target_samples = int(TARGET_SR * TARGET_DURATION)

        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))

        # Normalize amplitude
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95

        # Save
        sf.write(output_path, audio, TARGET_SR)
        return True

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def batch_preprocess(input_dir, output_dir):
    """Process all audio files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}

    processed = 0
    failed = 0

    for file in input_path.rglob('*'):
        if file.suffix.lower() in extensions:
            out_file = output_path / f"{file.stem}.wav"
            if preprocess_audio(str(file), str(out_file)):
                processed += 1
            else:
                failed += 1

    print(f"Processed: {processed}, Failed: {failed}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    batch_preprocess(args.input_dir, args.output_dir)
```

### Step 3: Data Labeling

#### Manual Labeling Guidelines

Create a simple labeling tool or use these rules:

**HUMAN Labels:**
- Direct responses: "Hello", "Hello?", "Yes?", "Speaking"
- Names: "This is John", "Mike here"
- Questions: "Who is this?", "Can I help you?"

**MACHINE Labels:**
- Voicemail systems: "You have reached...", "Please leave a message"
- Instructions: "Press 1 for...", "If you know your party's extension"
- Automated greetings: "The person you are calling is not available"
- Beeps and tones (no speech)

#### CSV Format for Whisper Fine-tuning

Create `data_for_whisper.csv`:

```csv
path,transcription
/path/to/processed/audio001.wav,Hello this is the voicemail of John Smith please leave a message
/path/to/processed/audio002.wav,Hello
/path/to/processed/audio003.wav,You have reached the voice messaging system
/path/to/processed/audio004.wav,Hi this is Sarah
/path/to/processed/audio005.wav,The number you have dialed is not available
```

**Tips:**
- Use absolute paths
- Transcribe exactly what you hear
- Include punctuation for clarity
- Lowercase is fine (normalized during training)

#### CSV Format for Text Classifier

Create `data_for_text.csv`:

```csv
text,label
hello this is the voicemail of john smith please leave a message,machine
hello,human
you have reached the voice messaging system,machine
hi this is sarah,human
the number you have dialed is not available,machine
yes speaking,human
press one for sales press two for support,machine
who is this,human
```

### Step 4: Data Splitting

Recommended splits:
- **Training:** 80%
- **Validation:** 10%
- **Test:** 10%

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data_for_text.csv")

# Split
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'])

# Save
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
```

### Data Quantity Recommendations

| Use Case | Minimum Samples | Recommended |
|----------|-----------------|-------------|
| Basic AMD | 500 per class | 2,000+ per class |
| High Accuracy | 2,000 per class | 10,000+ per class |
| Production | 5,000+ per class | 50,000+ per class |

---

## Whisper-VM-Finetune Complete Guide

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/whisper-vm-finetune.git
cd whisper-vm-finetune

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Directory Structure

```
whisper-vm-finetune/
├── client/
│   └── ws_client.py              # Test client
├── server/
│   └── ws_server.py              # Production WebSocket server
├── scripts/
│   ├── finetune_whisper_small.py # Whisper fine-tuning
│   ├── train_text_classifier.py  # Text classifier training
│   ├── bulk_ws_eval.py           # Batch evaluation
│   └── utils.py                  # Utilities
├── models/
│   ├── whisper-small-finetuned/  # Fine-tuned Whisper
│   ├── whisper-ct2/              # CTranslate2 optimized
│   └── text_cls.joblib           # Text classifier
├── data/                         # Training audio files
├── requirements.txt
└── README.md
```

### Training Pipeline

#### Step 1: Fine-tune Whisper

```bash
# Basic training
python scripts/finetune_whisper_small.py \
    --csv /absolute/path/to/data_for_whisper.csv \
    --out_dir models/whisper-small-finetuned \
    --batch 8 \
    --epochs 3 \
    --num_workers 4

# For limited GPU memory (4-6GB)
python scripts/finetune_whisper_small.py \
    --csv /absolute/path/to/data_for_whisper.csv \
    --out_dir models/whisper-small-finetuned \
    --batch 2 \
    --epochs 5 \
    --num_workers 2
```

**Training Parameters Explained:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--csv` | Required | Path to training CSV |
| `--out_dir` | Required | Output directory for model |
| `--batch` | 8 | Batch size (reduce for less VRAM) |
| `--epochs` | 3 | Number of training epochs |
| `--num_workers` | 4 | Data loading workers |
| `--lr` | 1e-5 | Learning rate |

**Expected Training Time:**

| Dataset Size | GPU | Time |
|--------------|-----|------|
| 1,000 samples | RTX 3080 | ~30 min |
| 10,000 samples | RTX 3080 | ~3 hours |
| 100,000 samples | RTX 3080 | ~30 hours |

#### Step 2: Convert to CTranslate2 (Required for Production)

```bash
# Install CTranslate2 converter
pip install ctranslate2

# Convert model
ct2-transformers-converter \
    --model models/whisper-small-finetuned \
    --output_dir models/whisper-ct2 \
    --copy_files tokenizer.json tokenizer_config.json preprocessor_config.json \
                 special_tokens_map.json vocab.json merges.txt normalizer.json \
                 generation_config.json \
    --quantization float16

# For CPU-only deployment
ct2-transformers-converter \
    --model models/whisper-small-finetuned \
    --output_dir models/whisper-ct2-cpu \
    --copy_files tokenizer.json tokenizer_config.json preprocessor_config.json \
                 special_tokens_map.json vocab.json merges.txt normalizer.json \
                 generation_config.json \
    --quantization int8_float16
```

#### Step 3: Train Text Classifier

```bash
python scripts/train_text_classifier.py \
    --csv /absolute/path/to/data_for_text.csv \
    --out models/text_cls.joblib \
    --word_ngrams 1,2 \
    --char_ngrams 3,6
```

**Expected Output:**
```
              precision    recall  f1-score   support

       human       0.95      0.97      0.96       150
     machine       0.97      0.95      0.96       150

    accuracy                           0.96       300
   macro avg       0.96      0.96      0.96       300
```

### Running the Server

```bash
# Set environment variables
export WHISPER_MODEL_DIR=models/whisper-ct2
export DEVICE=cuda                     # or 'cpu'
export COMPUTE_TYPE=float16            # or 'int8_float16' for CPU
export CLASSIFIER_PATH=models/text_cls.joblib
export EARLY_SEC=2.0

# Start server
uvicorn server.ws_server:app --host 0.0.0.0 --port 8080

# Production with multiple workers
uvicorn server.ws_server:app --host 0.0.0.0 --port 8080 --workers 4
```

### Testing

```bash
# Test single file
python client/ws_client.py --wav /path/to/test.wav --url ws://localhost:8080/ws/amd

# Batch evaluation
python scripts/bulk_ws_eval.py \
    --audio_dir /path/to/test_folder \
    --csv_out results.csv \
    --url ws://localhost:8080/ws/amd \
    --concurrency 4
```

### Response Format

```json
{
    "type": "final",
    "label": "machine",
    "confidence": 0.98,
    "proba_human": 0.02,
    "transcript": "please leave your name and number after the tone",
    "elapsed_ms": 150
}
```

| Field | Description |
|-------|-------------|
| `type` | "early" (2s decision) or "final" (complete) |
| `label` | "human" or "machine" |
| `confidence` | Confidence score (0.0-1.0) |
| `proba_human` | Probability of human (0.0-1.0) |
| `transcript` | Transcribed text |
| `elapsed_ms` | Processing time in milliseconds |

---

## Wav2Vec-VM-Finetune Complete Guide

### Model Overview

- **Base Model:** facebook/wav2vec2-xls-r-300m
- **Parameters:** 300 million
- **Input:** 2 seconds of audio at 16kHz
- **Output:** Binary classification (human/machine)
- **Accuracy:** 98% (reported)

### Installation

```bash
# Create new directory
mkdir wav2vec-vm-detection
cd wav2vec-vm-detection

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchaudio transformers datasets librosa soundfile numpy
pip install fastapi uvicorn websockets
```

### Download Pre-trained Model

```python
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# Download model
model = Wav2Vec2ForSequenceClassification.from_pretrained("jakeBland/wav2vec-vm-finetune")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("jakeBland/wav2vec-vm-finetune")

# Save locally
model.save_pretrained("models/wav2vec-vm")
feature_extractor.save_pretrained("models/wav2vec-vm")
```

### Inference Script

Create `wav2vec_inference.py`:

```python
#!/usr/bin/env python3
"""
wav2vec_inference.py - Inference with wav2vec-vm-finetune model
"""
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

class Wav2VecVoicemailDetector:
    def __init__(self, model_path="jakeBland/wav2vec-vm-finetune", device=None):
        """Initialize the voicemail detector."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model on {self.device}...")
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

        self.sample_rate = 16000
        self.max_duration = 2.0  # seconds

    def preprocess_audio(self, audio_path):
        """Load and preprocess audio file."""
        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Clip to 2 seconds
        max_samples = int(self.sample_rate * self.max_duration)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[1] < max_samples:
            # Pad with zeros
            padding = max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform.squeeze().numpy()

    def preprocess_bytes(self, audio_bytes):
        """Preprocess audio from bytes (PCM16 16kHz mono)."""
        # Convert bytes to numpy array
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Clip/pad to 2 seconds
        max_samples = int(self.sample_rate * self.max_duration)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)))

        return audio

    def predict(self, audio):
        """
        Predict if audio is human or machine.

        Args:
            audio: numpy array of audio samples or path to audio file

        Returns:
            dict with label, confidence, and probabilities
        """
        # Load if path
        if isinstance(audio, str):
            audio = self.preprocess_audio(audio)

        # Extract features
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        # Get prediction
        probs = probs.cpu().numpy()[0]
        predicted_class = np.argmax(probs)

        # Map to labels (adjust based on model's label mapping)
        # Typically: 0 = human, 1 = machine (verify with model config)
        labels = ["human", "machine"]

        return {
            "label": labels[predicted_class],
            "confidence": float(probs[predicted_class]),
            "proba_human": float(probs[0]),
            "proba_machine": float(probs[1])
        }

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--model", default="jakeBland/wav2vec-vm-finetune")
    args = parser.parse_args()

    detector = Wav2VecVoicemailDetector(args.model)
    result = detector.predict(args.audio)

    print(f"Label: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Human prob: {result['proba_human']:.2%}")
    print(f"Machine prob: {result['proba_machine']:.2%}")
```

### WebSocket Server for Wav2Vec

Create `wav2vec_server.py`:

```python
#!/usr/bin/env python3
"""
wav2vec_server.py - WebSocket server for wav2vec voicemail detection
"""
import os
import json
import time
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

app = FastAPI()

# Configuration
MODEL_PATH = os.getenv("WAV2VEC_MODEL", "jakeBland/wav2vec-vm-finetune")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
EARLY_SEC = float(os.getenv("EARLY_SEC", "2.0"))
SAMPLE_RATE = 16000

# Load model on startup
print(f"Loading model: {MODEL_PATH}")
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}")

def pcm16_to_float32(pcm_bytes):
    """Convert PCM16 bytes to float32 array."""
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

def classify_audio(audio_array):
    """Classify audio array as human or machine."""
    # Ensure 2 seconds
    target_samples = int(SAMPLE_RATE * EARLY_SEC)
    if len(audio_array) > target_samples:
        audio_array = audio_array[:target_samples]
    elif len(audio_array) < target_samples:
        audio_array = np.pad(audio_array, (0, target_samples - len(audio_array)))

    # Extract features
    inputs = feature_extractor(
        audio_array,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    predicted_class = int(np.argmax(probs))
    labels = ["human", "machine"]

    return {
        "label": labels[predicted_class],
        "confidence": float(probs[predicted_class]),
        "proba_human": float(probs[0]),
        "proba_machine": float(probs[1])
    }

@app.websocket("/ws/amd")
async def amd_websocket(ws: WebSocket):
    """WebSocket endpoint for streaming AMD."""
    await ws.accept()

    audio_buffer = bytearray()
    early_sent = False
    start_time = time.time()

    try:
        while True:
            message = await ws.receive()

            if "bytes" in message:
                # Binary audio frame
                audio_buffer.extend(message["bytes"])

                # Check for early decision
                samples = len(audio_buffer) // 2  # 2 bytes per sample
                duration = samples / SAMPLE_RATE

                if duration >= EARLY_SEC and not early_sent:
                    audio_array = pcm16_to_float32(bytes(audio_buffer))
                    result = classify_audio(audio_array)
                    result["type"] = "early"
                    result["elapsed_ms"] = int((time.time() - start_time) * 1000)
                    await ws.send_json(result)
                    early_sent = True

            elif "text" in message:
                # JSON control message
                data = json.loads(message["text"])

                if data.get("type") == "flush":
                    # Final classification
                    if len(audio_buffer) > 0:
                        audio_array = pcm16_to_float32(bytes(audio_buffer))
                        result = classify_audio(audio_array)
                        result["type"] = "final"
                        result["elapsed_ms"] = int((time.time() - start_time) * 1000)
                        await ws.send_json(result)
                    break

                elif data.get("type") == "reset":
                    audio_buffer = bytearray()
                    early_sent = False
                    start_time = time.time()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await ws.close(code=1011, reason=str(e))

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": MODEL_PATH, "device": DEVICE}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Fine-tuning Wav2Vec for Your Data

Create `finetune_wav2vec.py`:

```python
#!/usr/bin/env python3
"""
finetune_wav2vec.py - Fine-tune wav2vec2 for voicemail detection
"""
import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer
)
from dataclasses import dataclass
from typing import Dict, List, Union
import argparse

# Constants
SAMPLE_RATE = 16000
MAX_DURATION = 2.0  # seconds
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION)

def load_audio(path):
    """Load and preprocess audio file."""
    try:
        waveform, sr = torchaudio.load(path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Clip/pad to 2 seconds
        waveform = waveform.squeeze()
        if len(waveform) > MAX_SAMPLES:
            waveform = waveform[:MAX_SAMPLES]
        elif len(waveform) < MAX_SAMPLES:
            waveform = torch.nn.functional.pad(waveform, (0, MAX_SAMPLES - len(waveform)))

        return waveform.numpy()
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def prepare_dataset(csv_path):
    """Load CSV and prepare dataset."""
    df = pd.read_csv(csv_path)

    # Expected columns: path, label
    # label should be 'human' or 'machine'

    # Convert labels to integers
    label_map = {"human": 0, "machine": 1}
    df["label"] = df["label"].map(label_map)

    # Load audio
    print("Loading audio files...")
    audio_data = []
    labels = []

    for _, row in df.iterrows():
        audio = load_audio(row["path"])
        if audio is not None:
            audio_data.append(audio)
            labels.append(row["label"])

    print(f"Loaded {len(audio_data)} files")

    # Create dataset
    dataset = Dataset.from_dict({
        "audio": audio_data,
        "label": labels
    })

    return dataset

@dataclass
class DataCollatorForWav2Vec:
    """Data collator for wav2vec2 classification."""
    feature_extractor: Wav2Vec2FeatureExtractor

    def __call__(self, features: List[Dict[str, Union[List[float], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract audio arrays
        audio_arrays = [f["audio"] for f in features]
        labels = [f["label"] for f in features]

        # Process with feature extractor
        batch = self.feature_extractor(
            audio_arrays,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )

        batch["labels"] = torch.tensor(labels)
        return batch

def compute_metrics(eval_pred):
    """Compute accuracy metric."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to training CSV (path,label)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--base_model", default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    # Load dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(args.csv)

    # Split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # Load model and feature extractor
    print(f"Loading base model: {args.base_model}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.base_model)

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        label2id={"human": 0, "machine": 1},
        id2label={0: "human", 1: "machine"}
    )

    # Freeze feature extractor (optional, helps prevent overfitting)
    model.freeze_feature_encoder()

    # Data collator
    data_collator = DataCollatorForWav2Vec(feature_extractor)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size // 2,
        gradient_accumulation_steps=2,
        num_train_epochs=args.epochs,
        warmup_steps=500,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        save_total_limit=3,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving model to {args.out_dir}")
    trainer.save_model(args.out_dir)
    feature_extractor.save_pretrained(args.out_dir)

    print("Done!")

if __name__ == "__main__":
    main()
```

### Training Wav2Vec

```bash
# Prepare CSV with columns: path, label
# label should be 'human' or 'machine'

python finetune_wav2vec.py \
    --csv /path/to/training_data.csv \
    --out_dir models/wav2vec-vm-custom \
    --batch_size 8 \
    --epochs 10 \
    --lr 3e-4
```

---

## FreeSWITCH Integration

### Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  FreeSWITCH │────▶│  mod_audio   │────▶│  AMD Server   │
│             │◀────│  _stream     │◀────│  (WebSocket)  │
└─────────────┘     └──────────────┘     └───────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Your App    │
                    │  (ESL/API)   │
                    └──────────────┘
```

### Option 1: mod_audio_stream Integration

#### Install mod_audio_stream

```bash
# Clone mod_audio_stream
git clone https://github.com/drachtio/freeswitch-modules.git
cd freeswitch-modules/mod_audio_stream

# Build
./bootstrap.sh
./configure
make
make install

# Enable in modules.conf.xml
echo '<load module="mod_audio_stream"/>' >> /etc/freeswitch/autoload_configs/modules.conf.xml
```

#### Create AMD Application

Create `/etc/freeswitch/dialplan/default/amd.xml`:

```xml
<extension name="outbound_with_amd">
  <condition field="destination_number" expression="^amd_(.*)$">
    <!-- Set variables -->
    <action application="set" data="amd_ws_url=ws://localhost:8080/ws/amd"/>
    <action application="set" data="amd_timeout=5000"/>

    <!-- Answer and start AMD -->
    <action application="answer"/>
    <action application="audio_stream" data="${amd_ws_url} start"/>

    <!-- Wait for AMD decision -->
    <action application="sleep" data="2500"/>

    <!-- Stop streaming -->
    <action application="audio_stream" data="stop"/>

    <!-- Branch based on AMD result -->
    <action application="execute_extension" data="handle_amd_result"/>
  </condition>
</extension>

<extension name="handle_amd_result">
  <condition field="${amd_result}" expression="^human$">
    <action application="bridge" data="user/${destination_number}"/>
  </condition>
  <condition field="${amd_result}" expression="^machine$">
    <action application="playback" data="voicemail_message.wav"/>
    <action application="hangup"/>
  </condition>
</extension>
```

### Option 2: ESL (Event Socket Library) Integration

#### Python ESL Client

Create `freeswitch_amd.py`:

```python
#!/usr/bin/env python3
"""
freeswitch_amd.py - FreeSWITCH AMD integration via ESL
"""
import asyncio
import json
import websockets
import ESL  # pip install python-ESL

class FreeSwitchAMD:
    def __init__(self, fs_host="localhost", fs_port=8021, fs_password="ClueCon",
                 amd_url="ws://localhost:8080/ws/amd"):
        self.fs_host = fs_host
        self.fs_port = fs_port
        self.fs_password = fs_password
        self.amd_url = amd_url
        self.con = None

    def connect(self):
        """Connect to FreeSWITCH ESL."""
        self.con = ESL.ESLconnection(self.fs_host, str(self.fs_port), self.fs_password)
        if self.con.connected():
            print("Connected to FreeSWITCH")
            return True
        print("Failed to connect to FreeSWITCH")
        return False

    def originate_with_amd(self, destination, caller_id, context="default"):
        """Originate call with AMD detection."""
        # Create unique ID for this call
        call_uuid = self.con.api("create_uuid").getBody().strip()

        # Set up call with AMD
        dial_string = f"{{origination_uuid={call_uuid}," \
                     f"origination_caller_id_number={caller_id}," \
                     f"amd_enabled=true}}" \
                     f"sofia/gateway/your_gateway/{destination}"

        # Originate call
        result = self.con.api(f"originate {dial_string} &park()")

        return call_uuid

    async def run_amd(self, call_uuid, audio_data):
        """Run AMD on audio data."""
        async with websockets.connect(self.amd_url) as ws:
            # Send audio in chunks
            chunk_size = 640  # 20ms at 16kHz
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.02)  # 20ms between chunks

            # Request final decision
            await ws.send(json.dumps({"type": "flush"}))

            # Wait for response
            response = await ws.recv()
            return json.loads(response)

    def handle_amd_result(self, call_uuid, result):
        """Handle AMD result and take action."""
        label = result.get("label", "unknown")
        confidence = result.get("confidence", 0)

        print(f"AMD Result for {call_uuid}: {label} ({confidence:.2%})")

        if label == "human":
            # Connect to agent
            self.con.api(f"uuid_transfer {call_uuid} agent_queue")
        elif label == "machine":
            # Play voicemail message
            self.con.api(f"uuid_broadcast {call_uuid} voicemail.wav aleg")
            # Or hangup
            # self.con.api(f"uuid_kill {call_uuid}")

# Example usage
if __name__ == "__main__":
    amd = FreeSwitchAMD()
    if amd.connect():
        # Originate call
        uuid = amd.originate_with_amd("15551234567", "15559876543")
        print(f"Call UUID: {uuid}")
```

### Option 3: FreeSWITCH Lua Script

Create `/etc/freeswitch/scripts/amd.lua`:

```lua
-- amd.lua - AMD integration for FreeSWITCH
-- Requires lua-websockets and lua-cjson

local ws = require("websocket")
local json = require("cjson")

-- Configuration
local AMD_URL = "ws://localhost:8080/ws/amd"
local SAMPLE_RATE = 16000
local CHUNK_MS = 20
local CHUNK_SIZE = SAMPLE_RATE * CHUNK_MS / 1000 * 2  -- bytes

function amd_detect(session)
    local uuid = session:getVariable("uuid")
    freeswitch.consoleLog("INFO", "Starting AMD for " .. uuid .. "\n")

    -- Connect to AMD server
    local client = ws.client.sync()
    local ok, err = client:connect(AMD_URL)

    if not ok then
        freeswitch.consoleLog("ERR", "Failed to connect to AMD: " .. tostring(err) .. "\n")
        return "error"
    end

    -- Start recording to buffer
    local audio_buffer = ""
    local start_time = os.time()
    local max_duration = 3  -- seconds

    -- Read audio frames
    session:setInputCallback("on_dtmf", "")

    while session:ready() and (os.time() - start_time) < max_duration do
        local frame = session:read(CHUNK_MS, SAMPLE_RATE, 1)
        if frame then
            -- Send to AMD server
            client:send(frame, ws.TEXT)
        end

        -- Check for early response
        local response = client:receive(0)  -- non-blocking
        if response then
            local result = json.decode(response)
            if result.type == "early" and result.confidence > 0.9 then
                freeswitch.consoleLog("INFO", "Early AMD: " .. result.label .. "\n")
                client:close()
                return result.label
            end
        end
    end

    -- Request final decision
    client:send(json.encode({type = "flush"}))

    -- Wait for final response
    local response = client:receive(5000)
    client:close()

    if response then
        local result = json.decode(response)
        freeswitch.consoleLog("INFO", "Final AMD: " .. result.label .. " (" .. result.confidence .. ")\n")
        return result.label
    end

    return "unknown"
end

-- Main entry point
function main(session)
    session:answer()

    local result = amd_detect(session)
    session:setVariable("amd_result", result)

    if result == "human" then
        -- Transfer to agent
        session:transfer("agent_queue", "XML", "default")
    elseif result == "machine" then
        -- Play message and hangup
        session:streamFile("voicemail_message.wav")
        session:hangup()
    else
        -- Unknown - treat as human
        session:transfer("agent_queue", "XML", "default")
    end
end

main(session)
```

### Option 4: External Application via UniMRCP

For enterprise deployments, use UniMRCP to bridge FreeSWITCH to your AMD service:

```xml
<!-- mrcp_profiles/unimrcp.xml -->
<profile name="amd" version="2">
  <param name="server-ip" value="localhost"/>
  <param name="server-port" value="5060"/>
  <param name="resource-location" value=""/>
  <param name="speechrecog" value="speechrecog"/>
</profile>
```

### FreeSWITCH Configuration Best Practices

1. **Audio Format Settings**

```xml
<!-- vars.xml -->
<X-PRE-PROCESS cmd="set" data="sound_prefix=$${sounds_dir}/en/us/callie"/>
<X-PRE-PROCESS cmd="set" data="default_sample_rate=16000"/>
<X-PRE-PROCESS cmd="set" data="default_codec=PCMU"/>
```

2. **Call Leg Isolation**

Ensure AMD only sees the far end audio:

```xml
<action application="set" data="RECORD_STEREO=false"/>
<action application="set" data="RECORD_READ_ONLY=true"/>
```

3. **Timeout Handling**

```xml
<!-- Set AMD timeout -->
<action application="set" data="amd_timeout_ms=5000"/>
<action application="set" data="playback_terminators=none"/>
```

---

## Accuracy Enhancement Strategies

### 1. Data Quality Improvements

#### Balanced Dataset

```python
# Ensure equal samples per class
from sklearn.utils import resample

# Upsample minority class
df_human = df[df['label'] == 'human']
df_machine = df[df['label'] == 'machine']

if len(df_human) < len(df_machine):
    df_human = resample(df_human, n_samples=len(df_machine), random_state=42)
else:
    df_machine = resample(df_machine, n_samples=len(df_human), random_state=42)

df_balanced = pd.concat([df_human, df_machine])
```

#### Data Augmentation

```python
import torch
import torchaudio.transforms as T

def augment_audio(waveform, sr=16000):
    """Apply audio augmentations."""
    augmentations = []

    # Time stretch (slight speed variation)
    stretch = T.TimeStretch(n_freq=201)
    augmentations.append(stretch(waveform.unsqueeze(0), 0.9 + 0.2 * torch.rand(1).item()))

    # Pitch shift
    pitch = T.PitchShift(sr, n_steps=torch.randint(-2, 3, (1,)).item())
    augmentations.append(pitch(waveform))

    # Add noise
    noise = torch.randn_like(waveform) * 0.005
    augmentations.append(waveform + noise)

    # Volume variation
    volume = 0.8 + 0.4 * torch.rand(1).item()
    augmentations.append(waveform * volume)

    return augmentations
```

### 2. Model Improvements

#### Ensemble Methods

```python
def ensemble_predict(whisper_result, wav2vec_result, threshold=0.6):
    """Combine predictions from both models."""
    whisper_human = whisper_result['proba_human']
    wav2vec_human = wav2vec_result['proba_human']

    # Weighted average
    combined_human = 0.5 * whisper_human + 0.5 * wav2vec_human

    if combined_human >= threshold:
        return {"label": "human", "confidence": combined_human}
    else:
        return {"label": "machine", "confidence": 1 - combined_human}
```

#### Threshold Tuning

```python
from sklearn.metrics import precision_recall_curve

def find_optimal_threshold(y_true, y_proba, target_precision=0.95):
    """Find threshold that achieves target precision."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Find threshold where precision >= target
    for i, p in enumerate(precisions):
        if p >= target_precision:
            return thresholds[i]

    return 0.5  # default
```

### 3. Feature Engineering (Text Classifier)

```python
# Enhanced text features
from sklearn.feature_extraction.text import TfidfVectorizer

# Add domain-specific features
def extract_features(text):
    features = {}

    # Length features
    features['word_count'] = len(text.split())
    features['char_count'] = len(text)

    # Pattern features
    features['has_greeting'] = int(any(g in text.lower() for g in ['hello', 'hi', 'hey']))
    features['has_question'] = int('?' in text)
    features['has_instruction'] = int(any(i in text.lower() for i in ['press', 'dial', 'leave']))
    features['has_name'] = int(any(n in text.lower() for n in ['this is', 'speaking', 'name is']))

    return features
```

### 4. Error Analysis Pipeline

```python
def analyze_errors(predictions_csv, ground_truth_csv):
    """Analyze prediction errors."""
    import pandas as pd

    pred_df = pd.read_csv(predictions_csv)
    truth_df = pd.read_csv(ground_truth_csv)

    merged = pred_df.merge(truth_df, on='file')

    # Find errors
    errors = merged[merged['predicted_label'] != merged['true_label']]

    # Categorize errors
    false_positives = errors[errors['predicted_label'] == 'human']
    false_negatives = errors[errors['predicted_label'] == 'machine']

    print(f"Total errors: {len(errors)}")
    print(f"False positives (predicted human, was machine): {len(false_positives)}")
    print(f"False negatives (predicted machine, was human): {len(false_negatives)}")

    # Common patterns in errors
    print("\nFalse positive transcripts:")
    for _, row in false_positives.head(10).iterrows():
        print(f"  - {row['transcript']}")

    return errors
```

### 5. Continuous Learning Pipeline

```python
def retrain_on_feedback(feedback_csv, existing_model_path, output_path):
    """Retrain model incorporating user feedback."""
    import pandas as pd

    # Load feedback (corrected predictions)
    feedback = pd.read_csv(feedback_csv)

    # Validate feedback
    feedback = feedback[feedback['corrected_label'].isin(['human', 'machine'])]

    # Combine with existing training data
    existing = pd.read_csv("training_data.csv")
    combined = pd.concat([existing, feedback[['text', 'corrected_label']].rename(
        columns={'corrected_label': 'label'}
    )])

    # Remove duplicates
    combined = combined.drop_duplicates(subset=['text'])

    # Retrain
    # ... training code ...

    print(f"Retrained on {len(combined)} samples")
```

---

## Production Deployment

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.2-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Environment variables
ENV WHISPER_MODEL_DIR=/app/models/whisper-ct2
ENV CLASSIFIER_PATH=/app/models/text_cls.joblib
ENV DEVICE=cuda
ENV COMPUTE_TYPE=float16
ENV EARLY_SEC=2.0

EXPOSE 8080

CMD ["uvicorn", "server.ws_server:app", "--host", "0.0.0.0", "--port", "8080"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  amd-server:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - WHISPER_MODEL_DIR=/app/models/whisper-ct2
      - CLASSIFIER_PATH=/app/models/text_cls.joblib
      - DEVICE=cuda
      - COMPUTE_TYPE=float16
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Optional: Prometheus metrics
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  # Optional: Grafana dashboard
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  grafana-data:
```

### Kubernetes Deployment

Create `kubernetes/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: amd-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: amd-server
  template:
    metadata:
      labels:
        app: amd-server
    spec:
      containers:
      - name: amd-server
        image: your-registry/amd-server:latest
        ports:
        - containerPort: 8080
        env:
        - name: WHISPER_MODEL_DIR
          value: /app/models/whisper-ct2
        - name: DEVICE
          value: cuda
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 8Gi
          requests:
            memory: 4Gi
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: amd-server
spec:
  selector:
    app: amd-server
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
```

### Monitoring & Logging

Add to `ws_server.py`:

```python
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUESTS = Counter('amd_requests_total', 'Total AMD requests', ['result'])
LATENCY = Histogram('amd_latency_seconds', 'AMD latency in seconds')

# Structured logging
logging.basicConfig(
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Add to endpoint
@app.websocket("/ws/amd")
async def amd_websocket(ws: WebSocket):
    start = time.time()
    try:
        # ... existing code ...

        # Log result
        logger.info(json.dumps({
            "event": "amd_complete",
            "label": result["label"],
            "confidence": result["confidence"],
            "latency_ms": result["elapsed_ms"]
        }))

        # Update metrics
        REQUESTS.labels(result=result["label"]).inc()
        LATENCY.observe(time.time() - start)

    except Exception as e:
        logger.error(f"AMD error: {e}")
        REQUESTS.labels(result="error").inc()

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

## Troubleshooting

### Common Issues

#### 1. Model Loading Errors

```
Error: Could not find model weights
```

**Solution:**
```bash
# Check model files exist
ls -la models/whisper-ct2/

# Re-download or re-convert model
ct2-transformers-converter --model openai/whisper-small --output_dir models/whisper-ct2
```

#### 2. CUDA/GPU Errors

```
Error: CUDA out of memory
```

**Solution:**
```bash
# Use CPU instead
export DEVICE=cpu
export COMPUTE_TYPE=int8_float16

# Or reduce batch size in training
python scripts/finetune_whisper_small.py --batch 2
```

#### 3. Audio Format Issues

```
Error: Unable to read audio file
```

**Solution:**
```bash
# Install ffmpeg
sudo apt install ffmpeg

# Convert problematic files
ffmpeg -i input.mp3 -ar 16000 -ac 1 -f wav output.wav
```

#### 4. WebSocket Connection Issues

```
Error: Connection refused
```

**Solution:**
```bash
# Check server is running
curl http://localhost:8080/health

# Check firewall
sudo ufw allow 8080

# Check port binding
netstat -tlnp | grep 8080
```

#### 5. Low Accuracy

**Solution:**
1. Increase training data quantity
2. Balance dataset classes
3. Add data augmentation
4. Tune classification threshold
5. Review and clean mislabeled data

### Performance Tuning

```bash
# GPU performance
nvidia-smi -pm 1  # Enable persistence mode
nvidia-smi -pl 250  # Set power limit

# Server performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Increase workers
uvicorn server.ws_server:app --workers 4
```

### Debugging Commands

```bash
# Test single file
python client/ws_client.py --wav test.wav --url ws://localhost:8080/ws/amd

# Check model output
python -c "
from faster_whisper import WhisperModel
model = WhisperModel('models/whisper-ct2', device='cpu')
segments, _ = model.transcribe('test.wav')
for s in segments:
    print(s.text)
"

# Test classifier
python -c "
import joblib
clf = joblib.load('models/text_cls.joblib')
text = 'hello this is john'
# ... prediction code
"
```

---

## Quick Reference Commands

```bash
# === WHISPER-VM-FINETUNE ===

# Install
pip install -r requirements.txt

# Fine-tune Whisper
python scripts/finetune_whisper_small.py --csv data.csv --out_dir models/whisper-small-finetuned

# Convert to CTranslate2
ct2-transformers-converter --model models/whisper-small-finetuned --output_dir models/whisper-ct2 --quantization float16

# Train classifier
python scripts/train_text_classifier.py --csv text_labels.csv --out models/text_cls.joblib

# Start server
export WHISPER_MODEL_DIR=models/whisper-ct2
uvicorn server.ws_server:app --host 0.0.0.0 --port 8080

# Test
python client/ws_client.py --wav test.wav


# === WAV2VEC-VM-FINETUNE ===

# Download model
python -c "from transformers import *; m=Wav2Vec2ForSequenceClassification.from_pretrained('jakeBland/wav2vec-vm-finetune'); m.save_pretrained('models/wav2vec')"

# Fine-tune
python finetune_wav2vec.py --csv data.csv --out_dir models/wav2vec-custom

# Start server
python wav2vec_server.py


# === FREESWITCH ===

# Restart FreeSWITCH
systemctl restart freeswitch

# Check logs
tail -f /var/log/freeswitch/freeswitch.log

# Test ESL connection
fs_cli -x "status"
```

---

## Support & Resources

- **Whisper Documentation:** https://github.com/openai/whisper
- **wav2vec2 Documentation:** https://huggingface.co/docs/transformers/model_doc/wav2vec2
- **FreeSWITCH Documentation:** https://freeswitch.org/confluence/
- **CTranslate2:** https://github.com/OpenNMT/CTranslate2
- **faster-whisper:** https://github.com/guillaumekln/faster-whisper

---

*Last updated: December 2025*
*Version: 1.0*
