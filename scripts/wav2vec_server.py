#!/usr/bin/env python3
"""
wav2vec_server.py - WebSocket server for wav2vec voicemail detection

This server provides real-time voicemail detection using the wav2vec2 model.
It accepts PCM16 16kHz mono audio via WebSocket and returns classification results.

Environment Variables:
    WAV2VEC_MODEL: Model path or HuggingFace name (default: jakeBland/wav2vec-vm-finetune)
    DEVICE: 'cuda' or 'cpu' (default: auto-detect)
    EARLY_SEC: Early decision threshold in seconds (default: 2.0)
    THRESHOLD: Classification threshold (default: 0.5)

Usage:
    # Start server
    uvicorn scripts.wav2vec_server:app --host 0.0.0.0 --port 8081

    # Or run directly
    python scripts/wav2vec_server.py

WebSocket Protocol:
    - Connect to: ws://host:port/ws/amd
    - Send binary frames: PCM16 16kHz mono audio (e.g., 20ms chunks = 640 bytes)
    - Send JSON control: {"type": "flush"} to get final decision
    - Send JSON control: {"type": "reset"} to reset buffer
    - Receive JSON responses with classification results
"""
import os
import json
import time
import asyncio
import logging
from typing import Dict, Any

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

try:
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
except ImportError:
    print("Please install transformers: pip install transformers")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
MODEL_PATH = os.getenv("WAV2VEC_MODEL", "jakeBland/wav2vec-vm-finetune")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
EARLY_SEC = float(os.getenv("EARLY_SEC", "2.0"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
SAMPLE_RATE = 16000

# Initialize FastAPI
app = FastAPI(
    title="Wav2Vec Voicemail Detection API",
    description="Real-time voicemail detection using wav2vec2",
    version="1.0.0"
)

# Global model references
model = None
feature_extractor = None
id2label = {0: "human", 1: "machine"}


@app.on_event("startup")
async def load_model():
    """Load model on server startup."""
    global model, feature_extractor, id2label

    logger.info(f"Loading model: {MODEL_PATH}")
    logger.info(f"Device: {DEVICE}")

    try:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)

        model.to(DEVICE)
        model.eval()

        # Get label mapping from model config if available
        if hasattr(model.config, 'id2label'):
            id2label = model.config.id2label

        logger.info(f"Model loaded successfully. Labels: {id2label}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert PCM16 bytes to float32 array."""
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def classify_audio(audio_array: np.ndarray) -> Dict[str, Any]:
    """
    Classify audio array as human or machine.

    Args:
        audio_array: Float32 audio array at 16kHz

    Returns:
        Dictionary with classification results
    """
    start_time = time.time()

    # Ensure correct length (2 seconds)
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

    # Move to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    # Get prediction
    predicted_class = int(np.argmax(probs))
    predicted_label = id2label.get(predicted_class, "unknown")

    # Apply threshold for human classification
    label2id = {v: k for k, v in id2label.items()}
    human_idx = label2id.get("human", 0)
    machine_idx = label2id.get("machine", 1)

    if probs[human_idx] >= THRESHOLD:
        final_label = "human"
    else:
        final_label = "machine"

    elapsed_ms = int((time.time() - start_time) * 1000)

    return {
        "label": final_label,
        "confidence": float(probs[label2id.get(final_label, predicted_class)]),
        "proba_human": float(probs[human_idx]),
        "proba_machine": float(probs[machine_idx]),
        "elapsed_ms": elapsed_ms
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Wav2Vec Voicemail Detection",
        "model": MODEL_PATH,
        "device": DEVICE,
        "status": "ready"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_PATH,
        "device": DEVICE,
        "threshold": THRESHOLD,
        "early_sec": EARLY_SEC
    }


@app.websocket("/ws/amd")
async def amd_websocket(ws: WebSocket):
    """
    WebSocket endpoint for streaming AMD.

    Protocol:
        1. Connect to this endpoint
        2. Send binary frames containing PCM16 16kHz mono audio
        3. Receive JSON early decision after EARLY_SEC seconds
        4. Send {"type": "flush"} to get final decision
        5. Send {"type": "reset"} to clear buffer and start over
    """
    await ws.accept()

    client_id = id(ws)
    logger.info(f"Client {client_id} connected")

    audio_buffer = bytearray()
    early_sent = False
    start_time = time.time()

    try:
        while True:
            message = await ws.receive()

            if "bytes" in message:
                # Binary audio frame
                audio_buffer.extend(message["bytes"])

                # Check for early decision threshold
                samples = len(audio_buffer) // 2  # 2 bytes per PCM16 sample
                duration = samples / SAMPLE_RATE

                if duration >= EARLY_SEC and not early_sent:
                    # Convert and classify
                    audio_array = pcm16_to_float32(bytes(audio_buffer))
                    result = classify_audio(audio_array)

                    result["type"] = "early"
                    result["duration"] = duration

                    await ws.send_json(result)
                    early_sent = True

                    logger.info(f"Client {client_id} early: {result['label']} "
                              f"({result['confidence']:.2%})")

            elif "text" in message:
                # JSON control message
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await ws.send_json({"error": "Invalid JSON"})
                    continue

                msg_type = data.get("type")

                if msg_type == "flush":
                    # Final classification
                    if len(audio_buffer) > 0:
                        audio_array = pcm16_to_float32(bytes(audio_buffer))
                        result = classify_audio(audio_array)

                        result["type"] = "final"
                        result["duration"] = len(audio_buffer) / 2 / SAMPLE_RATE
                        result["total_elapsed_ms"] = int((time.time() - start_time) * 1000)

                        await ws.send_json(result)

                        logger.info(f"Client {client_id} final: {result['label']} "
                                  f"({result['confidence']:.2%})")
                    else:
                        await ws.send_json({
                            "type": "final",
                            "error": "No audio received",
                            "label": "unknown"
                        })

                    break  # Close connection after final

                elif msg_type == "reset":
                    # Reset buffer
                    audio_buffer = bytearray()
                    early_sent = False
                    start_time = time.time()

                    await ws.send_json({"type": "reset", "status": "ok"})
                    logger.info(f"Client {client_id} reset")

                else:
                    await ws.send_json({"error": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")

    except Exception as e:
        logger.error(f"Client {client_id} error: {e}")
        try:
            await ws.close(code=1011, reason=str(e))
        except:
            pass


@app.post("/classify")
async def classify_file(file_path: str):
    """
    Classify an audio file (for testing).

    Args:
        file_path: Path to audio file on server
    """
    import torchaudio

    try:
        waveform, sr = torchaudio.load(file_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze()

        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        audio_array = waveform.numpy()
        result = classify_audio(audio_array)
        result["file"] = file_path

        return result

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8081"))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"Starting wav2vec AMD server on {host}:{port}")
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Early decision: {EARLY_SEC}s")
    print(f"Threshold: {THRESHOLD}")

    uvicorn.run(app, host=host, port=port)
