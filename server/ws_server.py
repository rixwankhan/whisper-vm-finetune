\
import os, json, time
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from scipy import sparse
import joblib

from scripts.utils import (
    DEFAULT_SAMPLE_RATE, MAX_SEC, N_SAMPLES,
    wav_bytes_to_float32, ensure_2s, normalize_text
)

app = FastAPI(title="Whisper-only AMD WebSocket", version="1.1")

WHISPER_MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", "small")
DEVICE = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
EARLY_SEC = float(os.getenv("EARLY_SEC", "2.0"))  # set 0 to disable early decision

whisper = WhisperModel(WHISPER_MODEL_DIR, device=DEVICE, compute_type=COMPUTE_TYPE)

CLASSIFIER_PATH = os.getenv("CLASSIFIER_PATH", "models/text_cls.joblib")
bundle = joblib.load(CLASSIFIER_PATH)
clf, word_v, char_v = bundle["clf"], bundle["word_v"], bundle["char_v"]

def classify_text(text: str):
    Xt_w = word_v.transform([text])
    Xt_c = char_v.transform([text])
    X = sparse.hstack([Xt_w, Xt_c])
    proba_h = clf.predict_proba(X)[0,1]
    label = "human" if proba_h >= 0.5 else "machine"
    conf = float(max(proba_h, 1.0-proba_h))
    return label, conf, proba_h

@app.websocket("/ws/amd")
async def amd_ws(ws: WebSocket):
    await ws.accept()
    buf = np.zeros((0,), dtype=np.float32)
    made_early = False
    t_start = time.time()

    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                frame = wav_bytes_to_float32(msg["bytes"], sample_rate=DEFAULT_SAMPLE_RATE)
                buf = np.concatenate([buf, frame])

                # optional early decision on first EARLY_SEC seconds
                if EARLY_SEC > 0 and not made_early and buf.shape[0] >= int(DEFAULT_SAMPLE_RATE*EARLY_SEC):
                    await send_decision(ws, buf[:int(DEFAULT_SAMPLE_RATE*EARLY_SEC)], t_start, tag="early")
                    made_early = True

            elif "text" in msg and msg["text"] is not None:
                # JSON control messages
                try:
                    data = json.loads(msg["text"])
                except Exception:
                    data = {"type":"unknown"}

                if data.get("type") == "flush":
                    await send_decision(ws, buf, t_start, tag="final")
                elif data.get("type") == "reset":
                    buf = np.zeros((0,), dtype=np.float32)
                    made_early = False
                    t_start = time.time()
                    await ws.send_text(json.dumps({"type":"ack","msg":"reset"}))
                else:
                    await ws.send_text(json.dumps({"type":"ack","msg":"ok"}))

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.close(code=1011, reason=str(e))
        except Exception:
            pass

async def send_decision(ws: WebSocket, audio: np.ndarray, t_start: float, tag: str):
    # Transcribe the *entire provided audio* (could be early 2s or full buffer on flush)
    segments, info = whisper.transcribe(
        audio, language="en", beam_size=1, condition_on_previous_text=False
    )
    text = "".join([s.text for s in segments]).strip()
    text = normalize_text(text)

    label, conf, proba_h = classify_text(text)

    payload = {
        "type": tag,
        "label": label,
        "confidence": conf,
        "proba_human": float(proba_h),
        "transcript": text,
        "elapsed_ms": int((time.time()-t_start)*1000)
    }
    await ws.send_text(json.dumps(payload))
