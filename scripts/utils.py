\
import re, io, os, numpy as np, soundfile as sf
from pydub import AudioSegment

DEFAULT_SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
MAX_SEC = 2.0
N_SAMPLES = int(DEFAULT_SAMPLE_RATE*MAX_SEC)

MACHINE_KEYWORDS = [
    "pound key","press pound","record","please leave","leave a message",
    "can't come","can't get","forwarded","sorry i miss","sorry i missed",
    "we are not","not available","we're unable","we are unable",
    "can't take","can't answer","voice mail","voicemail","leave your name",
    "after the tone","at the tone","beep"
]
HUMAN_KEYWORDS = [
    "speaking","this is","how can i help","yeah","yes","okay","just a sec"
]
IGNORE_WORDS = ["hi","hello"]  # ignored globally

def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = t.lower().strip()
    # remove ignore words as isolated tokens
    for w in IGNORE_WORDS:
        t = re.sub(rf"\b{re.escape(w)}\b", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def keyword_hits(text: str, vocab) -> list[str]:
    hits = []
    for kw in vocab:
        if kw in text:
            hits.append(kw)
    return hits

def wav_bytes_to_float32(samples_bytes: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Decode linear PCM16 bytes to float32 [-1,1]."""
    arr = np.frombuffer(samples_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return arr

def ensure_2s(buf: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    need = int(sample_rate*MAX_SEC)
    if buf.shape[0] >= need:
        return buf[:need]
    out = np.zeros((need,), dtype=np.float32)
    out[:buf.shape[0]] = buf
    return out
