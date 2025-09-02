# AI-Driven Answering Machine Detection (AMD) with Whisper

Fast, production-ready AMD for outbound calling and IVR workflows using **Whisper-small** (with fine-tuned) for STT and a **supervised text classifier** for **Human vs Machine** detection.

## What this project does

- **2-second early decision**: buffer the first 2.0s of audio and emit an early label.
- **Full-transcript decision**: when the client sends `{"type":"flush"}`, transcribe the entire buffered audio and classify the complete transcript.
- **Whisper fine-tuning**: train Whisper-small on your own `path,transcription` CSV, clipped to 2s @ 16kHz for AMD speed.
- **Text classifier**: TF-IDF (word+char n-grams) + Logistic Regression, trained on `text,label` (`human|machine`). No hard-coded keyword rules: phrases like “leave a” are learned directly from data.
- **Low-latency inference**: export the fine-tuned model to **CTranslate2** for fast GPU/CPU decoding.
- **Live streaming**: WebSocket server receives **PCM16 16kHz mono** frames; client example streams audio and prints decisions.
- **Bulk evaluation**: script to batch 100k+ audios and export results to CSV.

## Who this is for

- **Dialers / Contact centers / CPaaS** engineers who need **robust AMD** without acoustic heuristics.
- **Telephony/IVR** teams integrating with FreeSWITCH, Asterisk, or WebRTC gateways.
- **Product teams** wanting a transparent, trainable AMD pipeline based on **ASR + text classification**.

---

## Repo layout

```
server/                 # WebSocket server (FastAPI + faster-whisper/CT2)
client/                 # Example streaming client
scripts/
  finetune_whisper_small.py   # fine-tune Whisper from CSV
  train_text_classifier.py    # train text classifier from CSV
  bulk_ws_eval.py             # stream a folder to the WS and save results to CSV
  utils.py                    # helpers (normalization, audio utils)
models/                 # fine-tuned and converted models live here
manifests/              # CSVs, logs, bad rows, batch results (created at runtime)
README.md               # this file
requirements.txt
```

---

## Setup & Installation

**Prereqs**
- Python 3.10+
- (Optional) **GPU** with CUDA 12 and **cuDNN 9** for best performance
- System decoders: `libsndfile` (installed via pip `soundfile`) and `ffmpeg` recommended

**Install**
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU runtime**: ensure CUDA 12 + cuDNN 9 are installed. If you see `libcudnn_ops.so.9` errors, install `libcudnn9-cuda-12` from the NVIDIA repo.

---

## Data formats

- **ASR fine-tune CSV**: `path,transcription` (or `path,text`)
- **Classifier CSV**: `text,label` where `label ∈ {human,machine}`

Audio should be readable by `soundfile`/`librosa`; sample rate is normalized to **16kHz mono** and clipped/padded to **2.0s** for training.

---

## Training

### 1) Fine-tune Whisper-small
```bash
python scripts/finetune_whisper_small.py   --csv /absolute/path/to/data_for_whisper.csv   --out_dir models/whisper-small-finetuned   --batch 4   --num_workers 4
```

- The trainer uses **lazy transforms** to avoid OOM with large datasets (100k+ files ok).
- On completion, you’ll see `[DONE] Saved to: .../models/whisper-small-finetuned`.

**Export to CTranslate2 for fast inference**
```bash
ct2-transformers-converter   --model models/whisper-small-finetuned   --output_dir models/whisper-ct2   --copy_files tokenizer.json tokenizer_config.json preprocessor_config.json                special_tokens_map.json vocab.json merges.txt normalizer.json generation_config.json   --quantization float16      # use int8_float16 for CPU-only
```

### 2) Train the text classifier
```bash
python scripts/train_text_classifier.py   --csv /absolute/path/to/data_for_text.csv   --out models/text_cls.joblib   --word_ngrams 1,2   --char_ngrams 3,6
```
The script prints precision/recall/F1 on a held-out split and writes `models/text_cls.joblib`.

---

## Running & Testing

### Start the WebSocket server
```bash
export WHISPER_MODEL_DIR=models/whisper-ct2
export DEVICE=cuda                 # or cpu
export COMPUTE_TYPE=float16        # or int8_float16 for CPU
export CLASSIFIER_PATH=models/text_cls.joblib
export EARLY_SEC=2.0               # set 0 to disable early decision

uvicorn server.ws_server:app --host 0.0.0.0 --port 8080
```

**WS endpoint:** `ws://<host>:8080/ws/amd`  
**Audio format:** Binary frames of **PCM16 mono @ 16kHz** (e.g., 20ms per message)

**Client example**
```bash
python client/ws_client.py --wav /path/to/test.wav
```
**Sample response**
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

### Bulk evaluation over a folder
```bash
python scripts/bulk_ws_eval.py   --audio_dir /path/to/folder   --csv_out manifests/bulk_results.csv   --url ws://127.0.0.1:8080/ws/amd   --concurrency 4   --chunk_ms 20   --resume
```
**CSV columns:** `file,type,label,confidence,proba_human,transcript,elapsed_ms`

---

## Production notes

- **Latency**: Use GPU with `compute_type=float16`, `beam_size=1`. Keep chunks at ~20ms.
- **Scaling**: Run multiple Uvicorn workers or multiple pods; front with a WS-capable LB. Each worker loads its own model.
- **Backpressure**: Limit concurrency on the client (e.g., 4–8 streams per worker to start) and use `--resume` in bulk jobs.
- **Reliability**: Add a `/healthz` route, enable structured logs, and log per-call JSON with decision + timings.
- **Thresholding**: If you need higher **machine** recall, adjust the decision threshold in `ws_server.py` (e.g., require `proba_human > 0.55` to call “human”).
- **Security**: Run behind TLS termination; validate origins if exposed publicly.
- **Compatibility**: If GPU libraries aren’t available, deploy **CPU** with `COMPUTE_TYPE=int8_float16` (fast path).

---

## License & data

This template is provided as-is. Ensure you have the rights to the audio you train on and follow applicable telephony/privacy regulations.

---

## Credit

Credit: **Rizwan Khan**
