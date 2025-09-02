# scripts/finetune_whisper_small.py
import argparse, os, sys, pandas as pd, numpy as np, soundfile as sf, librosa, torch
from datasets import Dataset
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)

SAMPLE_RATE = 16000
DUR_SEC = 2.0
N_SAMPLES = int(SAMPLE_RATE * DUR_SEC)

def load_2s(path: str) -> np.ndarray:
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
    except Exception:
        # fallback if soundfile cannot read
        audio, sr = librosa.load(path, sr=None, mono=True)
        audio = audio.astype(np.float32)

    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)

    if len(audio) >= N_SAMPLES:
        audio = audio[:N_SAMPLES]
    else:
        pad = np.zeros(N_SAMPLES, dtype=np.float32)
        pad[: len(audio)] = audio
        audio = pad
    return np.ascontiguousarray(audio, dtype=np.float32)

def make_transform():
    # set_transform can be called on single example or batch; handle both
    def _tx(batch):
        paths = batch["path"] if isinstance(batch["path"], list) else [batch["path"]]
        texts = batch["text"] if isinstance(batch["text"], list) else [batch["text"]]
        # we only attach raw 2s audio; features are made in data_collator
        audios = [load_2s(p) for p in paths]
        out = {
            "audio_2s": audios if len(audios) > 1 else audios[0],
            "text": texts if len(texts) > 1 else texts[0],
            "path": paths if len(paths) > 1 else paths[0],
        }
        return out
    return _tx

class DataCollatorWhisper2s:
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor
        self.tok = processor.tokenizer
        self.feat = processor.feature_extractor

    def __call__(self, features):
        # features contain: {"audio_2s": np.array, "text": str, "path": str}
        audios = [f["audio_2s"] for f in features]
        texts  = [f["text"] for f in features]

        # Whisper input features (80x3000), computed on-the-fly
        # processor.feature_extractor already pads to fixed shape
        inputs = self.feat(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features  # [B, 80, 3000]

        # Tokenize targets; pad, then replace pad_token_id with -100
        labels = self.tok(texts, return_tensors="pt", padding=True).input_ids
        labels[labels == self.tok.pad_token_id] = -100

        return {"input_features": input_features, "labels": labels}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: path,transcription OR path,text")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_model", default="openai/whisper-small")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    out_dir_abs = os.path.abspath(args.out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    print(f"[INFO] Will save to: {out_dir_abs}")

    head = pd.read_csv(args.csv, nrows=1)
    df = pd.read_csv(args.csv)
    if "transcription" in head.columns:
        df = df.rename(columns={"transcription": "text"})
    assert {"path","text"}.issubset(df.columns), "CSV must have columns path and transcription/text"
    df = df.dropna(subset=["path","text"])
    df = df[(df["path"].astype(str).str.len()>0) & (df["text"].astype(str).str.len()>0)]
    print(f"[INFO] Rows: {len(df)}")

    # Tiny, path-only dataset; we transform lazily
    ds = Dataset.from_pandas(df[["path","text"]], preserve_index=False)
    ds = ds.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = ds["train"], ds["test"]

    processor = WhisperProcessor.from_pretrained(args.base_model, language=args.lang, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.lang, task="transcribe")
    model.config.suppress_tokens = []  # better for short clips
    model.gradient_checkpointing_enable()  # less VRAM

    # lazy transforms (no big arrays stored)
    tx = make_transform()
    train_ds.set_transform(tx)
    eval_ds.set_transform(tx)

    collator = DataCollatorWhisper2s(processor)

    train_args = Seq2SeqTrainingArguments(
        output_dir=out_dir_abs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        dataloader_num_workers=args.num_workers,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        predict_with_generate=False,
        fp16=True,
        remove_unused_columns=False,   # keep our custom fields
        report_to=[]
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor.tokenizer,
        data_collator=collator,
    )

    try:
        trainer.train()
    finally:
        print("[INFO] Forcing save...")
        trainer.save_model(out_dir_abs)
        model.save_pretrained(out_dir_abs)
        processor.save_pretrained(out_dir_abs)
        open(os.path.join(out_dir_abs, "SAVE_OK"), "w").write("ok\n")
        print(f"[DONE] Saved to: {out_dir_abs}")

if __name__ == "__main__":
    main()
