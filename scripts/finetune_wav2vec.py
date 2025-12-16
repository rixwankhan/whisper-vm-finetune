#!/usr/bin/env python3
"""
finetune_wav2vec.py - Fine-tune wav2vec2 for voicemail detection

Fine-tune facebook/wav2vec2-xls-r-300m (or other wav2vec2 models) for
binary classification of voicemail vs. live human audio.

Data Format:
    CSV file with columns:
    - path: Absolute path to audio file
    - label: 'human' or 'machine'

Usage:
    python scripts/finetune_wav2vec.py \
        --csv /path/to/training_data.csv \
        --out_dir models/wav2vec-vm-custom \
        --batch_size 8 \
        --epochs 10

    # Resume from checkpoint
    python scripts/finetune_wav2vec.py \
        --csv /path/to/training_data.csv \
        --out_dir models/wav2vec-vm-custom \
        --resume models/wav2vec-vm-custom/checkpoint-1000
"""
import os
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Union, Optional

import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from datasets import Dataset, DatasetDict
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
SAMPLE_RATE = 16000
MAX_DURATION = 2.0  # seconds
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION)

# Label mapping
LABEL_MAP = {"human": 0, "machine": 1}
ID2LABEL = {0: "human", 1: "machine"}


def load_audio(path: str) -> Optional[np.ndarray]:
    """
    Load and preprocess audio file.

    Args:
        path: Path to audio file

    Returns:
        Preprocessed audio as numpy array, or None if error
    """
    try:
        waveform, sr = torchaudio.load(path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz
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


def prepare_dataset(csv_path: str, test_size: float = 0.1) -> DatasetDict:
    """
    Load CSV and prepare train/eval datasets.

    Args:
        csv_path: Path to CSV file with columns: path, label
        test_size: Fraction for evaluation set

    Returns:
        DatasetDict with train and test splits
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Validate columns
    required_cols = ['path', 'label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must have '{col}' column")

    # Validate labels
    invalid_labels = df[~df['label'].isin(LABEL_MAP.keys())]['label'].unique()
    if len(invalid_labels) > 0:
        raise ValueError(f"Invalid labels found: {invalid_labels}. Must be 'human' or 'machine'")

    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # Convert labels to integers
    df['label'] = df['label'].map(LABEL_MAP)

    # Load audio
    print("\nLoading audio files...")
    audio_data = []
    labels = []
    valid_paths = []

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing {idx}/{len(df)}...")

        audio = load_audio(row['path'])
        if audio is not None:
            audio_data.append(audio)
            labels.append(row['label'])
            valid_paths.append(row['path'])

    print(f"\nSuccessfully loaded {len(audio_data)}/{len(df)} files")

    # Create dataset
    dataset = Dataset.from_dict({
        "audio": audio_data,
        "label": labels,
        "path": valid_paths
    })

    # Split
    print(f"\nSplitting data (test_size={test_size})...")
    dataset = dataset.train_test_split(test_size=test_size, seed=42, stratify_by_column="label")

    print(f"Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")

    return dataset


@dataclass
class DataCollatorForWav2Vec:
    """Data collator for wav2vec2 sequence classification."""

    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[float], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract audio arrays
        audio_arrays = [f["audio"] for f in features]
        labels = [f["label"] for f in features]

        # Process with feature extractor
        batch = self.feature_extractor(
            audio_arrays,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=self.padding
        )

        batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_human": precision_per_class[0],
        "recall_human": recall_per_class[0],
        "f1_human": f1_per_class[0],
        "precision_machine": precision_per_class[1],
        "recall_machine": recall_per_class[1],
        "f1_machine": f1_per_class[1],
    }


def main():
    parser = argparse.ArgumentParser(description='Fine-tune wav2vec2 for voicemail detection')

    # Data arguments
    parser.add_argument('--csv', required=True, help='Path to training CSV (path,label)')
    parser.add_argument('--test_size', type=float, default=0.1, help='Test split size')

    # Model arguments
    parser.add_argument('--base_model', default='facebook/wav2vec2-xls-r-300m',
                        help='Base model to fine-tune')
    parser.add_argument('--out_dir', required=True, help='Output directory for model')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Evaluation batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--gradient_accumulation', type=int, default=2,
                        help='Gradient accumulation steps')

    # Other arguments
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze feature encoder (recommended)')
    parser.add_argument('--early_stopping', type=int, default=3,
                        help='Early stopping patience (0 to disable)')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loading workers')

    args = parser.parse_args()

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Prepare dataset
    dataset = prepare_dataset(args.csv, args.test_size)

    # Load model and feature extractor
    print(f"\nLoading base model: {args.base_model}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.base_model)

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        label2id=LABEL_MAP,
        id2label=ID2LABEL,
        problem_type="single_label_classification"
    )

    # Freeze feature encoder if requested
    if args.freeze_encoder:
        print("Freezing feature encoder...")
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
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=50,
        logging_dir=os.path.join(args.out_dir, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=args.num_workers,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
    )

    # Callbacks
    callbacks = []
    if args.early_stopping > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping))

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Base model: {args.base_model}")
    print(f"Output dir: {args.out_dir}")
    print(f"Batch size: {args.batch_size} (effective: {args.batch_size * args.gradient_accumulation})")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    print("="*60 + "\n")

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # Evaluate
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Save model
    print(f"\nSaving model to {args.out_dir}")
    trainer.save_model(args.out_dir)
    feature_extractor.save_pretrained(args.out_dir)

    # Save training info
    info = {
        "base_model": args.base_model,
        "train_samples": len(dataset["train"]),
        "eval_samples": len(dataset["test"]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "freeze_encoder": args.freeze_encoder,
        "final_metrics": {k: float(v) if isinstance(v, (int, float)) else v
                        for k, v in eval_results.items()}
    }

    import json
    with open(os.path.join(args.out_dir, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved to: {args.out_dir}")
    print(f"Final accuracy: {eval_results['eval_accuracy']:.2%}")
    print("\nTo use the model:")
    print(f"  python scripts/wav2vec_inference.py --audio test.wav --model {args.out_dir}")


if __name__ == "__main__":
    main()
