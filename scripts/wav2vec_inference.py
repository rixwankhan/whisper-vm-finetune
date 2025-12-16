#!/usr/bin/env python3
"""
wav2vec_inference.py - Inference with wav2vec-vm-finetune model

This script provides voicemail detection using the wav2vec2 model from:
https://huggingface.co/jakeBland/wav2vec-vm-finetune

Usage:
    # Single file inference
    python scripts/wav2vec_inference.py --audio /path/to/audio.wav

    # Batch inference
    python scripts/wav2vec_inference.py --audio_dir /path/to/folder --output results.csv

    # Download model locally
    python scripts/wav2vec_inference.py --download --model_dir models/wav2vec-vm
"""
import os
import argparse
import time
import json
import csv
from pathlib import Path
from typing import Union, List, Dict, Optional

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

try:
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
except ImportError:
    print("Please install transformers: pip install transformers")
    exit(1)


class Wav2VecVoicemailDetector:
    """
    Voicemail detection using wav2vec2 fine-tuned model.

    This uses the jakeBland/wav2vec-vm-finetune model which is trained
    to detect voicemail vs. live human from the first 2 seconds of audio.
    """

    def __init__(self,
                 model_path: str = "jakeBland/wav2vec-vm-finetune",
                 device: Optional[str] = None,
                 threshold: float = 0.5):
        """
        Initialize the voicemail detector.

        Args:
            model_path: HuggingFace model name or local path
            device: 'cuda', 'cpu', or None for auto-detection
            threshold: Classification threshold (default 0.5)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.sample_rate = 16000
        self.max_duration = 2.0  # seconds
        self.max_samples = int(self.sample_rate * self.max_duration)

        print(f"Loading model from {model_path}...")
        print(f"Using device: {self.device}")

        try:
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTrying to load from HuggingFace Hub...")
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                "jakeBland/wav2vec-vm-finetune"
            )
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "jakeBland/wav2vec-vm-finetune"
            )

        self.model.to(self.device)
        self.model.eval()

        # Determine label mapping
        self._setup_label_mapping()

        print("Model loaded successfully!")

    def _setup_label_mapping(self):
        """Set up label mapping from model config or defaults."""
        if hasattr(self.model.config, 'id2label'):
            self.id2label = self.model.config.id2label
        else:
            # Default mapping (verify with actual model)
            self.id2label = {0: "human", 1: "machine"}

        self.label2id = {v: k for k, v in self.id2label.items()}
        print(f"Label mapping: {self.id2label}")

    def preprocess_audio_file(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Preprocessed audio as numpy array, or None if error
        """
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to 16kHz if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # Clip or pad to max duration
            waveform = waveform.squeeze()
            if len(waveform) > self.max_samples:
                waveform = waveform[:self.max_samples]
            elif len(waveform) < self.max_samples:
                waveform = torch.nn.functional.pad(
                    waveform, (0, self.max_samples - len(waveform))
                )

            return waveform.numpy()

        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None

    def preprocess_audio_bytes(self, audio_bytes: bytes) -> np.ndarray:
        """
        Preprocess audio from PCM16 bytes.

        Args:
            audio_bytes: PCM16 16kHz mono audio bytes

        Returns:
            Preprocessed audio as numpy array
        """
        # Convert PCM16 to float32
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Clip or pad to max duration
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        elif len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))

        return audio

    def predict(self,
                audio: Union[str, np.ndarray, bytes],
                return_probs: bool = True) -> Dict:
        """
        Predict whether audio is human or machine (voicemail).

        Args:
            audio: Audio file path, numpy array, or PCM16 bytes
            return_probs: Whether to return probabilities

        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()

        # Preprocess audio based on input type
        if isinstance(audio, str):
            audio_array = self.preprocess_audio_file(audio)
            if audio_array is None:
                return {"error": "Failed to load audio", "label": "unknown"}
        elif isinstance(audio, bytes):
            audio_array = self.preprocess_audio_bytes(audio)
        else:
            audio_array = audio

        # Extract features
        inputs = self.feature_extractor(
            audio_array,
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
        probs_np = probs.cpu().numpy()[0]
        predicted_class = int(np.argmax(probs_np))
        predicted_label = self.id2label[predicted_class]
        confidence = float(probs_np[predicted_class])

        elapsed_ms = int((time.time() - start_time) * 1000)

        result = {
            "label": predicted_label,
            "confidence": confidence,
            "elapsed_ms": elapsed_ms
        }

        if return_probs:
            for i, label in self.id2label.items():
                result[f"proba_{label}"] = float(probs_np[i])

        return result

    def predict_batch(self,
                      audio_list: List[Union[str, np.ndarray]],
                      batch_size: int = 8) -> List[Dict]:
        """
        Predict on multiple audio samples.

        Args:
            audio_list: List of audio file paths or numpy arrays
            batch_size: Batch size for processing

        Returns:
            List of prediction dictionaries
        """
        results = []

        for i in range(0, len(audio_list), batch_size):
            batch = audio_list[i:i+batch_size]

            # Preprocess batch
            audio_arrays = []
            batch_indices = []

            for j, audio in enumerate(batch):
                if isinstance(audio, str):
                    arr = self.preprocess_audio_file(audio)
                else:
                    arr = audio

                if arr is not None:
                    audio_arrays.append(arr)
                    batch_indices.append(i + j)
                else:
                    results.append({"error": "Failed to load audio", "label": "unknown"})

            if not audio_arrays:
                continue

            # Extract features
            inputs = self.feature_extractor(
                audio_arrays,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            probs_np = probs.cpu().numpy()

            for j, prob in enumerate(probs_np):
                predicted_class = int(np.argmax(prob))
                results.append({
                    "label": self.id2label[predicted_class],
                    "confidence": float(prob[predicted_class]),
                    "proba_human": float(prob[self.label2id.get("human", 0)]),
                    "proba_machine": float(prob[self.label2id.get("machine", 1)])
                })

        return results

    def save_model(self, output_dir: str):
        """
        Save model locally.

        Args:
            output_dir: Directory to save model
        """
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.feature_extractor.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")


def download_model(model_dir: str):
    """Download and save model locally."""
    print("Downloading model from HuggingFace...")
    detector = Wav2VecVoicemailDetector()
    detector.save_model(model_dir)
    print(f"Model downloaded and saved to {model_dir}")


def process_directory(audio_dir: str, output_csv: str,
                      model_path: str = "jakeBland/wav2vec-vm-finetune",
                      device: str = None):
    """Process all audio files in a directory."""
    audio_path = Path(audio_dir)

    # Find all audio files
    extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    files = []
    for ext in extensions:
        files.extend(audio_path.rglob(f'*{ext}'))
        files.extend(audio_path.rglob(f'*{ext.upper()}'))

    files = sorted(set(files))

    if not files:
        print(f"No audio files found in {audio_dir}")
        return

    print(f"Found {len(files)} audio files")

    # Initialize detector
    detector = Wav2VecVoicemailDetector(model_path, device)

    # Process files
    results = []
    for file in tqdm(files, desc="Processing"):
        result = detector.predict(str(file))
        result["file"] = str(file)
        results.append(result)

    # Save results
    with open(output_csv, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"Results saved to {output_csv}")

    # Print summary
    human_count = sum(1 for r in results if r.get("label") == "human")
    machine_count = sum(1 for r in results if r.get("label") == "machine")
    print(f"\nSummary: {human_count} human, {machine_count} machine")


def main():
    parser = argparse.ArgumentParser(
        description='Voicemail detection using wav2vec2 model'
    )

    parser.add_argument('--audio', type=str, help='Path to single audio file')
    parser.add_argument('--audio_dir', type=str, help='Directory of audio files')
    parser.add_argument('--output', type=str, default='results.csv',
                        help='Output CSV path for batch processing')
    parser.add_argument('--model', type=str, default='jakeBland/wav2vec-vm-finetune',
                        help='Model path or HuggingFace model name')
    parser.add_argument('--model_dir', type=str, default='models/wav2vec-vm',
                        help='Directory to save model when downloading')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        help='Device to use (auto-detect if not specified)')
    parser.add_argument('--download', action='store_true',
                        help='Download model to local directory')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')

    args = parser.parse_args()

    # Download mode
    if args.download:
        download_model(args.model_dir)
        return

    # Single file mode
    if args.audio:
        detector = Wav2VecVoicemailDetector(args.model, args.device, args.threshold)
        result = detector.predict(args.audio)

        print(f"\nResult for: {args.audio}")
        print(f"  Label: {result['label']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        if 'proba_human' in result:
            print(f"  Human probability: {result['proba_human']:.2%}")
            print(f"  Machine probability: {result['proba_machine']:.2%}")
        print(f"  Processing time: {result['elapsed_ms']}ms")
        return

    # Batch mode
    if args.audio_dir:
        process_directory(args.audio_dir, args.output, args.model, args.device)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
