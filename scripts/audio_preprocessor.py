#!/usr/bin/env python3
"""
audio_preprocessor.py - Prepare audio files for voicemail detection training

This script preprocesses audio files by:
1. Loading audio in various formats (WAV, MP3, FLAC, M4A, OGG)
2. Converting to mono
3. Resampling to 16kHz
4. Clipping or padding to 2 seconds
5. Normalizing amplitude
6. Saving as WAV

Usage:
    python scripts/audio_preprocessor.py --input_dir /path/to/raw --output_dir /path/to/processed
    python scripts/audio_preprocessor.py --input_file /path/to/file.mp3 --output_file /path/to/output.wav
"""
import os
import argparse
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Constants
TARGET_SR = 16000
TARGET_DURATION = 2.0  # seconds
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)
SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus', '.wma'}


def preprocess_audio(input_path: str, output_path: str = None,
                     target_sr: int = TARGET_SR,
                     target_duration: float = TARGET_DURATION,
                     normalize: bool = True) -> tuple:
    """
    Load, resample, convert to mono, and clip/pad audio to target duration.

    Args:
        input_path: Path to input audio file
        output_path: Path to save processed audio (optional)
        target_sr: Target sample rate (default: 16000)
        target_duration: Target duration in seconds (default: 2.0)
        normalize: Whether to normalize amplitude (default: True)

    Returns:
        tuple: (audio_array, sample_rate, success, error_message)
    """
    try:
        # Load audio with librosa (handles most formats)
        audio, sr = librosa.load(input_path, sr=target_sr, mono=True)

        # Calculate target samples
        target_samples = int(target_sr * target_duration)

        # Clip or pad to target duration
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')

        # Normalize amplitude
        if normalize:
            max_amp = np.max(np.abs(audio))
            if max_amp > 0:
                audio = audio / max_amp * 0.95

        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            sf.write(output_path, audio, target_sr)

        return audio, target_sr, True, None

    except Exception as e:
        return None, None, False, str(e)


def batch_preprocess(input_dir: str, output_dir: str,
                     target_sr: int = TARGET_SR,
                     target_duration: float = TARGET_DURATION,
                     max_workers: int = 4,
                     verbose: bool = True) -> dict:
    """
    Process all audio files in a directory.

    Args:
        input_dir: Input directory containing audio files
        output_dir: Output directory for processed files
        target_sr: Target sample rate
        target_duration: Target duration in seconds
        max_workers: Number of parallel workers
        verbose: Whether to show progress bar

    Returns:
        dict: Statistics about processing
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(input_path.rglob(f'*{ext}'))
        files.extend(input_path.rglob(f'*{ext.upper()}'))

    files = list(set(files))  # Remove duplicates

    if not files:
        print(f"No audio files found in {input_dir}")
        return {"total": 0, "processed": 0, "failed": 0, "errors": []}

    print(f"Found {len(files)} audio files")

    stats = {
        "total": len(files),
        "processed": 0,
        "failed": 0,
        "errors": []
    }

    def process_file(file):
        # Preserve relative directory structure
        rel_path = file.relative_to(input_path)
        out_file = output_path / rel_path.with_suffix('.wav')

        _, _, success, error = preprocess_audio(
            str(file), str(out_file), target_sr, target_duration
        )

        return file, success, error

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in files}

        iterator = as_completed(futures)
        if verbose:
            iterator = tqdm(iterator, total=len(files), desc="Processing")

        for future in iterator:
            file, success, error = future.result()
            if success:
                stats["processed"] += 1
            else:
                stats["failed"] += 1
                stats["errors"].append({"file": str(file), "error": error})

    print(f"\nProcessed: {stats['processed']}, Failed: {stats['failed']}")

    return stats


def analyze_audio(audio_dir: str, sample_size: int = 100) -> dict:
    """
    Analyze audio files in a directory to understand their properties.

    Args:
        audio_dir: Directory containing audio files
        sample_size: Number of files to sample for analysis

    Returns:
        dict: Analysis results
    """
    audio_path = Path(audio_dir)

    # Find all audio files
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(audio_path.rglob(f'*{ext}'))

    files = list(set(files))[:sample_size]

    if not files:
        return {"error": "No audio files found"}

    durations = []
    sample_rates = []
    channels = []

    for file in tqdm(files, desc="Analyzing"):
        try:
            info = sf.info(str(file))
            durations.append(info.duration)
            sample_rates.append(info.samplerate)
            channels.append(info.channels)
        except:
            try:
                # Fallback to librosa
                y, sr = librosa.load(str(file), sr=None, mono=False)
                durations.append(len(y) / sr if y.ndim == 1 else len(y[0]) / sr)
                sample_rates.append(sr)
                channels.append(1 if y.ndim == 1 else y.shape[0])
            except Exception as e:
                print(f"Could not analyze {file}: {e}")

    return {
        "files_analyzed": len(durations),
        "duration": {
            "min": min(durations) if durations else 0,
            "max": max(durations) if durations else 0,
            "mean": np.mean(durations) if durations else 0,
            "median": np.median(durations) if durations else 0
        },
        "sample_rates": list(set(sample_rates)),
        "channels": list(set(channels))
    }


def create_csv_manifest(audio_dir: str, output_csv: str,
                        include_transcription: bool = False) -> None:
    """
    Create a CSV manifest of audio files.

    Args:
        audio_dir: Directory containing audio files
        output_csv: Output CSV path
        include_transcription: Whether to include empty transcription column
    """
    import csv

    audio_path = Path(audio_dir)

    # Find all audio files
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(audio_path.rglob(f'*{ext}'))

    files = sorted(set(files))

    with open(output_csv, 'w', newline='') as f:
        if include_transcription:
            writer = csv.writer(f)
            writer.writerow(['path', 'transcription'])
            for file in files:
                writer.writerow([str(file.absolute()), ''])
        else:
            writer = csv.writer(f)
            writer.writerow(['path', 'label'])
            for file in files:
                writer.writerow([str(file.absolute()), ''])

    print(f"Created manifest with {len(files)} files: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess audio files for voicemail detection training'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Batch processing
    batch_parser = subparsers.add_parser('batch', help='Batch process directory')
    batch_parser.add_argument('--input_dir', required=True, help='Input directory')
    batch_parser.add_argument('--output_dir', required=True, help='Output directory')
    batch_parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    batch_parser.add_argument('--duration', type=float, default=2.0, help='Target duration (seconds)')
    batch_parser.add_argument('--sample_rate', type=int, default=16000, help='Target sample rate')

    # Single file processing
    single_parser = subparsers.add_parser('single', help='Process single file')
    single_parser.add_argument('--input_file', required=True, help='Input file')
    single_parser.add_argument('--output_file', required=True, help='Output file')
    single_parser.add_argument('--duration', type=float, default=2.0, help='Target duration (seconds)')
    single_parser.add_argument('--sample_rate', type=int, default=16000, help='Target sample rate')

    # Analysis
    analyze_parser = subparsers.add_parser('analyze', help='Analyze audio directory')
    analyze_parser.add_argument('--input_dir', required=True, help='Input directory')
    analyze_parser.add_argument('--sample_size', type=int, default=100, help='Files to sample')

    # Create manifest
    manifest_parser = subparsers.add_parser('manifest', help='Create CSV manifest')
    manifest_parser.add_argument('--input_dir', required=True, help='Input directory')
    manifest_parser.add_argument('--output_csv', required=True, help='Output CSV path')
    manifest_parser.add_argument('--for_asr', action='store_true', help='Include transcription column')

    args = parser.parse_args()

    if args.command == 'batch':
        batch_preprocess(
            args.input_dir, args.output_dir,
            target_sr=args.sample_rate,
            target_duration=args.duration,
            max_workers=args.workers
        )

    elif args.command == 'single':
        _, _, success, error = preprocess_audio(
            args.input_file, args.output_file,
            target_sr=args.sample_rate,
            target_duration=args.duration
        )
        if success:
            print(f"Successfully processed: {args.output_file}")
        else:
            print(f"Error: {error}")

    elif args.command == 'analyze':
        results = analyze_audio(args.input_dir, args.sample_size)
        print("\nAudio Analysis Results:")
        print(f"  Files analyzed: {results['files_analyzed']}")
        print(f"  Duration (seconds):")
        print(f"    Min: {results['duration']['min']:.2f}")
        print(f"    Max: {results['duration']['max']:.2f}")
        print(f"    Mean: {results['duration']['mean']:.2f}")
        print(f"    Median: {results['duration']['median']:.2f}")
        print(f"  Sample rates: {results['sample_rates']}")
        print(f"  Channels: {results['channels']}")

    elif args.command == 'manifest':
        create_csv_manifest(args.input_dir, args.output_csv, args.for_asr)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
