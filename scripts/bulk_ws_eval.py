#!/usr/bin/env python3
import argparse, asyncio, json, os, sys, csv, time, glob, pathlib
import numpy as np
import soundfile as sf
import librosa
import websockets

SAMPLE_RATE = 16000

def list_audio_files(root: str, patterns: list[str]) -> list[str]:
    exts = patterns or ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"]
    files = []
    for ext in exts:
        files.extend(pathlib.Path(root).rglob(ext))
    return sorted(map(str, files))

def load_audio_to_pcm16_chunks(path: str, chunk_ms: int = 20):
    """Decode any audio -> 16k mono PCM16, yield bytes chunks of ~chunk_ms."""
    # decode
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    except Exception:
        # fallback decoder
        audio, sr = librosa.load(path, sr=None, mono=True)
        audio = audio.astype(np.float32)

    # resample
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)

    # convert to int16
    audio = np.clip(audio, -1.0, 1.0)
    int16 = (audio * 32768.0).astype(np.int16)

    # chunking
    frames_per_chunk = int(SAMPLE_RATE * (chunk_ms/1000.0))
    pos = 0
    n = len(int16)
    while pos < n:
        end = min(n, pos + frames_per_chunk)
        yield int16[pos:end].tobytes()
        pos = end

async def stream_one(url: str, path: str, chunk_ms: int, timeout_s: float):
    """Stream one file; return dict for CSV row. Always attempts to return a result."""
    start = time.time()
    last_msg = None
    try:
        async with websockets.connect(url, max_size=2**23, ping_interval=None) as ws:
            # read responses concurrently
            async def receiver():
                nonlocal last_msg
                try:
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                        except Exception:
                            continue
                        # keep the latest message (prefer final later outside)
                        last_msg = data
                except Exception:
                    pass

            recv_task = asyncio.create_task(receiver())

            # send audio chunks
            for chunk in load_audio_to_pcm16_chunks(path, chunk_ms=chunk_ms):
                await ws.send(chunk)

            # request final decision (full transcript)
            await ws.send(json.dumps({"type": "flush"}))

            # wait until we see a "final" OR timeout
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                await asyncio.sleep(0.05)
                if isinstance(last_msg, dict) and last_msg.get("type") == "final":
                    break

            try:
                await ws.close()
            except Exception:
                pass
            await asyncio.wait_for(recv_task, timeout=1.0)
    except Exception as e:
        return {
            "file": path,
            "type": "error",
            "label": "",
            "confidence": "",
            "proba_human": "",
            "transcript": f"[error] {e}",
            "elapsed_ms": int((time.time()-start)*1000),
        }

    # prefer final; else early; else empty
    data = last_msg or {}
    return {
        "file": path,
        "type": data.get("type", "unknown"),
        "label": data.get("label", ""),
        "confidence": data.get("confidence", ""),
        "proba_human": data.get("proba_human", ""),
        "transcript": data.get("transcript", ""),
        "elapsed_ms": data.get("elapsed_ms", int((time.time()-start)*1000)),
    }

def load_done(csv_path: str) -> set[str]:
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "file" not in reader.fieldnames:
            return done
        for row in reader:
            fp = row.get("file")
            if fp:
                done.add(fp)
    return done

async def producer_consumer(url, files, csv_path, chunk_ms, concurrency, timeout_s):
    # open CSV and write header if new
    new_file = not os.path.exists(csv_path)
    f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=["file","type","label","confidence","proba_human","transcript","elapsed_ms"])
    if new_file:
        writer.writeheader()
        f.flush()

    sem = asyncio.Semaphore(concurrency)
    total = len(files)
    done_count = 0

    async def one(path):
        nonlocal done_count
        async with sem:
            res = await stream_one(url, path, chunk_ms, timeout_s)
            writer.writerow(res)
            f.flush()
            done_count += 1
            if done_count % 100 == 0:
                print(f"[{done_count}/{total}] last: {os.path.basename(path)} - {res.get('type')}/{res.get('label')}")

    tasks = [asyncio.create_task(one(p)) for p in files]
    await asyncio.gather(*tasks)
    f.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True, help="Root folder containing audio files (recursive).")
    ap.add_argument("--csv_out", required=True, help="Output CSV path.")
    ap.add_argument("--url", default="ws://127.0.0.1:8080/ws/amd")
    ap.add_argument("--chunk_ms", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=4, help="Number of parallel WS connections.")
    ap.add_argument("--timeout_s", type=float, default=10.0, help="Max seconds to wait for 'final' per file.")
    ap.add_argument("--patterns", nargs="*", default=None, help='Glob patterns, e.g. *.wav *.flac (default: common audio types)')
    ap.add_argument("--resume", action="store_true", help="Skip files that already appear in CSV.")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of files (debug).")
    args = ap.parse_args()

    files = list_audio_files(args.audio_dir, args.patterns)
    if args.resume and os.path.exists(args.csv_out):
        done = load_done(args.csv_out)
        before = len(files)
        files = [p for p in files if p not in done]
        print(f"[resume] skipping {before - len(files)} already in {args.csv_out}")

    if args.limit and args.limit > 0:
        files = files[:args.limit]

    print(f"[start] files to process: {len(files)}  |  concurrency={args.concurrency}  url={args.url}")
    if not files:
        print("Nothing to do.")
        return

    asyncio.run(producer_consumer(
        url=args.url,
        files=files,
        csv_path=args.csv_out,
        chunk_ms=args.chunk_ms,
        concurrency=args.concurrency,
        timeout_s=args.timeout_s,
    ))

if __name__ == "__main__":
    main()
