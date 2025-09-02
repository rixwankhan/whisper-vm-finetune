\
import argparse, asyncio, websockets, wave, json, sys

def read_wav_pcm16(path, chunk_ms=20):
    with wave.open(path, 'rb') as wf:
        assert wf.getsampwidth() == 2, "Expect 16-bit PCM WAV"
        sr = wf.getframerate()
        assert sr == 16000, f"Expect 16kHz, got {sr}"
        ch = wf.getnchannels()
        assert ch == 1, f"Expect mono, got {ch}"

        frames_per_chunk = int(sr * (chunk_ms/1000.0))
        while True:
            data = wf.readframes(frames_per_chunk)
            if not data:
                break
            yield data

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8080/ws/amd")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--chunk_ms", type=int, default=20)
    args = ap.parse_args()

    async with websockets.connect(args.url, max_size=2**23) as ws:
        async def receiver():
            try:
                async for msg in ws:
                    print("SERVER:", msg)
            except Exception as e:
                print("Receiver ended:", e, file=sys.stderr)

        recv_task = asyncio.create_task(receiver())

        for chunk in read_wav_pcm16(args.wav, args.chunk_ms):
            await ws.send(chunk)
            await asyncio.sleep(args.chunk_ms/1000.0)

        await ws.send(json.dumps({"type":"flush"}))
        await asyncio.sleep(1.0)
        await ws.close()
        await recv_task

if __name__ == "__main__":
    asyncio.run(main())
