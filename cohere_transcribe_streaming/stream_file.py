"""Print live captions and a final transcript for an audio file.

Run: python stream_file.py speech.wav --language en --lookahead 6 --realtime
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import soundfile as sf

from model import DEFAULT_CHECKPOINT, CohereStreaming

SAMPLE_RATE = 16000


def load_audio(path: str) -> np.ndarray:
    """Read an audio file as 16 kHz mono float32."""
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(-1)
    if sample_rate != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return audio


def stream_audio(model: CohereStreaming, audio: np.ndarray, language: str = "en", realtime: bool = False) -> str:
    """Print caption changes while feeding the audio, then return the final transcript."""
    model.reset(language=language)
    chunk_samples = SAMPLE_RATE * model.hop_ms // 1000
    previous_text = None
    start_time = time.perf_counter()
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        audio_time = (start + chunk_samples) / SAMPLE_RATE

        text = model.step(chunk)
        if text != previous_text:
            print(f"[{audio_time:6.2f}s] {text}", flush=True)
            previous_text = text

        if realtime:
            elapsed = time.perf_counter() - start_time
            time.sleep(max(0.0, audio_time - elapsed))

    finish_start = time.perf_counter()
    final_text = model.finish()
    finish_ms = (time.perf_counter() - finish_start) * 1000
    print(f"\nFINAL ({finish_ms:.0f} ms after end of audio): {final_text}")
    return final_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print live captions for an audio file.")
    parser.add_argument("audio", help="path to an audio file")
    parser.add_argument("--language", default="en", help="en, fr, de, es, it, pt, nl, pl")
    parser.add_argument("--lookahead", type=int, default=6, choices=[0, 1, 3, 6, 13])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--realtime", action="store_true", help="feed the audio at real-time speed")
    return parser.parse_args()


def main():
    args = parse_args()
    audio = load_audio(args.audio)
    model = CohereStreaming(args.checkpoint, lookahead=args.lookahead)
    stream_audio(model, audio, language=args.language, realtime=args.realtime)


if __name__ == "__main__":
    main()
