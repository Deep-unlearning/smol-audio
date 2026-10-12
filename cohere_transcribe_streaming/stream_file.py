"""Stream an audio file through Cohere Streaming as if it were a live microphone.

The model reads 80 ms of audio at a time and prints the live caption whenever it changes. When the audio ends it
produces a final, more accurate transcript.

    python stream_file.py speech.wav
    python stream_file.py speech.wav --language fr --lookahead 13
    python stream_file.py speech.wav --realtime      # feed the audio at real-time speed

`--lookahead` (R) trades latency for accuracy: the live caption lags by (R + 1) * 80 ms, and R must be one of
0, 1, 3, 6, 13 (80 ms ... 1.12 s). Decoder updates and compute add latency. For utterances within the default
30-second re-encode limit, the final pass uses the same lookahead regardless of R. Longer recordings fall back
to streamed states and can hit the final decode's 256-token budget; reset at speech boundaries.

The whole pattern is three calls, which is all you need to copy into your own code:

    model.reset(language="en")
    text = model.step(audio_80ms)   # call repeatedly, once per 80 ms of audio
    final = model.finish()          # call once, when the audio ends
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import soundfile as sf

from model import CohereStreaming

SAMPLE_RATE = 16000  # the model expects 16 kHz mono audio


def load_audio(path: str) -> np.ndarray:
    """Read an audio file as 16 kHz mono float32."""
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:  # stereo -> mono
        audio = audio.mean(-1)
    if sample_rate != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="path to an audio file")
    parser.add_argument("--language", default="en", help="en, fr, de, es, it, pt, nl, pl")
    parser.add_argument("--lookahead", type=int, default=6, choices=[0, 1, 3, 6, 13])
    parser.add_argument("--checkpoint", default="Steveeeeeeen/cohere-streaming")
    parser.add_argument("--realtime", action="store_true", help="sleep so the audio is fed at real-time speed")
    args = parser.parse_args()

    audio = load_audio(args.audio)
    model = CohereStreaming(args.checkpoint, lookahead=args.lookahead)
    model.reset(language=args.language)

    hop = SAMPLE_RATE * model.hop_ms // 1000  # samples per step: 80 ms
    last_text = None
    start_time = time.perf_counter()
    for start in range(0, len(audio), hop):
        chunk = audio[start : start + hop]
        if len(chunk) < hop:  # pad the final partial hop with silence
            chunk = np.pad(chunk, (0, hop - len(chunk)))
        end = start + hop

        text = model.step(chunk)  # the live caption so far
        if text != last_text:
            print(f"[{end / SAMPLE_RATE:6.2f}s] {text}", flush=True)
            last_text = text

        if args.realtime:  # wait until this much audio would really have been spoken
            time.sleep(max(0.0, end / SAMPLE_RATE - (time.perf_counter() - start_time)))

    finish_start = time.perf_counter()
    final = model.finish()  # the audio ended: get the final transcript
    print(f"\nFINAL ({(time.perf_counter() - finish_start) * 1000:.0f} ms after end of audio): {final}")


if __name__ == "__main__":
    main()
