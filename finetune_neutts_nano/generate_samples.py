"""Generate voice-cloned samples with Neuphonic's official NeuTTS engine.

This script intentionally uses `neutts.NeuTTS(...).infer(...)`, the same
official inference path used for the 100-sample WER comparison. It avoids
manually constructing chat prompts, calling `AutoModelForCausalLM.generate`,
or decoding speech tokens by hand.

Example:
    python generate_samples.py \
        --model-checkpoint ./runs/neutts-official-yodas-300k-5s30s/yodas-300k-5s30s-official-trainer-it-b128-gc \
        --encoded-dataset Steveeeeeeen/yodas-granary-it-neucodec-300k-5s30s \
        --reference-index 0 \
        --language it \
        --prompts "Ciao mondo." "Oggi piove molto." \
        --output-dir ./samples
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from neutts import NeuTTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-checkpoint", required=True, help="Path or Hub id of the fine-tuned model.")
    parser.add_argument("--codec-checkpoint", default="neuphonic/neucodec")
    parser.add_argument("--encoded-dataset", help="Dataset with `text` + `codes` to source the reference voice.")
    parser.add_argument("--reference-split", default="train")
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--reference-audio-path", help="Optional WAV/FLAC reference instead of --encoded-dataset.")
    parser.add_argument("--reference-text", help="Reference transcript. Required with --reference-audio-path.")
    parser.add_argument("--prompts", nargs="+", required=True, help="One or more target texts to synthesize.")
    parser.add_argument("--output-dir", default="./samples")
    parser.add_argument("--language", default="it")
    parser.add_argument("--backbone-device", default="cuda")
    parser.add_argument("--codec-device", default="cuda")
    return parser.parse_args()


def codes_to_text(codes: torch.Tensor) -> str:
    return "".join(f"<|speech_{int(code)}|>" for code in codes.view(-1).tolist())


def audio_stats(wav, sample_rate: int) -> dict:
    arr = np.asarray(wav, dtype=np.float64)
    return {
        "sample_rate": sample_rate,
        "duration_s": float(len(arr) / sample_rate),
        "mean_abs": float(np.mean(np.abs(arr))) if len(arr) else 0.0,
        "rms": float(np.sqrt(np.mean(arr * arr))) if len(arr) else 0.0,
        "peak": float(np.max(np.abs(arr))) if len(arr) else 0.0,
    }


def load_reference(args: argparse.Namespace, tts: NeuTTS, output_dir: Path):
    reference_audio_path = output_dir / "reference.wav"

    if args.encoded_dataset:
        dataset = load_dataset(args.encoded_dataset, split=args.reference_split)
        sample = dataset[args.reference_index]
        ref_text = args.reference_text or sample["text"]
        ref_codes = torch.tensor(sample["codes"], dtype=torch.long).view(-1)
        ref_wav = tts._decode(codes_to_text(ref_codes))
        sf.write(reference_audio_path, ref_wav, tts.sample_rate)
        return ref_codes, ref_text, reference_audio_path, {
            "encoded_dataset": args.encoded_dataset,
            "reference_split": args.reference_split,
            "reference_index": args.reference_index,
        }

    if not args.reference_audio_path or not args.reference_text:
        raise ValueError("Provide --encoded-dataset or --reference-audio-path with --reference-text.")

    shutil.copyfile(args.reference_audio_path, reference_audio_path)
    ref_codes = tts.encode_reference(reference_audio_path)
    return ref_codes, args.reference_text, reference_audio_path, {
        "source_reference_audio_path": args.reference_audio_path,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tts = NeuTTS(
        backbone_repo=args.model_checkpoint,
        backbone_device=args.backbone_device,
        codec_repo=args.codec_checkpoint,
        codec_device=args.codec_device,
        language=args.language,
    )
    ref_codes, ref_text, reference_audio_path, reference_meta = load_reference(args, tts, output_dir)

    results = []
    for idx, prompt in enumerate(args.prompts, start=1):
        with torch.inference_mode():
            wav = tts.infer(prompt, ref_codes, ref_text)
        audio_path = output_dir / f"sample_{idx:02d}.wav"
        sf.write(audio_path, wav, tts.sample_rate)
        item = {
            "engine": "neutts.NeuTTS",
            "model_checkpoint": args.model_checkpoint,
            "codec_checkpoint": args.codec_checkpoint,
            "language": args.language,
            "reference_audio_path": str(reference_audio_path),
            "reference_text": ref_text,
            **reference_meta,
            "prompt": prompt,
            "output_path": str(audio_path),
            "reference_codes": int(len(ref_codes)),
            **audio_stats(wav, tts.sample_rate),
        }
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    metadata = {
        "engine": "neutts.NeuTTS",
        "model_checkpoint": args.model_checkpoint,
        "codec_checkpoint": args.codec_checkpoint,
        "language": args.language,
        "reference_audio_path": str(reference_audio_path),
        "reference_text": ref_text,
        **reference_meta,
        "samples": results,
    }
    (output_dir / "samples.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
