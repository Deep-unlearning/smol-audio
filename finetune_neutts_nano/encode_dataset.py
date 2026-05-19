"""Encode a Hugging Face audio dataset into NeuCodec tokens for NeuTTS training.

Reads a source dataset that has a text column and an audio column, resamples each
clip to 16 kHz mono for NeuCodec input, runs `neucodec.NeuCodec.encode_code`,
and writes a dataset with just `text` and `codes` -- the schema expected by
`finetune_neutts_nano.py`. NeuCodec decodes those codes back to 24 kHz audio.

Example:
    python encode_dataset.py \
        --source-dataset nvidia/Granary \
        --source-config it_voxpopuli \
        --source-split asr \
        --max-samples 200 \
        --output-dir ./granary_it_neucodec

    python encode_dataset.py \
        --source-dataset nvidia/Granary --source-config it_voxpopuli \
        --source-split asr --output-dataset your-user/granary-it-neutts
"""

from __future__ import annotations

import argparse
from typing import Iterator

import librosa
import numpy as np
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import create_repo

NEUCODEC_INPUT_SAMPLE_RATE = 16_000
NEUCODEC_OUTPUT_SAMPLE_RATE = 24_000
CODEBOOK_SIZE = 65_536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dataset", required=True, help="Hub id or local path of the source dataset.")
    parser.add_argument("--source-config", default=None, help="Optional dataset config name (e.g. 'it_voxpopuli').")
    parser.add_argument("--source-split", default="train", help="Split to encode (e.g. 'train', 'asr', 'train[:1000]').")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--codec-checkpoint", default="neuphonic/neucodec")
    parser.add_argument("--max-samples", type=int, default=None, help="Encode at most this many rows.")
    parser.add_argument("--output-dir", default=None, help="Save the encoded dataset to this local directory.")
    parser.add_argument("--output-dataset", default=None, help="Push the encoded dataset to this Hub repo id.")
    parser.add_argument("--hub-private", action="store_true", help="Create the Hub repo as private.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def load_codec(checkpoint: str, device: torch.device):
    from neucodec import NeuCodec  # local import keeps CLI usable without the dep at parse-time

    codec = NeuCodec.from_pretrained(checkpoint).eval().to(device)
    return codec


def to_mono_16k(audio_field) -> np.ndarray:
    array = np.asarray(audio_field["array"], dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=0)
    sr = int(audio_field["sampling_rate"])
    if sr != NEUCODEC_INPUT_SAMPLE_RATE:
        array = librosa.resample(array, orig_sr=sr, target_sr=NEUCODEC_INPUT_SAMPLE_RATE)
    return array


def encode_one(codec, wav: np.ndarray, device: torch.device) -> list[int]:
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        codes = codec.encode_code(audio_or_path=wav_tensor)
    codes = codes.squeeze().detach().cpu().view(-1).tolist()
    codes = [int(c) for c in codes]
    if any(c < 0 or c >= CODEBOOK_SIZE for c in codes):
        raise ValueError("NeuCodec produced an out-of-range code id.")
    return codes


def iter_records(source, codec, device: torch.device, text_column: str, audio_column: str, max_samples: int | None) -> Iterator[dict]:
    seen = 0
    for sample in source:
        if max_samples is not None and seen >= max_samples:
            break
        text = sample.get(text_column)
        audio = sample.get(audio_column)
        if not text or audio is None:
            continue
        try:
            wav = to_mono_16k(audio)
            codes = encode_one(codec, wav, device)
        except Exception as exc:
            print(f"[skip row] {exc}")
            continue
        seen += 1
        if seen % 50 == 0:
            print(f"  encoded {seen} rows ...")
        yield {"text": text, "codes": codes}


def main() -> None:
    args = parse_args()
    if not args.output_dir and not args.output_dataset:
        raise SystemExit("Provide --output-dir or --output-dataset (or both).")

    device = resolve_device(args.device)
    print(f"Loading NeuCodec on {device} ...")
    codec = load_codec(args.codec_checkpoint, device)

    print(f"Loading source dataset {args.source_dataset} (config={args.source_config}, split={args.source_split}) ...")
    load_kwargs = {"split": args.source_split}
    if args.source_config:
        load_kwargs["name"] = args.source_config
    source = load_dataset(args.source_dataset, **load_kwargs)

    print("Encoding ...")
    records = list(iter_records(source, codec, device, args.text_column, args.audio_column, args.max_samples))
    if not records:
        raise SystemExit("No rows were encoded -- check --text-column / --audio-column / --max-samples.")
    print(f"Encoded {len(records)} rows.")
    print(f"Sample text: {records[0]['text'][:80]!r}")
    print(f"Sample codes[:10]: {records[0]['codes'][:10]}")

    dataset = Dataset.from_list(records)

    if args.output_dir:
        print(f"Saving to {args.output_dir} ...")
        dataset.save_to_disk(args.output_dir)

    if args.output_dataset:
        print(f"Pushing to Hub repo {args.output_dataset} ...")
        create_repo(args.output_dataset, repo_type="dataset", private=args.hub_private, exist_ok=True)
        dataset.push_to_hub(args.output_dataset, private=args.hub_private)

    print("Done.")


if __name__ == "__main__":
    main()
