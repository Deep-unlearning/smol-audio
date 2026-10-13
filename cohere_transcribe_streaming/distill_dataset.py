"""Label audio with Cohere Transcribe and prepare the labelled rows for training.

Each row contains 16 kHz FLAC audio, a language, duration, teacher token IDs and
text, an optional reference transcript, and an ID. Incomplete teacher transcripts
get an empty label and are excluded from training.

Use --shard i --num-shards N to split labelling across independent jobs. Each job
saves or uploads a shardNNNN split; rerun failed shards with the same arguments.
"""

from __future__ import annotations

import argparse
import glob
import io
import math
import os
import random
import re
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    Features,
    List,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

from model import COHERE_MODEL_ID, log_mel, normalize

SAMPLE_RATE = 16000
MIN_SECONDS = 0.3
IGNORE = -100  # Cross-entropy ignores this target value.

TOKENS_PER_SECOND = 12
BASE_NEW_TOKENS = 16
MAX_NEW_TOKENS = 448

PUSH_ATTEMPTS = 5
PUSH_FIRST_WAIT = 5.0
RETRY_STATUS = {409, 412, 429, 500, 502, 503, 504}

# Store audio bytes directly so reading and writing do not require torchcodec.
FEATURES = Features({
    "id": Value("string"),
    "language": Value("string"),
    "duration": Value("float32"),
    "teacher_ids": List(Value("int32")),
    "teacher_text": Value("string"),
    "reference": Value("string"),
    "audio": {"bytes": Value("binary"), "path": Value("string")},
})


def decode_flac(audio_bytes: bytes) -> np.ndarray:
    """Decode audio bytes (FLAC, or any format soundfile reads) to 16 kHz mono float32."""
    waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)
    if sample_rate != SAMPLE_RATE:
        import librosa

        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return waveform


def audio_bytes(field: dict) -> bytes:
    """Read encoded audio from a dataset cell containing bytes or a file path."""
    if field.get("bytes"):
        return field["bytes"]
    return Path(field["path"]).read_bytes()


def encode_flac(audio) -> bytes:
    """Encode float audio as 16-bit FLAC bytes."""
    buffer = io.BytesIO()
    waveform = np.asarray(audio, dtype=np.float32)
    sf.write(buffer, waveform, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


class Teacher:
    """The offline Cohere Transcribe model, used to produce token ids for a batch of clips."""

    def __init__(self, dtype: str = "auto"):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        if dtype == "auto":
            dtype = "float32"
            if self.device == "cuda":
                if torch.cuda.is_bf16_supported(including_emulation=False):
                    dtype = "bfloat16"
                else:
                    dtype = "float16"

        self.processor = AutoProcessor.from_pretrained(COHERE_MODEL_ID)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            COHERE_MODEL_ID, dtype=getattr(torch, dtype)
        ).to(self.device).eval()
        self.tokenizer = self.processor.tokenizer
        self.special_ids = set(self.tokenizer.all_special_ids)

    def transcribe(self, waves: list[np.ndarray], language: str, longest_seconds: float) -> list[list[int] | None]:
        """Return text token IDs for each clip, or None for an incomplete transcript."""
        inputs = self.processor(
            waves,
            language=language,
            punctuation=True,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).to(self.device, dtype=self.model.dtype)
        # Generation includes the prompt; only the tokens after it are transcript text.
        prompt_len = len(self.processor.get_decoder_prompt_ids(language=language, punctuation=True))
        max_positions = getattr(self.model.config, "max_position_embeddings", None)
        budget = token_budget(longest_seconds, prompt_len, max_positions)
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=budget)
        eos_id = self.tokenizer.eos_token_id
        results = []
        for sequence in generated.tolist():
            new_tokens = sequence[prompt_len:]
            if eos_id not in new_tokens:
                results.append(None)
                continue
            text_tokens = [token for token in new_tokens if token not in self.special_ids]
            results.append(text_tokens)
        return results


def token_budget(longest_seconds: float, prompt_len: int = 0, max_positions: int | None = None) -> int:
    """Allow more tokens for longer clips while leaving decoder space for the prompt."""
    budget = min(MAX_NEW_TOKENS, int(BASE_NEW_TOKENS + TOKENS_PER_SECOND * longest_seconds))
    if max_positions:
        budget = min(budget, max_positions - prompt_len)
    return max(budget, 1)


class LabelStats:
    """Count written rows and incomplete teacher transcripts."""

    def __init__(self):
        self.labelled = 0
        self.truncated = 0


def make_labeller(args: argparse.Namespace, teacher: Teacher, stats: LabelStats | None = None, source_order=None):
    """Build a Dataset.map function that labels a batch and tracks its source positions."""
    if stats is None:
        stats = LabelStats()
    nonspeech = re.compile(args.nonspeech_regex) if args.nonspeech_regex else None

    def label(batch: dict, indices: list[int]) -> dict:
        waves = [decode_flac(audio_bytes(field)) for field in batch[args.audio_column]]
        references = batch.get(args.text_column) or [None] * len(waves)
        if source_order is None:
            positions = list(indices)
        else:
            positions = [int(source_order[index]) for index in indices]
        row_ids = batch.get("id") or [f"{args.shard:04d}-{position}" for position in positions]
        kept_indices = [
            index for index, waveform in enumerate(waves)
            if args.min_seconds <= len(waveform) / SAMPLE_RATE <= args.max_seconds
        ]
        output = {name: [] for name in FEATURES}
        if source_order is not None:
            output["_pos"] = []
        if not kept_indices:
            return output

        waves = [waves[index] for index in kept_indices]
        durations = [len(waveform) / SAMPLE_RATE for waveform in waves]
        transcripts = teacher.transcribe(waves, args.language, max(durations))
        for index, waveform, duration, token_ids in zip(kept_indices, waves, durations, transcripts):
            reference = references[index]
            if nonspeech is not None and reference is not None and nonspeech.fullmatch(reference.strip()):
                token_ids = []
            elif token_ids is None:
                stats.truncated += 1
                token_ids = []

            output["id"].append(str(row_ids[index]))
            output["language"].append(args.language)
            output["duration"].append(duration)
            output["teacher_ids"].append(token_ids)
            output["teacher_text"].append(teacher.tokenizer.decode(token_ids))
            output["reference"].append(reference)
            output["audio"].append({"bytes": encode_flac(waveform), "path": None})
            if source_order is not None:
                output["_pos"].append(positions[index])
            stats.labelled += 1
        return output

    label.stats = stats
    return label


def clip_seconds(field: dict) -> float:
    """Read the clip duration from its audio header."""
    try:
        if field.get("bytes"):
            source = io.BytesIO(field["bytes"])
        else:
            source = field["path"]
        return float(sf.info(source).duration)
    except Exception:
        # Sort unreadable clips last; labelling will report the decoding error.
        return 0.0


def order_by_duration(source: Dataset, audio_column: str, duration_column: str | None = None,
                      batch_size: int = 256) -> list[int]:
    """Group similar lengths, longest first so oversized batches fail early."""
    if duration_column:
        durations = [float(duration) if duration is not None else 0.0 for duration in source[duration_column]]
    else:
        durations = []
        for batch in source.select_columns([audio_column]).iter(batch_size=batch_size):
            durations.extend(clip_seconds(field) for field in batch[audio_column])
    return sorted(range(len(durations)), key=lambda index: -durations[index])


def relabel(args: argparse.Namespace, stats: LabelStats | None = None) -> Dataset:
    """Label this job's shard and return its rows in source order."""
    if stats is None:
        stats = LabelStats()
    source = load_dataset(args.dataset, args.config, split=args.split, streaming=args.streaming)
    source = source.cast_column(args.audio_column, Audio(decode=False))
    if args.num_shards > 1:
        if args.streaming and source.num_shards < args.num_shards:
            # When there are fewer source files than jobs, distribute individual rows.
            source = source.filter(lambda _, index: index % args.num_shards == args.shard, with_indices=True)
        elif args.streaming:
            source = source.shard(num_shards=args.num_shards, index=args.shard)
        else:
            source = source.shard(num_shards=args.num_shards, index=args.shard, contiguous=False)

    if args.limit is not None:
        if args.streaming:
            source = source.take(args.limit)
        else:
            source = source.select(range(min(args.limit, len(source))))

    order = None
    if not args.streaming:
        order = order_by_duration(source, args.audio_column, getattr(args, "duration_column", None))
        source = source.select(order)
    labeller = make_labeller(args, Teacher(args.dtype), stats, source_order=order)
    features = FEATURES
    options = {}
    if order is not None:
        features = Features({**FEATURES, "_pos": Value("int64")})
    if not args.streaming:
        options["load_from_cache_file"] = False
    labelled = source.map(
        labeller,
        batched=True,
        batch_size=args.bs,
        with_indices=True,
        remove_columns=list(source.column_names or []) or None,
        features=features,
        **options,
    )
    if args.streaming:
        # A fresh fingerprint prevents cached rows from skipping labelling and its counters.
        labelled = Dataset.from_generator(lambda: iter(labelled), features=FEATURES, fingerprint=uuid.uuid4().hex)
    if order is not None:
        labelled = labelled.sort("_pos").remove_columns("_pos")
    return labelled.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, decode=False))


def push_with_retry(dataset: Dataset, repo: str, split: str, attempts: int = PUSH_ATTEMPTS,
                    first_wait: float = PUSH_FIRST_WAIT, sleep=time.sleep) -> None:
    """Upload a private split, retrying conflicts and temporary connection errors."""
    for attempt in range(1, attempts + 1):
        try:
            dataset.push_to_hub(repo, split=split, private=True)
            return
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status is None:
                retryable = isinstance(error, (OSError, TimeoutError))
            else:
                retryable = status in RETRY_STATUS
            if not retryable or attempt == attempts:
                raise
            wait = first_wait * 2 ** (attempt - 1)
            print(f"push to {repo} failed ({type(error).__name__}: {error}); retry {attempt}/{attempts - 1} in {wait:.0f}s",
                  flush=True)
            sleep(wait)


def split_exists(repo: str, split: str) -> bool | None:
    """Check for the uploaded split; return None when the Hub check fails."""
    try:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(repo, repo_type="dataset")
    except Exception:
        return None
    return any(filename.startswith(f"data/{split}-") for filename in files)


def _load_one(spec: str) -> Dataset:
    """Read Parquet files, saved dataset folders, or all splits of a Hub dataset."""
    if spec.startswith("hf://") or spec.endswith(".parquet"):
        return load_dataset("parquet", data_files=spec, split="train")
    local_paths = sorted(glob.glob(spec))
    if local_paths:
        return concatenate_datasets([load_from_disk(path) for path in local_paths])
    loaded = load_dataset(spec)
    if isinstance(loaded, DatasetDict):
        return concatenate_datasets(list(loaded.values()))
    return loaded


def load_distill_dataset(data: str, max_seconds: float) -> Dataset:
    """Join comma-separated datasets and keep usable clips with non-empty labels."""
    dataset = concatenate_datasets([_load_one(spec.strip()) for spec in data.split(",")])
    dataset = dataset.cast_column("audio", Audio(decode=False))

    def usable_rows(durations, labels):
        return [
            MIN_SECONDS <= duration <= max_seconds and len(token_ids) > 0
            for duration, token_ids in zip(durations, labels)
        ]

    return dataset.filter(usable_rows, input_columns=["duration", "teacher_ids"], batched=True)


def build_decoder_io(
    prompt_ids: list[int] | list[list[int]], label_ids: list[list[int]], eos_id: int, pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build padded decoder inputs, attention masks, and next-token targets.

    A prompt can be shared or supplied per clip. Prompt positions are ignored in
    the loss except the final one, which predicts the first text token.
    """
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompts = prompt_ids
    else:
        prompts = [prompt_ids] * len(label_ids)

    input_sequences = []
    target_sequences = []
    for prompt, tokens in zip(prompts, label_ids):
        input_sequences.append(list(prompt) + list(tokens))
        target_sequences.append([IGNORE] * (len(prompt) - 1) + list(tokens) + [eos_id])

    batch_size = len(input_sequences)
    max_length = max(len(sequence) for sequence in input_sequences)
    input_ids = torch.full((batch_size, max_length), pad_id, dtype=torch.long)
    targets = torch.full((batch_size, max_length), IGNORE, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    for index, (input_tokens, target_tokens) in enumerate(zip(input_sequences, target_sequences)):
        input_ids[index, :len(input_tokens)] = torch.tensor(input_tokens)
        targets[index, :len(target_tokens)] = torch.tensor(target_tokens)
        attention_mask[index, :len(input_tokens)] = 1
    return input_ids, attention_mask, targets


def collate_items(items, processor, stats, norm_mode, device="cpu", rng=None, prior_range=None, gain_db=0.0):
    """Create a training batch from (audio bytes, label metadata) pairs.

    Random gain and normalisation priors affect only the student's input.
    """
    rng = rng or random
    audio = [torch.from_numpy(decode_flac(encoded_audio)) for encoded_audio, _ in items]
    teacher_features, audio_mask = log_mel(processor.feature_extractor, audio)
    student_features = teacher_features
    if gain_db:
        gained_audio = [waveform * 10 ** (rng.uniform(-gain_db, gain_db) / 20) for waveform in audio]
        student_features, _ = log_mel(processor.feature_extractor, gained_audio)

    prior_frames = 100
    if prior_range:
        low, high = prior_range
        prior_frames = int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))

    labels = [metadata["teacher_ids"] for _, metadata in items]
    prompts = [
        processor.get_decoder_prompt_ids(language=metadata["language"], punctuation=True)
        for _, metadata in items
    ]
    tokenizer = processor.tokenizer
    decoder_inputs, decoder_mask, decoder_targets = build_decoder_io(
        prompts, labels, eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id
    )
    student_features = normalize(student_features, audio_mask, norm_mode, stats, prior_frames=prior_frames)
    teacher_features = normalize(teacher_features, audio_mask, "utterance")
    return {
        "student_features": student_features.to(device),
        "teacher_features": teacher_features.to(device),
        "attention_mask": audio_mask.long().to(device),
        "decoder_input_ids": decoder_inputs.to(device),
        "decoder_attention_mask": decoder_mask.to(device),
        "decoder_targets": decoder_targets.to(device),
        "ctc_targets": labels,
    }


class DistillCollator:
    """Convert dataset rows to training batches inside each DataLoader worker."""

    def __init__(self, processor, stats, norm_mode, prior_range=None, gain_db=0.0, seed=0):
        self.processor = processor
        self.stats = stats
        self.norm_mode = norm_mode
        self.prior_range = prior_range
        self.gain_db = gain_db
        self.seed = seed
        self._rng = None
        self._rng_key = None

    def _rng_of_this_process(self) -> random.Random:
        """Keep a repeatable RNG that differs by process, rank, and worker restart."""
        worker = torch.utils.data.get_worker_info()
        worker_id = -1
        worker_seed = 0
        process_seed = None
        if worker is not None:
            worker_id = worker.id
            worker_seed = worker.seed
            process_seed = worker.seed
        key = (os.getpid(), process_seed)
        if self._rng is None or self._rng_key != key:
            rank = int(os.environ.get("RANK", "0"))
            seed = f"{self.seed}/{rank}/{worker_id}/{worker_seed}"
            self._rng = random.Random(seed)
            self._rng_key = key
        return self._rng

    def __call__(self, examples: list[dict]) -> dict:
        items = [
            (audio_bytes(example["audio"]), {"language": example["language"], "teacher_ids": example["teacher_ids"]})
            for example in examples
        ]
        return collate_items(
            items,
            self.processor,
            self.stats,
            self.norm_mode,
            rng=self._rng_of_this_process(),
            prior_range=self.prior_range,
            gain_db=self.gain_db,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    dataset_args = parser.add_argument_group("input dataset")
    dataset_args.add_argument("--dataset", required=True, help="Hub dataset ID, e.g. openslr/librispeech_asr")
    dataset_args.add_argument("--config", default=None, help="dataset config name, if it has one")
    dataset_args.add_argument("--split", required=True, help="split to read, e.g. train.clean.100")
    dataset_args.add_argument("--audio-column", default="audio")
    dataset_args.add_argument("--text-column", default="text", help="optional reference transcript column")
    dataset_args.add_argument("--language", required=True, help="language code given to the teacher, e.g. en")
    dataset_args.add_argument("--min-seconds", type=float, default=MIN_SECONDS, help="skip clips shorter than this")
    dataset_args.add_argument("--max-seconds", type=float, default=30.0, help="skip clips longer than this")
    dataset_args.add_argument(
        "--nonspeech-regex", default=None,
        help="references fully matching this pattern get an empty label (e.g. '<MUSIC>' or '<NOISE>')",
    )
    dataset_args.add_argument(
        "--duration-column", default=None,
        help="clip length column in seconds, used for sorting with --no-streaming",
    )
    dataset_args.add_argument("--limit", type=int, default=None, help="stop after this many source rows")
    dataset_args.add_argument(
        "--streaming", action=argparse.BooleanOptionalAction, default=True,
        help="stream the source; use --no-streaming to download it first",
    )

    shard_args = parser.add_argument_group("work splitting: rerun failed shards to resume")
    shard_args.add_argument("--shard", type=int, default=0, help="this job's index, from 0 to num-shards - 1")
    shard_args.add_argument("--num-shards", type=int, default=1, help="number of jobs sharing the dataset")

    teacher_args = parser.add_argument_group("teacher run")
    teacher_args.add_argument("--bs", type=int, default=32, help="clips transcribed per batch")
    teacher_args.add_argument(
        "--dtype", default="auto", choices=("auto", "bfloat16", "float16", "float32"),
        help="auto uses native CUDA bf16, otherwise CUDA float16 or CPU/MPS float32",
    )

    output_args = parser.add_argument_group("output: choose at least one")
    output_args.add_argument(
        "--out", default=None,
        help="save_to_disk folder; each job uses <out>/shardNNNN when num-shards > 1",
    )
    output_args.add_argument(
        "--push-to", default=None,
        help="private Hub dataset repo; each job uploads and verifies its shardNNNN split",
    )
    args = parser.parse_args()
    if not (args.out or args.push_to):
        parser.error("give --out and/or --push-to")
    return args


def main():
    args = parse_args()
    stats = LabelStats()
    labelled = relabel(args, stats)
    print(f"labelled {len(labelled)} rows, {sum(labelled['duration']) / 3600:.1f} h", flush=True)
    print(f"{stats.truncated} of {stats.labelled} clips had a truncated teacher output (hit the {MAX_NEW_TOKENS}-token "
          "cap or the per-second budget) and got an empty label; they are dropped when training", flush=True)
    split = f"shard{args.shard:04d}"
    if args.out:
        output_path = args.out
        if args.num_shards > 1:
            output_path = os.path.join(args.out, split)
        labelled.save_to_disk(output_path)
        print(f"saved -> {output_path}", flush=True)
    if args.push_to:
        push_with_retry(labelled, args.push_to, split)
        print(f"pushed -> {args.push_to} (split {split})", flush=True)
        if split_exists(args.push_to, split) is False:
            print(f"WARNING: split {split} not found in {args.push_to} after the push; check the repo and re-run "
                  f"--shard {args.shard} if it is missing", flush=True)


if __name__ == "__main__":
    main()
