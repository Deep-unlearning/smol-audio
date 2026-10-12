"""Everything about the data for distillation: label audio with the teacher, and read the result for training.

Why: the streaming student is trained to imitate the offline Cohere Transcribe teacher's output rather than human
transcripts, so every training clip first needs the teacher's transcript (as token ids). Run this file once to
produce that labelled dataset; `train.py` then imports the reading side (`load_distill_dataset`, `DistillCollator`).

The labelled dataset is an ordinary Hugging Face `datasets` dataset with one row per utterance:
  audio         the clip as 16 kHz FLAC (an `Audio` feature), so the dataset is complete training data on its own
                (about 60 MB per hour of audio),
  language      language code given to the teacher,
  duration      seconds,
  teacher_ids   the teacher's text token ids (no prompt, no end-of-text),
  teacher_text  the same tokens decoded to text,
  reference     the source dataset's own transcript, if it has one (handy for filtering rows where teacher and
                reference disagree badly),
  id            a row id.

To split the work over several GPUs or jobs give each one --shard i --num-shards N (the jobs never overlap). With
--push-to every job uploads its part as its own split (`shard0000`, `shard0001`, ...) of the same Hub dataset, and
`--data <that repo>` in train.py reads them all.

Resuming a crashed run: `--shard i --num-shards N` is the resumability mechanism. A labelling job over a big dataset
can run for hours, and a job that dies loses its progress, but the shards are independent: keep N fairly large (so each
shard is short), and when some jobs fail re-run ONLY those shards with the same arguments. Shards that already finished
(saved under --out, or pushed as a split) are left alone. With --out and --num-shards > 1 each job saves to its own
folder `<out>/shardNNNN`, so a re-run never overwrites a finished shard (`--data '<out>/*'` in train.py reads them all).

Pushing from several jobs at once: every `push_to_hub` edits the same README / config metadata of the repo, so two jobs
finishing together can hit a commit conflict. `push_with_retry` retries such errors (and temporary server / network
errors) up to PUSH_ATTEMPTS times with exponential backoff, so shards pushed simultaneously are safe. After the push the
job also checks via the Hub api that its split really exists and prints a warning if not. To be sure at the end,
verify the whole repo once with `load_distill_dataset("<repo>", 30.0)`: it should list every `shardNNNN` split.

Teacher outputs that are cut off: the teacher is given a token budget that grows with the clip length (see
`token_budget`). A generation that hits the budget without finishing (no end-of-text token) is a truncated transcript;
training on it would teach the student to stop mid-sentence, so such clips get an EMPTY label (which
`load_distill_dataset` drops) and the final log line reports how many there were. Seeing many of them means the
budget (TOKENS_PER_SECOND) is too small for that language, or the clips contain looping hallucinations.

Speed: in non-streaming mode (`--no-streaming`) the clips of a shard are sorted by duration before batching, so the
clips of one teacher batch have similar length and little padding is wasted (the output is put back in source order).
Streaming mode cannot sort (it sees the rows one after another) and uses source order.

Layout of this file: 1. labelling with the teacher, 2. reading the labelled dataset, 3. batches for training,
4. command line.

Example
-------
    python distill_dataset.py --dataset openslr/librispeech_asr --config all --split train.clean.100 \
        --language en --out data/relabel/librispeech-100 --shard 0 --num-shards 8

    # upload instead of (or as well as) saving locally
    python distill_dataset.py --dataset facebook/multilingual_librispeech --config french --split train \
        --text-column transcript --language fr --push-to my-org/distill-mls-fr --shard 0 --num-shards 4
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
from datasets import (Audio, Dataset, DatasetDict, Features, List, Value, concatenate_datasets, load_dataset,
                      load_from_disk)
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

from model import COHERE_MODEL_ID, log_mel, normalize

SAMPLE_RATE = 16000
MIN_SECONDS = 0.3  # clips shorter than this are skipped
IGNORE = -100  # label value that cross-entropy skips

# Teacher generation budget. A clip of `s` seconds gets BASE_NEW_TOKENS + TOKENS_PER_SECOND * s new tokens, but never
# more than MAX_NEW_TOKENS, and never more than the decoder can hold (see `token_budget`). Ordinary speech needs about
# 3-8 tokens per second (fast speech and languages written with many small tokens, e.g. Japanese, need more), so 12 per
# second is generous: a generation that still hits the budget is almost certainly cut off or stuck in a loop.
TOKENS_PER_SECOND = 12
BASE_NEW_TOKENS = 16  # room for short clips (a one-word clip still needs a few tokens)
MAX_NEW_TOKENS = 448  # hard cap of the teacher's decoder budget

# Pushing to the Hub from several jobs at once can conflict; see `push_with_retry`.
PUSH_ATTEMPTS = 5
PUSH_FIRST_WAIT = 5.0  # seconds before the 2nd attempt; doubles every attempt (5, 10, 20, 40)
RETRY_STATUS = {409, 412, 429, 500, 502, 503, 504}  # HTTP codes that are worth trying again

# Columns of the labelled dataset. The audio is written as the plain {bytes, path} struct that `Audio(decode=False)`
# stores (writing through the `Audio` feature itself needs torchcodec); `relabel` then casts it to `Audio` so the
# Hub viewer shows a player. We never use `datasets`' own audio decoding: `decode_flac` reads the bytes.
FEATURES = Features({
    "id": Value("string"), "language": Value("string"), "duration": Value("float32"),
    "teacher_ids": List(Value("int32")), "teacher_text": Value("string"), "reference": Value("string"),
    "audio": {"bytes": Value("binary"), "path": Value("string")},
})


def decode_flac(audio_bytes: bytes) -> np.ndarray:
    """Decode audio bytes (FLAC, or any format soundfile reads) to 16 kHz mono float32."""
    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(-1)
    if sr != SAMPLE_RATE:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    return wav


def audio_bytes(field: dict) -> bytes:
    """Encoded bytes of an undecoded `Audio(decode=False)` cell, `{"bytes": ...}` or `{"path": ...}`."""
    return field["bytes"] if field.get("bytes") else Path(field["path"]).read_bytes()


def encode_flac(audio) -> bytes:
    """Encode float audio as 16-bit FLAC bytes."""
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


# ============================================================================================================
# 1. Labelling with the teacher
# ============================================================================================================


class Teacher:
    """The offline Cohere Transcribe model, used to produce token ids for a batch of clips."""

    def __init__(self, dtype: str = "auto"):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if dtype == "auto":
            dtype = ("bfloat16" if self.device == "cuda" and
                     torch.cuda.is_bf16_supported(including_emulation=False) else
                     "float16" if self.device == "cuda" else "float32")
        self.processor = AutoProcessor.from_pretrained(COHERE_MODEL_ID)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            COHERE_MODEL_ID, dtype=getattr(torch, dtype)).to(self.device).eval()
        self.tokenizer = self.processor.tokenizer
        self.special_ids = set(self.tokenizer.all_special_ids)

    def transcribe(self, waves: list[np.ndarray], language: str, longest_seconds: float) -> list[list[int] | None]:
        """Teacher token ids (no prompt, no special tokens) for each waveform.

        An entry is None when the teacher did not finish: the generation hit the token budget without producing its
        end-of-text token, so the transcript is cut off (or looping) and must not be used as a label.
        """
        inputs = self.processor(waves, language=language, punctuation=True, sampling_rate=SAMPLE_RATE,
                                return_tensors="pt").to(self.device, dtype=self.model.dtype)
        # generate() returns the decoder prompt too; its length tells us where the transcript starts
        prompt_len = len(self.processor.get_decoder_prompt_ids(language=language, punctuation=True))
        max_positions = getattr(self.model.config, "max_position_embeddings", None)
        budget = token_budget(longest_seconds, prompt_len, max_positions)
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=budget)
        eos_id = self.tokenizer.eos_token_id
        results = []
        for sequence in generated.tolist():
            new_tokens = sequence[prompt_len:]
            finished = eos_id in new_tokens  # the teacher stopped by itself, it was not cut off by the budget
            results.append([t for t in new_tokens if t not in self.special_ids] if finished else None)
        return results


def token_budget(longest_seconds: float, prompt_len: int = 0, max_positions: int | None = None) -> int:
    """How many new tokens the teacher may generate for a batch whose longest clip lasts `longest_seconds`.

    Grows with the clip length, is capped by MAX_NEW_TOKENS, and (when the decoder size is known) leaves room for the
    prompt so prompt + new tokens never exceed the decoder's `max_positions`.
    """
    budget = min(MAX_NEW_TOKENS, int(BASE_NEW_TOKENS + TOKENS_PER_SECOND * longest_seconds))
    if max_positions:
        budget = min(budget, max_positions - prompt_len)
    return max(budget, 1)


class LabelStats:
    """Counters the labeller fills in while it runs, so `main` can report them at the end."""

    def __init__(self):
        self.labelled = 0  # rows written
        self.truncated = 0  # clips whose teacher output was cut off at the token budget (given an empty label)


def make_labeller(args: argparse.Namespace, teacher: Teacher, stats: LabelStats | None = None, source_order=None):
    """The function given to `Dataset.map`: a batch of source rows in, a batch of labelled rows out.

    `stats` collects counters (e.g. how many generations were truncated). `source_order` is only used when the source
    was re-sorted by duration: `source_order[i]` is the original position of row i. The labeller then names rows after
    their original position and adds a `_pos` column, so `relabel` can put the rows back in source order.
    """
    stats = stats if stats is not None else LabelStats()
    nonspeech = re.compile(args.nonspeech_regex) if args.nonspeech_regex else None

    def label(batch: dict, indices: list[int]) -> dict:
        waves = [decode_flac(audio_bytes(a)) for a in batch[args.audio_column]]
        references = batch.get(args.text_column) or [None] * len(waves)
        positions = [int(source_order[i]) for i in indices] if source_order is not None else list(indices)
        ids_of_row = batch.get("id") or [f"{args.shard:04d}-{p}" for p in positions]
        keep = [i for i, w in enumerate(waves) if args.min_seconds <= len(w) / SAMPLE_RATE <= args.max_seconds]
        out = {name: [] for name in FEATURES}
        if source_order is not None:
            out["_pos"] = []
        if not keep:
            return out
        waves = [waves[i] for i in keep]
        durations = [len(w) / SAMPLE_RATE for w in waves]
        for i, w, duration, ids in zip(keep, waves, durations, teacher.transcribe(waves, args.language, max(durations))):
            reference = references[i]
            if nonspeech is not None and reference is not None and nonspeech.fullmatch(reference.strip()):
                ids = []  # non-speech segment: an empty label (train.py skips such rows)
            elif ids is None:  # the teacher was cut off at its token budget: an unusable label, also left empty
                stats.truncated += 1
                ids = []
            out["id"].append(str(ids_of_row[i]))
            out["language"].append(args.language)
            out["duration"].append(duration)
            out["teacher_ids"].append(ids)
            out["teacher_text"].append(teacher.tokenizer.decode(ids))
            out["reference"].append(reference)
            out["audio"].append({"bytes": encode_flac(w), "path": None})
            if source_order is not None:
                out["_pos"].append(positions[i])
            stats.labelled += 1
        return out

    label.stats = stats
    return label


def clip_seconds(field: dict) -> float:
    """Duration of an undecoded `Audio(decode=False)` cell, read from the file header only (no decoding)."""
    try:
        source = io.BytesIO(field["bytes"]) if field.get("bytes") else field["path"]
        return float(sf.info(source).duration)
    except Exception:  # unreadable audio: sort it last, the labeller will deal with (or fail on) it
        return 0.0


def order_by_duration(source: Dataset, audio_column: str, duration_column: str | None = None,
                      batch_size: int = 256) -> list[int]:
    """Row positions of `source`, longest clip first (similar lengths end up in the same teacher batch).

    Longest first means an out-of-memory error shows up at the very start of the job instead of hours in.
    With `duration_column` (a column of seconds in the source) the audio is not touched at all. Otherwise only the
    audio column is read, in batches (never the whole column at once) and only the file headers are parsed.
    """
    if duration_column:
        durations = [float(d) if d is not None else 0.0 for d in source[duration_column]]
    else:
        durations = []
        for batch in source.select_columns([audio_column]).iter(batch_size=batch_size):
            durations.extend(clip_seconds(field) for field in batch[audio_column])
    return sorted(range(len(durations)), key=lambda i: -durations[i])


def relabel(args: argparse.Namespace, stats: LabelStats | None = None) -> Dataset:
    """Stream the source dataset, label it with the teacher, return the labelled dataset."""
    stats = stats if stats is not None else LabelStats()
    source = load_dataset(args.dataset, args.config, split=args.split, streaming=args.streaming)
    source = source.cast_column(args.audio_column, Audio(decode=False))  # decode ourselves: no FFmpeg needed
    if args.num_shards > 1:  # this job's share of the dataset
        if args.streaming and source.num_shards < args.num_shards:  # fewer files than jobs: deal rows out instead
            source = source.filter(lambda _, i: i % args.num_shards == args.shard, with_indices=True)
        else:
            source = (source.shard(num_shards=args.num_shards, index=args.shard) if args.streaming else
                      source.shard(num_shards=args.num_shards, index=args.shard, contiguous=False))
    if args.limit is not None:  # pilot runs
        source = source.take(args.limit) if args.streaming else source.select(range(min(args.limit, len(source))))
    order = None
    if not args.streaming:  # all rows are available, so sort by duration: less padding, faster teacher batches
        order = order_by_duration(source, args.audio_column, getattr(args, "duration_column", None))
        source = source.select(order)
    labeller = make_labeller(args, Teacher(args.dtype), stats, source_order=order)
    features = FEATURES if order is None else Features({**FEATURES, "_pos": Value("int64")})
    options = {} if args.streaming else {"load_from_cache_file": False}  # a cache hit would skip the teacher (and stats)
    labelled = source.map(labeller, batched=True, batch_size=args.bs, with_indices=True,
                          remove_columns=list(source.column_names or []) or None, features=features, **options)
    if args.streaming:  # materialise the stream on disk
        # `from_generator` caches on disk under a key made from a hash of the generator (here a closure over the whole
        # Teacher). A fresh fingerprint per call makes every run build the dataset again: a cache hit would skip the
        # teacher, leaving stats at 0 and pushing stale rows. It also saves hashing the teacher model.
        labelled = Dataset.from_generator(lambda: iter(labelled), features=FEATURES, fingerprint=uuid.uuid4().hex)
    if order is not None:  # put the rows back in source order and drop the helper column
        labelled = labelled.sort("_pos").remove_columns("_pos")
    return labelled.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, decode=False))


def push_with_retry(dataset: Dataset, repo: str, split: str, attempts: int = PUSH_ATTEMPTS,
                    first_wait: float = PUSH_FIRST_WAIT, sleep=time.sleep) -> None:
    """`dataset.push_to_hub(repo, split=split)`, trying again when the Hub reports a conflict or a temporary error.

    Jobs for different shards push to the same repo at the same time and all update its README / config metadata;
    the slower job can get a commit conflict (HTTP 409/412). Waiting and pushing again fixes it, so this makes
    simultaneous shard pushes safe. Errors that retrying cannot fix (wrong token, no such repo, ...) are raised at once.
    """
    for attempt in range(1, attempts + 1):
        try:
            dataset.push_to_hub(repo, split=split, private=True)
            return
        except Exception as error:  # HTTP errors (huggingface_hub) and connection errors are all OSError subclasses
            status = getattr(getattr(error, "response", None), "status_code", None)
            retryable = status in RETRY_STATUS if status is not None else isinstance(error, (OSError, TimeoutError))
            if not retryable or attempt == attempts:
                raise
            wait = first_wait * 2 ** (attempt - 1)
            print(f"push to {repo} failed ({type(error).__name__}: {error}); retry {attempt}/{attempts - 1} in {wait:.0f}s",
                  flush=True)
            sleep(wait)


def split_exists(repo: str, split: str) -> bool | None:
    """Whether the Hub dataset `repo` has data files for `split` (True/False), or None if we could not ask the Hub."""
    try:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(repo, repo_type="dataset")
    except Exception:
        return None
    return any(f.startswith(f"data/{split}-") for f in files)


# ============================================================================================================
# 2. Reading the labelled dataset
# ============================================================================================================


def _load_one(spec: str) -> Dataset:
    """One labelled dataset: parquet files (local or `hf://datasets/...` glob), a folder written by `--out`
    (globs allowed), or a Hub dataset id (every split, e.g. one per labelling job, is used)."""
    if spec.startswith("hf://") or spec.endswith(".parquet"):
        return load_dataset("parquet", data_files=spec, split="train")
    local = sorted(glob.glob(spec))
    if local:
        return concatenate_datasets([load_from_disk(path) for path in local])
    loaded = load_dataset(spec)
    return concatenate_datasets(list(loaded.values())) if isinstance(loaded, DatasetDict) else loaded


def load_distill_dataset(data: str, max_seconds: float) -> Dataset:
    """Load `--data` (several sources separated by commas are concatenated) and drop unusable rows.

    Rows are kept when their duration is sensible and the teacher label is non-empty. Audio stays encoded (decoded
    per batch by `DistillCollator`), so the dataset is memory-mapped and costs almost no RAM.
    """
    ds = concatenate_datasets([_load_one(spec.strip()) for spec in data.split(",")])
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds.filter(lambda duration, ids: [MIN_SECONDS <= d <= max_seconds and len(i) > 0 for d, i in zip(duration, ids)],
                     input_columns=["duration", "teacher_ids"], batched=True)


# ============================================================================================================
# 3. Batches for training
#
# `Trainer` picks the rows of a batch (grouped by duration, see `train_sampling_strategy`); `DistillCollator` turns
# them into tensors: decode audio, log-mel features for the student and for the teacher, decoder inputs and targets.
# ============================================================================================================


def build_decoder_io(
    prompt_ids: list[int] | list[list[int]], label_ids: list[list[int]], eos_id: int, pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Teacher-forcing tensors for `prompt + text`.

    input  = prompt + text            (padded with pad_id)
    target = prompt[1:] + text + eos  with every prompt position masked except the last one, which predicts
             the first text token.
    `prompt_ids` is one prompt shared by the batch, or one per sample (e.g. per-sample language tokens).

    Returns:
        (input_ids, attention_mask, targets), each (B, N) with N the longest prompt + text.
    """
    prompts = prompt_ids if prompt_ids and isinstance(prompt_ids[0], list) else [prompt_ids] * len(label_ids)
    inputs, targets = [], []
    for prompt, ids in zip(prompts, label_ids):
        inputs.append(list(prompt) + list(ids))
        targets.append([IGNORE] * (len(prompt) - 1) + list(ids) + [eos_id])
    n = max(len(x) for x in inputs)
    inp = torch.full((len(inputs), n), pad_id, dtype=torch.long)
    tgt = torch.full((len(inputs), n), IGNORE, dtype=torch.long)
    att = torch.zeros((len(inputs), n), dtype=torch.long)
    for i, (x, y) in enumerate(zip(inputs, targets)):
        inp[i, : len(x)] = torch.tensor(x)
        tgt[i, : len(y)] = torch.tensor(y)
        att[i, : len(x)] = 1
    return inp, att, tgt


def collate_items(items, processor, stats, norm_mode, device="cpu", rng=None, prior_range=None, gain_db=0.0):
    """Turn (audio bytes, label meta) items into one training batch (features, decoder inputs, targets).

    `meta` is a dict with "language" and "teacher_ids". Robustness augmentations are applied to the *student*
    input only (the teacher always sees the original audio):
    * gain_db: random per-utterance gain in [-gain_db, +gain_db] dB.
    * prior_range: the causal normaliser starts from a "prior" of this many frames; drawing it log-uniformly per
      batch stops the student relying on how fast the running statistics adapt to a new recording condition.
    """
    rng = rng or random
    audio = [torch.from_numpy(decode_flac(a)) for a, _ in items]
    feats, mask = log_mel(processor.feature_extractor, audio)
    student_feats = feats
    if gain_db:  # normalisation does not fully cancel gain (log floor), so the augmentation has an effect
        gained = [a * 10 ** (rng.uniform(-gain_db, gain_db) / 20) for a in audio]
        student_feats, _ = log_mel(processor.feature_extractor, gained)
    prior_frames = 100
    if prior_range:
        low, high = prior_range
        prior_frames = int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))
    labels = [m["teacher_ids"] for _, m in items]
    prompts = [processor.get_decoder_prompt_ids(language=m["language"], punctuation=True) for _, m in items]
    tok = processor.tokenizer
    inp, att, tgt = build_decoder_io(prompts, labels, eos_id=tok.eos_token_id, pad_id=tok.pad_token_id)
    return dict(student_features=normalize(student_feats, mask, norm_mode, stats, prior_frames=prior_frames).to(device),
                teacher_features=normalize(feats, mask, "utterance").to(device),
                attention_mask=mask.long().to(device), decoder_input_ids=inp.to(device),
                decoder_attention_mask=att.to(device), decoder_targets=tgt.to(device), ctc_targets=labels)


class DistillCollator:
    """`data_collator` for `Trainer`: dataset rows -> `collate_items` batch. Runs inside DataLoader workers."""

    def __init__(self, processor, stats, norm_mode, prior_range=None, gain_db=0.0, seed=0):
        self.processor, self.stats, self.norm_mode = processor, stats, norm_mode
        self.prior_range, self.gain_db, self.seed = prior_range, gain_db, seed
        self._rng, self._rng_key = None, None

    def _rng_of_this_process(self) -> random.Random:
        """One random generator per worker process, rank and DataLoader (re)start, so augmentations differ between them.

        DataLoader draws a new `base_seed` every time it creates its workers (every epoch, and again after a resume);
        worker `i` gets `info.seed = base_seed + i`. Mixing that in makes the sequence differ from epoch to epoch while
        staying a function of the global seed (`torch.manual_seed`), the user seed and the rank. Without workers the
        generator is simply kept for the life of the process, so it keeps advancing through the epochs.
        """
        info = torch.utils.data.get_worker_info()
        key = (os.getpid(), info.seed if info else None)
        if self._rng is None or self._rng_key != key:
            seed = f"{self.seed}/{int(os.environ.get('RANK', '0'))}/{info.id if info else -1}/{info.seed if info else 0}"
            self._rng, self._rng_key = random.Random(seed), key  # str seeds are hashed deterministically (not by PYTHONHASHSEED)
        return self._rng

    def __call__(self, examples: list[dict]) -> dict:
        items = [(audio_bytes(e["audio"]), {"language": e["language"], "teacher_ids": e["teacher_ids"]})
                 for e in examples]
        return collate_items(items, self.processor, self.stats, self.norm_mode, rng=self._rng_of_this_process(),
                             prior_range=self.prior_range, gain_db=self.gain_db)


# ============================================================================================================
# 4. Command line
# ============================================================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = ap.add_argument_group("input dataset")
    g.add_argument("--dataset", required=True, help="Hub dataset id, e.g. openslr/librispeech_asr")
    g.add_argument("--config", default=None, help="dataset config name, if it has one")
    g.add_argument("--split", required=True, help="split to read, e.g. train.clean.100")
    g.add_argument("--audio-column", default="audio")
    g.add_argument("--text-column", default="text", help="column with the reference transcript (optional)")
    g.add_argument("--language", required=True, help="language code given to the teacher, e.g. en")
    g.add_argument("--min-seconds", type=float, default=MIN_SECONDS, help="skip clips shorter than this")
    g.add_argument("--max-seconds", type=float, default=30.0, help="skip clips longer than this")
    g.add_argument("--nonspeech-regex", default=None,
                   help="references fully matching this (e.g. GigaSpeech '<MUSIC>'/'<NOISE>' segments) get an "
                        "empty teacher label instead of whatever the teacher hallucinates there")
    g.add_argument("--duration-column", default=None,
                   help="source column with the clip length in seconds (--no-streaming): sorting by duration then "
                        "does not need to read the audio headers")
    g.add_argument("--limit", type=int, default=None, help="stop after this many rows (pilot runs)")
    g.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True,
                   help="stream the source instead of downloading it first (--no-streaming to download)")

    g = ap.add_argument_group("work splitting (also how to resume: re-run only the shards that failed)")
    g.add_argument("--shard", type=int, default=0,
                   help="index of this job (0 .. num-shards-1). If a job crashes, re-run just its --shard i with the "
                        "same other arguments; finished shards are independent and stay as they are")
    g.add_argument("--num-shards", type=int, default=1,
                   help="total number of jobs sharing the dataset (more, smaller shards = less work lost per crash)")

    g = ap.add_argument_group("teacher run")
    g.add_argument("--bs", type=int, default=32, help="clips transcribed per batch")
    g.add_argument("--dtype", default="auto", choices=("auto", "bfloat16", "float16", "float32"),
                   help="teacher precision (auto: native bf16 GPU, otherwise float16 CUDA / float32 CPU or MPS)")

    g = ap.add_argument_group("output (at least one)")
    g.add_argument("--out", default=None, help="folder for the labelled dataset (datasets `save_to_disk`); with "
                                                   "--num-shards > 1 each job writes its own <out>/shardNNNN")
    g.add_argument("--push-to", default=None, help="Hub dataset repo (private); this job's part becomes a split "
                                                   "named shardNNNN (parallel pushes are retried "
                                                   "on conflicts; the split is checked after the push)")
    args = ap.parse_args()
    if not (args.out or args.push_to):
        ap.error("give --out and/or --push-to")
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
        out = os.path.join(args.out, split) if args.num_shards > 1 else args.out
        labelled.save_to_disk(out)
        print(f"saved -> {out}", flush=True)
    if args.push_to:
        push_with_retry(labelled, args.push_to, split)
        print(f"pushed -> {args.push_to} (split {split})", flush=True)
        if split_exists(args.push_to, split) is False:
            print(f"WARNING: split {split} not found in {args.push_to} after the push; check the repo and re-run "
                  f"--shard {args.shard} if it is missing", flush=True)


if __name__ == "__main__":
    main()
