"""Tests for the labelling side of distill_dataset.py: truncated teacher outputs, duration sorting, push retry."""

import argparse
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Audio, Dataset, Features, Value

import distill_dataset as dd
from distill_dataset import FEATURES, encode_flac, load_distill_dataset, make_labeller


def _flac_row(seconds=1.0):
    return {"bytes": encode_flac(0.05 * np.random.default_rng(0).standard_normal(int(16000 * seconds))), "path": None}


def _args(**kw):
    base = dict(audio_column="audio", text_column="text", language="en", min_seconds=0.3, max_seconds=30.0,
                nonspeech_regex=None, shard=0)
    return argparse.Namespace(**(base | kw))


def test_token_budget_grows_with_length_and_respects_limits():
    assert dd.token_budget(1.0) == dd.BASE_NEW_TOKENS + dd.TOKENS_PER_SECOND
    assert dd.token_budget(1000.0) == dd.MAX_NEW_TOKENS  # capped
    assert dd.token_budget(1000.0, prompt_len=10, max_positions=100) == 90  # leaves room for the prompt
    assert dd.token_budget(0.0, prompt_len=10, max_positions=5) == 1  # never below one token


def test_teacher_marks_generations_without_eos_as_truncated():
    """A sequence that has no end-of-text token (it hit the cap) comes back as None, a finished one as its ids."""
    eos, pad, prompt_len = 3, 2, 2
    sequences = torch.tensor([[7, 8, 20, 21, eos, pad],  # finished: transcript 20 21
                              [7, 8, 20, 21, 22, 23]])  # cut off: no eos
    seen = {}

    class FakeModel:
        dtype = torch.float32
        config = SimpleNamespace(max_position_embeddings=1024)

        def generate(self, max_new_tokens, **inputs):
            seen["max_new_tokens"] = max_new_tokens
            return sequences

    class FakeInputs(dict):
        def to(self, *a, **k):
            return self

    teacher = object.__new__(dd.Teacher)
    teacher.device = "cpu"
    teacher.model = FakeModel()
    teacher.tokenizer = SimpleNamespace(eos_token_id=eos)
    teacher.special_ids = {eos, pad, 7, 8}
    teacher.processor = type("P", (), {"__call__": lambda self, *a, **k: FakeInputs(x=1),
                                       "get_decoder_prompt_ids": lambda self, **k: [7, 8]})()
    out = teacher.transcribe([np.zeros(16000, dtype=np.float32)] * 2, "en", longest_seconds=2.0)
    assert out == [[20, 21], None]
    assert seen["max_new_tokens"] == dd.token_budget(2.0, prompt_len, 1024)


@pytest.mark.parametrize("cuda,native_bf16,expected", [(False, False, torch.float32),
                                                        (True, False, torch.float16),
                                                        (True, True, torch.bfloat16)])
def test_teacher_default_precision_matches_device(monkeypatch, cuda, native_bf16, expected):
    """The documented T4 labelling path needs float16; CPU uses float32 and newer GPUs can use native bf16."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **k: native_bf16)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(dd.AutoProcessor, "from_pretrained",
                        lambda *a, **k: SimpleNamespace(tokenizer=SimpleNamespace(all_special_ids=[])))
    seen = []

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(dd.CohereAsrForConditionalGeneration, "from_pretrained",
                        lambda *a, **k: (seen.append(k["dtype"]), FakeModel())[1])
    dd.Teacher()
    assert seen == [expected]


def test_labeller_gives_truncated_generations_an_empty_label_and_counts_them(tmp_path):
    class FakeTeacher:
        tokenizer = SimpleNamespace(decode=lambda ids: "t" + "".join(map(str, ids)))

        def transcribe(self, waves, language, longest_seconds):
            return [[1, 2], None, [3]][: len(waves)]  # the 2nd clip was truncated

    stats = dd.LabelStats()
    source = Dataset.from_dict({"audio": [_flac_row(1.0)] * 3, "text": ["a", "b", "c"]})
    source = source.cast_column("audio", Audio(decode=False))
    out = source.map(make_labeller(_args(), FakeTeacher(), stats), batched=True, batch_size=3, with_indices=True,
                     remove_columns=source.column_names, features=FEATURES)
    assert out["teacher_ids"] == [[1, 2], [], [3]]
    assert (stats.truncated, stats.labelled) == (1, 3)
    # the empty label is dropped by the reader, so the truncated clip never reaches training
    out = out.cast_column("audio", Audio(decode=False))
    out.save_to_disk(str(tmp_path / "saved"))
    assert load_distill_dataset(str(tmp_path / "saved"), 30.0)["teacher_ids"] == [[1, 2], [3]]


def test_sorted_by_duration_and_restored_to_source_order():
    """Sorting only changes batching: ids are the original positions and `_pos` restores the source order."""
    seconds = [1.0, 3.0, 2.0]
    source = Dataset.from_dict({"audio": [_flac_row(s) for s in seconds], "text": ["a", "b", "c"]})
    source = source.cast_column("audio", Audio(decode=False))
    order = dd.order_by_duration(source, "audio")
    assert order == [1, 2, 0]  # longest first

    batches = []

    class FakeTeacher:
        tokenizer = SimpleNamespace(decode=lambda ids: "")

        def transcribe(self, waves, language, longest_seconds):
            batches.append([len(w) / 16000 for w in waves])
            return [[1]] * len(waves)

    features = Features({**FEATURES, "_pos": Value("int64")})
    out = source.select(order).map(make_labeller(_args(), FakeTeacher(), source_order=order), batched=True,
                                   batch_size=2, with_indices=True, remove_columns=source.column_names,
                                   features=features)
    assert batches == [[3.0, 2.0], [1.0]]  # similar lengths share a batch
    out = out.sort("_pos")
    assert out["_pos"] == [0, 1, 2] and out["id"] == ["0000-0", "0000-1", "0000-2"]
    assert out["duration"] == pytest.approx(seconds)


class _FlakyDataset:
    def __init__(self, errors):
        self.errors, self.calls = list(errors), 0

    def push_to_hub(self, repo, split, private):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)


def _http_error(status):
    error = OSError(f"HTTP {status}")
    error.response = SimpleNamespace(status_code=status)
    return error


def test_push_retries_conflicts_with_exponential_backoff():
    ds, waits = _FlakyDataset([_http_error(409), _http_error(412)]), []
    dd.push_with_retry(ds, "org/repo", "shard0001", first_wait=1.0, sleep=waits.append)
    assert ds.calls == 3 and waits == [1.0, 2.0]


def test_push_gives_up_after_all_attempts_and_skips_hopeless_errors():
    ds = _FlakyDataset([_http_error(409)] * 5)
    with pytest.raises(OSError):
        dd.push_with_retry(ds, "org/repo", "shard0001", attempts=3, sleep=lambda s: None)
    assert ds.calls == 3
    ds = _FlakyDataset([_http_error(401)])  # bad token: retrying cannot help
    with pytest.raises(OSError):
        dd.push_with_retry(ds, "org/repo", "shard0001", sleep=lambda s: None)
    assert ds.calls == 1


def _streaming_args(**kw):
    return _args(dataset="fake", config=None, split="train", streaming=True, num_shards=1, limit=None, bs=2,
                 dtype="float32", **kw)


_TEACHER_CALLS = []  # module level: state kept out of the closures `from_generator` hashes, so reruns hash equal


class _CountingTeacher:
    tokenizer = SimpleNamespace(decode=lambda ids: "t")

    def __init__(self, dtype):
        pass

    def transcribe(self, waves, language, longest_seconds):
        _TEACHER_CALLS.append(len(waves))
        return [[1]] * len(waves)


def test_streaming_relabel_runs_the_teacher_on_every_run(monkeypatch, tmp_path):
    """A rerun with identical arguments must label again (no on-disk cache hit that skips the teacher)."""
    import datasets.config

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", str(tmp_path / "cache"))
    _TEACHER_CALLS.clear()
    source = Dataset.from_dict({"audio": [_flac_row(1.0)] * 3, "text": ["a", "b", "c"]})
    monkeypatch.setattr(dd, "load_dataset", lambda *a, **k: source.to_iterable_dataset())
    monkeypatch.setattr(dd, "Teacher", _CountingTeacher)
    for run in range(2):
        stats = dd.LabelStats()
        out = dd.relabel(_streaming_args(), stats)
        assert len(out) == 3 and out["teacher_ids"] == [[1]] * 3
        assert stats.labelled == 3, f"run {run} did not run the teacher"
    assert _TEACHER_CALLS == [2, 1, 2, 1]


def test_order_by_duration_uses_duration_column_without_reading_audio():
    seconds = [1.0, 3.0, 2.0]
    source = Dataset.from_dict({"audio": [_flac_row(1.0)] * 3, "dur": seconds})  # audio lengths all equal
    source = source.cast_column("audio", Audio(decode=False))
    assert dd.order_by_duration(source, "audio") == [0, 1, 2]  # headers say all equal (stable sort)
    assert dd.order_by_duration(source, "audio", duration_column="dur") == [1, 2, 0]


def test_order_by_duration_reads_headers_in_batches():
    seconds = [0.5, 3.0, 1.0, 2.0, 0.7]
    source = Dataset.from_dict({"audio": [_flac_row(x) for x in seconds]}).cast_column("audio", Audio(decode=False))
    assert dd.order_by_duration(source, "audio", batch_size=2) == [1, 3, 2, 4, 0]


class _DrawCollator(dd.DistillCollator):
    """Collator whose batches are just this process's next random draw."""

    def __call__(self, examples):
        return {"draw": torch.tensor([self._rng_of_this_process().random()])}


def _draw(collator, worker=None):
    """First draw of a collator, as if running in a DataLoader worker (id, seed) or the main process."""
    old = torch.utils.data.get_worker_info
    torch.utils.data.get_worker_info = lambda: SimpleNamespace(id=worker[0], seed=worker[1]) if worker else None
    try:
        return collator._rng_of_this_process().random()
    finally:
        torch.utils.data.get_worker_info = old


def test_augmentation_rng_differs_per_worker_and_per_dataloader_restart_but_is_deterministic(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    make = lambda seed=7: dd.DistillCollator(None, None, "utterance", seed=seed)
    base = _draw(make(), (0, 1000))
    assert _draw(make(), (0, 1000)) == base  # deterministic given seeds
    assert _draw(make(), (1, 1001)) != base  # other worker
    assert _draw(make(), (0, 2000)) != base  # same worker id, workers recreated (new epoch / resume)
    assert _draw(make(seed=8), (0, 1000)) != base  # user seed
    monkeypatch.setenv("RANK", "1")
    assert _draw(make(), (0, 1000)) != base  # other rank


def test_augmentation_rng_changes_every_epoch_with_real_dataloader_workers():
    def epochs(seed):
        torch.manual_seed(seed)  # the global seed Trainer sets; DataLoader derives base_seed from it each epoch
        loader = torch.utils.data.DataLoader(list(range(2)), batch_size=1, num_workers=1,
                                             collate_fn=_DrawCollator(None, None, "utterance", seed=3))
        return [[b["draw"].item() for b in loader] for _ in range(3)]

    first = epochs(0)
    assert len({tuple(e) for e in first}) == 3  # each epoch draws a different sequence
    assert epochs(0) == first  # but the whole run is reproducible from the global seed
