"""Student model, dataset reading/labelling and Trainer tests on tiny random models (no large model downloads)."""

import numpy as np
import torch
from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration, ParakeetEncoderConfig

from distill_dataset import FEATURES, IGNORE, build_decoder_io, encode_flac, load_distill_dataset, make_labeller
from model import build_streaming_encoder
from train import (DistillArguments, DistillTrainer, DistillTrainingArguments, LossWeights, StreamingStudent,
                   build_optimizer)


def tiny_cohere(seed=0):
    torch.manual_seed(seed)
    enc = ParakeetEncoderConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128, num_mel_bins=128,
        subsampling_conv_channels=16, scale_input=False, dropout=0.0, dropout_positions=0.0, layerdrop=0.0,
        activation_dropout=0.0, attention_dropout=0.0,
    )
    cfg = CohereAsrConfig(encoder_config=enc.to_dict() | {"model_type": "parakeet_encoder"}, vocab_size=64,
                          hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
                          decoder_start_token_id=4)
    return CohereAsrForConditionalGeneration(cfg).eval()


def test_build_decoder_io():
    """Decoder inputs, targets and mask are built and padded correctly for teacher forcing."""
    inp, att, tgt = build_decoder_io([7, 8, 9], [[20, 21], [22]], eos_id=3, pad_id=2)
    assert inp.tolist() == [[7, 8, 9, 20, 21], [7, 8, 9, 22, 2]]
    assert tgt.tolist() == [[IGNORE, IGNORE, 20, 21, 3], [IGNORE, IGNORE, 22, 3, IGNORE]]
    assert att.tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]


def _batch():
    torch.manual_seed(1)
    x = torch.randn(2, 240, 128)
    mask = torch.ones(2, 240, dtype=torch.long)
    mask[1, 180:] = 0
    labels = [[10, 11, 12, 13], [14, 15]]
    inp, att, tgt = build_decoder_io([5, 6], labels, eos_id=3, pad_id=2)
    return dict(student_features=x, teacher_features=x, attention_mask=mask, decoder_input_ids=inp,
                decoder_attention_mask=att, decoder_targets=tgt, ctc_targets=labels)


def test_student_losses_and_grads():
    """All losses are finite and gradients reach the student and CTC head but never the frozen teacher."""
    m = tiny_cohere()
    enc = build_streaming_encoder(m.model.encoder, m.config, conv_init="truncate", left_context=14, lookahead=1)
    student = StreamingStudent(m, enc, lookaheads=(0, 1), kd_layers=(-1, 1)).train()
    out = student(**_batch())
    for k in ("aed", "ctc", "kd", "loss"):
        assert torch.isfinite(out[k]), k
    out["loss"].backward()
    assert all(p.grad is None for p in student.teacher.parameters())
    assert student.ctc_head.weight.grad is not None
    assert any(p.grad is not None for p in student.encoder.parameters())


def test_student_overfits_one_batch():
    """The student can overfit one batch, showing the loss is wired up end to end."""
    m = tiny_cohere()
    enc = build_streaming_encoder(m.model.encoder, m.config, conv_init="truncate", left_context=14, lookahead=1)
    student = StreamingStudent(m, enc, lookaheads=(1,), weights=LossWeights(aed=1.0, ctc=0.3, kd=1.0)).train()
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=3e-3)
    batch = _batch()
    first = None
    for _ in range(30):
        loss = student(**batch)["loss"]
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first


def _draw_lookaheads(student, n=1000):
    """The lookaheads `forward` would pick on `n` training steps (it samples with `student.rng`)."""
    return [student.rng.choices(student.lookaheads, weights=student.lookahead_weights)[0] for _ in range(n)]


def test_lookahead_weights_bias_sampling():
    """Lookahead weights bias which lookahead is sampled during training."""
    m = tiny_cohere()
    enc = build_streaming_encoder(m.model.encoder, m.config, left_context=14, lookahead=1)
    s = StreamingStudent(m, enc, lookaheads=(0, 1), lookahead_weights=(9.0, 1.0)).train()
    assert 850 < _draw_lookaheads(s).count(0) < 950


def test_lookahead_draws_differ_per_rank_and_are_reproducible():
    """Every rank gets its own lookahead sequence (even if the global `random` seed is identical), repeatable."""
    import random

    def student(rank):
        m = tiny_cohere()
        enc = build_streaming_encoder(m.model.encoder, m.config, left_context=14, lookahead=13)
        return StreamingStudent(m, enc, lookaheads=(0, 1, 3, 6, 13), seed=0, rank=rank).train()

    random.seed(0)  # what Trainer.set_seed does on every rank: the global generator is no help
    rank0, rank1, rank0_again = _draw_lookaheads(student(0), 50), _draw_lookaheads(student(1), 50), \
        _draw_lookaheads(student(0), 50)
    assert rank0 != rank1
    assert rank0 == rank0_again
    # and `forward` really uses it: the lookahead it reports is the first draw of that generator
    s = student(1)
    expected = s.rng.choices(s.lookaheads, weights=s.lookahead_weights)[0]
    s.rng = random.Random(1000)
    assert s(**_batch())["lookahead"].item() == expected


def test_build_student_seed_controls_new_head_initialization(monkeypatch):
    """A CLI seed must control random model weights before Trainer gets a chance to set its global seed."""
    import copy
    import train

    teacher = tiny_cohere()
    monkeypatch.setattr(train.AutoProcessor, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(train.CohereAsrForConditionalGeneration, "from_pretrained",
                        lambda *a, **k: copy.deepcopy(teacher))
    args = DistillArguments(data="x", lookaheads="0,1", kd_layers="-1")
    first, _, _ = train.build_student(args, "cpu", seed=7)
    torch.manual_seed(999)  # unrelated work between constructions must not change the CLI seed's result
    same, _, _ = train.build_student(args, "cpu", seed=7)
    different, _, _ = train.build_student(args, "cpu", seed=8)
    assert torch.equal(first.ctc_head.weight, same.ctc_head.weight)
    assert not torch.equal(first.ctc_head.weight, different.ctc_head.weight)


def test_teacher_in_bf16_with_float32_features_on_cpu():
    """A bf16 teacher fed the collator's float32 features (no bf16 training flag, CPU) still runs the KD loss."""
    m = tiny_cohere()
    enc = build_streaming_encoder(m.model.encoder, m.config, conv_init="truncate", left_context=14, lookahead=1)
    student = StreamingStudent(m, enc, lookaheads=(1,), kd_layers=(-1, 1), weights=LossWeights(1.0, 0.3, 1.0)).train()
    student.teacher.to(torch.bfloat16)
    batch = _batch()
    assert batch["teacher_features"].dtype == torch.float32
    out = student(**batch)
    assert torch.isfinite(out["kd"]) and torch.isfinite(out["loss"])
    out["loss"].backward()  # and the student still trains through it


def _tiny_student(**kwargs):
    m = tiny_cohere()
    enc = build_streaming_encoder(m.model.encoder, m.config, left_context=14, lookahead=1)
    return StreamingStudent(m, enc, lookaheads=(0, 1), **kwargs).train()


def test_only_zero_ctc_weight_needs_ddp_find_unused_parameters(tmp_path):
    """Only --ctc_weight 0 leaves parameters without gradient; zero AED/KD weights and a frozen decoder do not."""
    from datasets import Dataset

    from train import needs_find_unused_parameters

    def trainer_args(**flags):
        distill = DistillArguments(data="x", **flags)
        student = _tiny_student(weights=LossWeights(distill.aed_weight, distill.ctc_weight, distill.kd_weight),
                                freeze_decoder=distill.freeze_decoder)
        args = DistillTrainingArguments(output_dir=str(tmp_path), max_steps=1, per_device_train_batch_size=2,
                                        eval_strategy="no", eval_on_start=False, save_strategy="no",
                                        dataloader_num_workers=0, use_cpu=True, bf16=False, report_to="none")
        DistillTrainer(model=student, args=args, train_dataset=Dataset.from_dict({"duration": [1.0] * 4}),
                       data_collator=lambda rows: {}, distill_args=distill, stats={})
        return args

    assert trainer_args(ctc_weight=0.0).ddp_find_unused_parameters is True
    for flags in ({}, {"aed_weight": 0.0}, {"kd_weight": 0.0}, {"freeze_decoder": True}):
        assert not trainer_args(**flags).ddp_find_unused_parameters, flags
        assert not needs_find_unused_parameters(DistillArguments(data="x", **flags)), flags


def test_unused_parameters_match_the_condition():
    """Backward really leaves a trainable parameter without gradient exactly when `needs_find_unused_parameters`."""
    from train import needs_find_unused_parameters

    for flags in ({"ctc_weight": 0.0}, {"aed_weight": 0.0}, {"kd_weight": 0.0}, {"freeze_decoder": True}):
        distill = DistillArguments(data="x", **flags)
        student = _tiny_student(weights=LossWeights(distill.aed_weight, distill.ctc_weight, distill.kd_weight),
                                freeze_decoder=distill.freeze_decoder)
        student(**_batch())["loss"].backward()
        unused = [n for n, p in student.named_parameters() if p.requires_grad and p.grad is None]
        assert bool(unused) == needs_find_unused_parameters(distill), (flags, unused[:3])


def _flac_row(seconds=1.0):
    return {"bytes": encode_flac(0.05 * np.random.default_rng(0).standard_normal(int(16000 * seconds))), "path": None}


def test_load_distill_dataset_filters_rows(tmp_path):
    """Rows that are too short, too long or have an empty teacher label are dropped, from a folder or parquet."""
    from datasets import Dataset

    rows = {"id": ["a", "b", "c", "d"], "language": ["en"] * 4, "duration": [1.0, 0.1, 50.0, 2.0],
            "teacher_ids": [[5, 6], [7], [8], []], "teacher_text": ["x"] * 4, "reference": [None] * 4,
            "audio": [_flac_row()] * 4}
    from datasets import Audio

    ds = Dataset.from_dict(rows, features=FEATURES).cast_column("audio", Audio(decode=False))
    ds.save_to_disk(str(tmp_path / "saved"))
    ds.to_parquet(str(tmp_path / "labels.parquet"))
    for spec in (str(tmp_path / "saved"), str(tmp_path / "*.parquet")):
        loaded = load_distill_dataset(spec, 30.0)
        assert loaded["id"] == ["a"] and loaded[0]["teacher_ids"] == [5, 6]
        assert loaded[0]["audio"]["bytes"]


def test_labeller_map_writes_the_dataset_format():
    """`make_labeller` (the `Dataset.map` function of distill_dataset.py) outputs exactly `FEATURES`."""
    import argparse
    from types import SimpleNamespace

    from datasets import Audio, Dataset

    class FakeTeacher:
        tokenizer = SimpleNamespace(decode=lambda ids: "t" + "".join(map(str, ids)))

        def transcribe(self, waves, language, longest_seconds):
            return [[1, 2]] * len(waves)

    args = argparse.Namespace(audio_column="audio", text_column="text", language="en", min_seconds=0.3,
                              max_seconds=30.0, nonspeech_regex="<NOISE>", shard=0)
    source = Dataset.from_dict({"audio": [_flac_row(1.0), _flac_row(0.1), _flac_row(2.0)],
                                "text": ["hello", "short", "<NOISE>"]})
    source = source.cast_column("audio", Audio(decode=False))
    out = source.map(make_labeller(args, FakeTeacher()), batched=True, batch_size=2, with_indices=True,
                     remove_columns=source.column_names, features=FEATURES)
    assert len(out) == 2  # the 0.1 s clip is skipped
    assert out["teacher_ids"] == [[1, 2], []] and out["teacher_text"] == ["t12", "t"]  # non-speech: empty label
    assert out["reference"] == ["hello", "<NOISE>"] and out[0]["audio"]["bytes"]


def test_collator_gain_aug_student_only():
    """Gain augmentation perturbs the student features but leaves the teacher features untouched."""
    import io
    import random

    import soundfile as sf
    from transformers import AutoProcessor

    from distill_dataset import collate_items

    proc = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
    proc.feature_extractor.dither = 0.0
    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    sf.write(buf, (0.05 * rng.standard_normal(16000 * 2)).astype("float32"), 16000, format="FLAC")
    items = [(buf.getvalue(), {"language": "en", "duration": 2.0, "teacher_ids": [5, 6]})]
    stats = {"mean": [-10.0] * 128, "std": [3.0] * 128}
    plain = collate_items(items, proc, stats, "causal", rng=random.Random(0))
    aug = collate_items(items, proc, stats, "causal", rng=random.Random(0), prior_range=(10, 100), gain_db=10)
    assert torch.equal(plain["teacher_features"], aug["teacher_features"])  # teacher: original audio, untouched
    assert (plain["student_features"] - aug["student_features"]).abs().max() > 1e-2  # student: perturbed


def test_trainer_trains_and_writes_loadable_checkpoint(tmp_path):
    """`DistillTrainer` runs on a tiny student; its checkpoint has the layout `model.CohereStreaming` reads and
    can be resumed from."""
    import json

    from datasets import Dataset
    from safetensors.torch import load_file

    from train import load_student_checkpoint

    def make_student():
        m = tiny_cohere()
        enc = build_streaming_encoder(m.model.encoder, m.config, conv_init="truncate", left_context=14, lookahead=1)
        return StreamingStudent(m, enc, lookaheads=(0, 1), kd_layers=(-1, 1)).train()

    batch = _batch()
    distill = DistillArguments(data="unused", conv_init="truncate", left=14)
    args = DistillTrainingArguments(output_dir=str(tmp_path / "run"), max_steps=3, per_device_train_batch_size=2,
                                    eval_strategy="no", eval_on_start=False, save_steps=2, logging_steps=1,
                                    dataloader_num_workers=0, use_cpu=True, bf16=False, report_to="none")
    student = make_student()
    stats = {"mean": [0.0] * 128, "std": [1.0] * 128, "frames": 1}
    data = Dataset.from_dict({"duration": [1.0] * 6})
    trainer = DistillTrainer(model=student, args=args, train_dataset=data, data_collator=lambda rows: dict(batch),
                             distill_args=distill, stats=stats)
    trainer.train()
    trainer.save_model()
    ckpt = tmp_path / "run"
    for folder in (ckpt, ckpt / "checkpoint-2"):
        assert {"student.safetensors", "config.json", "mel_stats.json"} <= {p.name for p in folder.iterdir()}
    final, intermediate = load_file(str(ckpt / "student.safetensors")), load_file(str(ckpt / "checkpoint-2" / "student.safetensors"))
    assert all(v.dtype == torch.bfloat16 for v in final.values())  # the release artifact is bf16 ...
    assert all(v.dtype == torch.float32 for v in intermediate.values())  # ... resumable checkpoints are fp32
    keys = final.keys()
    assert not any(k.startswith("cohere.model.encoder.") for k in keys)  # the public teacher is not stored
    assert any(k.startswith("encoder.") for k in keys) and any(k.startswith("ctc_head.") for k in keys)
    config = json.loads((ckpt / "config.json").read_text())
    assert config["args"]["conv_init"] == "truncate" and config["encoder"]["sliding_window"] == 15
    logged = [h for h in trainer.state.log_history if "aed_loss" in h]
    assert logged and "ctc_loss" in logged[0] and "kd_loss" in logged[0]
    fresh = make_student()
    load_student_checkpoint(fresh, str(ckpt))  # the final bf16 model loads, rounded to bf16
    assert torch.equal(fresh.ctc_head.weight, student.ctc_head.weight.to(torch.bfloat16).float())


def test_resume_restores_exact_fp32_weights(tmp_path):
    """Resuming from `checkpoint-N/` gives back the weights saved at step N bit for bit (no bf16 rounding)."""
    from datasets import Dataset
    from safetensors.torch import load_file

    def run(max_steps, resume=None):
        m = tiny_cohere()
        enc = build_streaming_encoder(m.model.encoder, m.config, conv_init="truncate", left_context=14, lookahead=1)
        student = StreamingStudent(m, enc, lookaheads=(0, 1)).train()
        args = DistillTrainingArguments(output_dir=str(tmp_path / "run"), max_steps=max_steps,
                                        per_device_train_batch_size=2, eval_strategy="no", eval_on_start=False,
                                        save_steps=2, logging_steps=1, dataloader_num_workers=0, use_cpu=True,
                                        bf16=False, report_to="none")
        batch = _batch()
        trainer = DistillTrainer(model=student, args=args, train_dataset=Dataset.from_dict({"duration": [1.0] * 6}),
                                 data_collator=lambda rows: dict(batch), distill_args=DistillArguments(data="x"),
                                 stats={"mean": [0.0] * 128, "std": [1.0] * 128, "frames": 1})
        return student, trainer

    _, first = run(2)
    first.train()
    saved = load_file(str(tmp_path / "run" / "checkpoint-2" / "student.safetensors"))
    student, second = run(3)
    second._load_from_checkpoint(str(tmp_path / "run" / "checkpoint-2"))  # what `--resume_from_checkpoint` calls
    state = student.state_dict()
    for name, weights in saved.items():
        assert torch.equal(state[name], weights), name  # exact fp32 equality
    assert any(v.abs().max() > 0 for v in saved.values()) and not any(v.dtype == torch.bfloat16 for v in saved.values())


def test_optimizer_follows_training_arguments(tmp_path):
    """--optim / --adam_epsilon / --adam_beta2 / --weight_decay reach the optimizer, the head has its own lr and
    biases and norm weights get no weight decay."""
    m = tiny_cohere()
    enc = build_streaming_encoder(m.model.encoder, m.config, conv_init="truncate", left_context=14, lookahead=1)
    student = StreamingStudent(m, enc, lookaheads=(0, 1))
    distill = DistillArguments(data="x", head_lr=0.01)
    args = DistillTrainingArguments(output_dir=str(tmp_path), optim="adamw_torch", adam_epsilon=1e-6, adam_beta2=0.9,
                                    weight_decay=0.1, learning_rate=1e-4, use_cpu=True, bf16=False, report_to="none")
    opt = build_optimizer(student, distill, args)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.defaults["eps"] == 1e-6 and opt.defaults["betas"][1] == 0.9
    by_lr = {(g["lr"], g["weight_decay"]) for g in opt.param_groups}
    assert by_lr == {(1e-4, 0.1), (1e-4, 0.0), (0.01, 0.0)}
    head_group = next(g for g in opt.param_groups if g["lr"] == 0.01)
    assert {id(p) for p in head_group["params"]} == {id(p) for p in student.ctc_head.parameters()}
    no_decay = next(g for g in opt.param_groups if g["lr"] == 1e-4 and g["weight_decay"] == 0.0)
    assert all(p.ndim <= 1 or "bias" in n for n, p in student.named_parameters() if any(p is q for q in no_decay["params"]))
    covered = sum(len(g["params"]) for g in opt.param_groups)
    assert covered == sum(p.requires_grad for p in student.parameters())  # every trainable parameter, once
    args_sgd = DistillTrainingArguments(output_dir=str(tmp_path), optim="sgd", use_cpu=True, bf16=False, report_to="none")
    assert isinstance(build_optimizer(student, distill, args_sgd), torch.optim.SGD)  # --optim is honoured


def test_trainer_evaluate_logs_wer_metrics(tmp_path, monkeypatch):
    """`DistillTrainer.evaluate` turns the per-set, per-lookahead WERs into flat `eval_*` metrics in the log."""
    import train

    monkeypatch.setattr(train, "load_eval_set", lambda spec, n=None: ([], "en"))
    monkeypatch.setattr(train, "evaluate", lambda *a, **k: {"R0": {"aed_wer": 5.0, "ctc_wer": 9.0}})
    from datasets import Dataset

    m = tiny_cohere()
    enc = build_streaming_encoder(m.model.encoder, m.config, conv_init="truncate", left_context=14, lookahead=1)
    student = StreamingStudent(m, enc, lookaheads=(0, 1)).train()
    distill = DistillArguments(data="unused", eval_sets="librispeech:test.other")
    args = DistillTrainingArguments(output_dir=str(tmp_path), max_steps=1, per_device_train_batch_size=2,
                                    eval_steps=1, save_strategy="no", dataloader_num_workers=0, use_cpu=True,
                                    bf16=False, report_to="none")
    batch = _batch()
    trainer = DistillTrainer(model=student, args=args, train_dataset=Dataset.from_dict({"duration": [1.0] * 4}),
                             data_collator=lambda rows: dict(batch), distill_args=distill, stats={})
    trainer.train()
    evals = [h for h in trainer.state.log_history if "eval_librispeech_test.other_R0_aed_wer" in h]
    assert len(evals) == 2 and evals[0]["eval_librispeech_test.other_R0_ctc_wer"] == 9.0  # at start and at the end


def test_best_model_is_restored_from_student_checkpoint(tmp_path, monkeypatch):
    """A run selecting step 1 as best must restore it rather than silently releasing the final step's weights."""
    import train
    from datasets import Dataset
    from safetensors.torch import load_file

    monkeypatch.setattr(train, "load_eval_set", lambda spec, n=None: ([], "en"))
    scores = iter([1.0, 2.0])
    monkeypatch.setattr(train, "evaluate", lambda *a, **k: {"R0": {"aed_wer": next(scores), "ctc_wer": 9.0}})
    args = DistillTrainingArguments(output_dir=str(tmp_path), max_steps=2, per_device_train_batch_size=2,
                                    eval_steps=1, save_steps=1, eval_on_start=False, load_best_model_at_end=True,
                                    metric_for_best_model="eval_librispeech_test.other_R0_aed_wer",
                                    greater_is_better=False, dataloader_num_workers=0, use_cpu=True,
                                    bf16=False, report_to="none")
    student = _tiny_student()
    trainer = DistillTrainer(model=student, args=args, train_dataset=Dataset.from_dict({"duration": [1.0] * 4}),
                             data_collator=lambda rows: _batch(), distill_args=DistillArguments(data="x"), stats={})
    trainer.train()
    best = load_file(str(tmp_path / "checkpoint-1" / "student.safetensors"))
    last = load_file(str(tmp_path / "checkpoint-2" / "student.safetensors"))
    assert trainer.state.best_model_checkpoint == str(tmp_path / "checkpoint-1")
    assert not torch.equal(best["ctc_head.weight"], last["ctc_head.weight"])
    assert all(torch.equal(student.state_dict()[name], value) for name, value in best.items())


def test_evaluation_metrics_reach_nonzero_rank(tmp_path, monkeypatch):
    """Metrics used by best-model selection and callbacks must be the same on every DDP rank."""
    import train

    trainer, _ = _trainer_with_fake_eval(tmp_path, monkeypatch, max_steps=1, eval_steps=1,
                                          metric_for_best_model="eval_librispeech_test.other_R0_aed_wer")
    metric = {"eval_librispeech_test.other_R0_aed_wer": 5.0}
    monkeypatch.setattr(trainer, "is_world_process_zero", lambda: False)
    monkeypatch.setattr(train, "broadcast_object_list", lambda objects, from_process: [metric])
    received = trainer.evaluate()
    assert received == metric
    assert trainer._determine_best_metric(received, None) is True


def test_trainer_evaluation_honors_full_precision(tmp_path, monkeypatch):
    """Disabling training mixed precision also disables the evaluation autocast that bypasses Trainer's forward."""
    import train

    trainer, _ = _trainer_with_fake_eval(tmp_path, monkeypatch, max_steps=1, eval_steps=1)
    seen = []
    monkeypatch.setattr(train, "evaluate", lambda *a, **k: (seen.append(k["autocast_dtype"]),
                                                         {"R0": {"aed_wer": 5.0, "ctc_wer": 9.0}})[1])
    trainer.evaluate()
    assert seen == [None]


def test_resume_restores_private_lookahead_generator(tmp_path, monkeypatch):
    """A resumed run continues its lookahead draws instead of repeating the beginning of the recipe."""
    trainer, _ = _trainer_with_fake_eval(tmp_path, monkeypatch, max_steps=1, eval_steps=1)
    _draw_lookaheads(trainer.model, 20)
    trainer._save_rng_state(str(tmp_path))
    expected = _draw_lookaheads(trainer.model, 20)
    resumed, _ = _trainer_with_fake_eval(tmp_path, monkeypatch, max_steps=1, eval_steps=1)
    resumed._load_rng_state(str(tmp_path))
    assert _draw_lookaheads(resumed.model, 20) == expected


def _trainer_with_fake_eval(tmp_path, monkeypatch, max_steps, eval_steps, **extra):
    """A trainer whose `evaluate` WER function is faked; returns it and the steps at which evals ran."""
    import train
    from datasets import Dataset

    monkeypatch.setattr(train, "load_eval_set", lambda spec, n=None: ([], "en"))
    monkeypatch.setattr(train, "evaluate", lambda *a, **k: {"R0": {"aed_wer": 5.0, "ctc_wer": 9.0}})
    args = DistillTrainingArguments(output_dir=str(tmp_path), max_steps=max_steps, per_device_train_batch_size=2,
                                    eval_steps=eval_steps, eval_on_start=False, save_strategy="no", use_cpu=True,
                                    dataloader_num_workers=0, bf16=False, report_to="none", **extra)
    batch = _batch()
    trainer = DistillTrainer(model=_tiny_student(), args=args, train_dataset=Dataset.from_dict({"duration": [1.0] * 4}),
                             data_collator=lambda rows: dict(batch), distill_args=DistillArguments(data="x"), stats={})
    steps = []
    real = trainer.evaluate
    trainer.evaluate = lambda *a, **k: (steps.append(trainer.state.global_step), real(*a, **k))[1]
    return trainer, steps


def test_final_evaluation_runs_when_trainer_did_not_evaluate_the_last_step(tmp_path, monkeypatch):
    """If Trainer's own schedule skipped the last step (here: --eval_delay 10 holds back every eval of a 3-step
    run), `evaluate_at_end` evaluates it, once."""
    trainer, steps = _trainer_with_fake_eval(tmp_path, monkeypatch, max_steps=3, eval_steps=2, eval_delay=10)
    trainer.train()
    assert steps == []
    assert trainer.evaluate_at_end() is True and steps == [3]
    assert trainer.evaluate_at_end() is False and steps == [3]


def test_no_duplicate_final_evaluation_with_step_strategy(tmp_path, monkeypatch):
    """--eval_steps 2 with 3 steps: the final step is evaluated exactly once, whether by Trainer or `main`."""
    trainer, steps = _trainer_with_fake_eval(tmp_path, monkeypatch, max_steps=3, eval_steps=2)
    trainer.train()
    trainer.evaluate_at_end()
    assert steps == [2, 3]
    trainer, steps = _trainer_with_fake_eval(tmp_path, monkeypatch, max_steps=2, eval_steps=2)
    trainer.train()
    assert trainer.evaluate_at_end() is False and steps == [2]


def test_no_final_evaluation_when_evaluation_is_off(tmp_path, monkeypatch):
    trainer, steps = _trainer_with_fake_eval(tmp_path, monkeypatch, max_steps=1, eval_steps=5, eval_strategy="no")
    trainer.train()
    assert trainer.evaluate_at_end() is False and steps == []


def test_final_model_is_pushed_exactly_once(tmp_path, monkeypatch):
    """With --push_to_hub the final model is saved and uploaded once (save_model itself would push, too)."""
    import huggingface_hub
    from transformers import Trainer

    uploads = []
    monkeypatch.setattr(Trainer, "init_hf_repo", lambda self, token=None: setattr(self, "hub_model_id", "u/m"))
    monkeypatch.setattr(huggingface_hub.HfApi, "upload_folder", lambda self, **k: uploads.append(k))
    monkeypatch.setattr("transformers.trainer.hf_api", lambda: huggingface_hub.HfApi(), raising=False)
    trainer, _ = _trainer_with_fake_eval(tmp_path, monkeypatch, max_steps=1, eval_steps=5, push_to_hub=True,
                                         hub_model_id="u/m", hub_strategy="end")
    trainer.train()
    uploads.clear()
    trainer.save_and_push_final_model()
    assert len(uploads) == 1
    assert (tmp_path / "student.safetensors").exists() and (tmp_path / "config.json").exists()


def test_final_model_without_push_only_saves(tmp_path, monkeypatch):
    trainer, _ = _trainer_with_fake_eval(tmp_path, monkeypatch, max_steps=1, eval_steps=5)
    trainer.train()
    trainer.save_and_push_final_model()
    assert (tmp_path / "student.safetensors").exists()
