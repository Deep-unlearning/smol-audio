"""Distil the offline Cohere Transcribe model into a cache-aware streaming student, with `transformers.Trainer`.

The recipe, in plain words
--------------------------
* Teacher: the original offline Cohere Transcribe model. It sees the whole utterance at once and is accurate,
  but it cannot run live. It is frozen and never updated.
* Student: the same model with its encoder swapped for a *streaming* encoder (each frame only sees a limited
  window of past audio plus a few "lookahead" frames), so it can transcribe while someone is still talking.
  The student starts from the teacher's weights, so training only has to adapt it, not teach it from scratch.
* Training data: audio that `distill_dataset.py` already transcribed with the teacher. We train on the teacher's
  output (token ids) instead of human transcripts, so the student learns to imitate the teacher.
* Three losses are added up (weights are CLI flags):
    - AED loss  (--aed_weight): the student's decoder must predict the teacher's tokens (cross-entropy).
    - CTC loss  (--ctc_weight): a small extra output head on the encoder, trained to predict the same tokens.
    - KD loss   (--kd_weight):  encoder hidden states at a few layers (--kd_layers) must match the teacher's
                                encoder states on the same audio (knowledge distillation).
  Each step picks a random lookahead from --lookaheads so one student works at several latency settings.

Why `transformers.Trainer`
--------------------------
Distillation here is an ordinary supervised training loop with a custom loss, so `Trainer` gives us everything
else for free: multi-GPU (`torchrun` / `accelerate launch`), bf16, gradient accumulation, LR schedule, logging
(`--report_to trackio`), checkpoints, resume (`--resume_from_checkpoint`) and pushing to the
Hub (`--push_to_hub --hub_model_id ...`). Only three things are custom: `compute_loss` (the distillation losses),
`evaluate` (WER of the student's decoders, which needs generation) and the checkpoint layout, which is what
`model.CohereStreaming` loads. TRL has no trainer for encoder-decoder audio distillation, so it is not used.

Layout of this file: 1. the student model, 2. command-line arguments, 3. evaluation, 4. the trainer, 5. main.

Example
-------
    python train.py \
        --data "hf://datasets/<owner>/<repo>/runs/relabel-pilot/**/*.parquet" \
        --max_steps 500 --output_dir checkpoint/train_pilot

    # several GPUs: the same flags under torchrun (or `accelerate launch train.py ...`)
    torchrun --nproc_per_node 8 train.py --data "my_labels/*" --max_steps 2000 --report_to trackio

Every `TrainingArguments` flag works (`--per_device_train_batch_size`, `--gradient_accumulation_steps`,
`--save_steps`, ...); `python train.py --help` lists them with the distillation flags below. That includes the
optimizer flags (`--optim`, `--adam_epsilon`, `--adam_beta1/2`, `--weight_decay`): the optimizer is the one
`--optim` names, with two extras of this recipe: the CTC head gets its own learning rate (`--head_lr`) and biases and
norm weights get no weight decay.

Checkpoint precision: the intermediate `checkpoint-N/` folders store the student in float32, so
`--resume_from_checkpoint` restores the weights exactly. The final model in `--output_dir` is stored in bfloat16
(half the size; this is the release artifact `model.CohereStreaming` loads, which converts back to float32).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import jiwer
import numpy as np
import torch
from accelerate.utils import broadcast_object_list
from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from transformers import (AutoProcessor, CohereAsrForConditionalGeneration, HfArgumentParser, Trainer,
                          TrainingArguments, set_seed)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer

from distill_dataset import (IGNORE, DistillCollator, audio_bytes, build_decoder_io,  # noqa: F401 (re-exported)
                             decode_flac, load_distill_dataset)
from model import (COHERE_MODEL_ID, aed_transcribe, build_streaming_encoder, ctc_greedy, estimate_stats,
                   save_stats)


# ============================================================================================================
# 1. The student model
#
# `StreamingStudent` bundles the pieces needed to train the streaming encoder and computes the loss.
# * teacher encoder - the frozen offline Cohere encoder, fed its native per-utterance-normalised features
# * student encoder - `CohereStreamingEncoder` (initialised from the teacher by `build_streaming_encoder`), fed
#                     streamable (causal-normalised) features, chunked-limited attention with a lookahead sampled
#                     per batch from `lookaheads` (multi-lookahead training: one model, many latencies)
# * CTC head        - new linear head on the student encoder for low-latency first-pass partials
# * AED decoder     - Cohere's decoder + `proj_out`, second pass at the endpoint over student encoder states
#
# Loss = w_aed * CE(decoder | student states) + w_ctc * CTC + w_kd * feature-KD(student, teacher).
# Labels are the teacher's own offline transcripts as token ids (sequence-level KD), so no retokenisation.
# ============================================================================================================


@dataclass
class LossWeights:
    """How much each loss term counts in the total (0 disables the term).

    Only a zero CTC weight leaves parameters without a gradient: the CTC head is then never run. Plain training
    does not mind, but multi-GPU DDP needs `find_unused_parameters=True`; `DistillTrainer` turns that on by itself
    (see `needs_find_unused_parameters`). A zero AED weight still runs the decoder and multiplies its loss by 0.0
    (zero gradients, not missing ones), a zero KD weight only skips the frozen teacher (the student encoder is still
    used by the other losses), and `--freeze_decoder` sets `requires_grad=False`, which DDP ignores.
    """

    aed: float = 1.0
    ctc: float = 0.3
    kd: float = 1.0


def needs_find_unused_parameters(distill: "DistillArguments") -> bool:
    """True if some trainable parameters get no gradient, which DDP rejects unless `find_unused_parameters=True`.

    That is the case for --ctc_weight 0 only (the CTC head is skipped). --aed_weight 0 and --kd_weight 0 leave every
    trainable parameter in the graph, and frozen parameters (--freeze_decoder, the teacher) are ignored by DDP.
    """
    return distill.ctc_weight == 0


def masked_kd(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor, kind: str = "cosine") -> torch.Tensor:
    """Frame-level feature distillation over valid frames; lengths may differ by a frame (causal vs symmetric
    subsampling padding), so both are truncated to the shorter one.

    Args:
        student, teacher: encoder states (B, T, D).
        mask: (B, T) with 1 for real frames.
        kind: "cosine" (1 - cosine similarity) or "mse".
    """
    t = min(student.shape[1], teacher.shape[1], mask.shape[1])
    s, te, m = student[:, :t], teacher[:, :t].detach(), mask[:, :t].bool()
    if kind == "cosine":
        per = 1.0 - F.cosine_similarity(s.float(), te.float(), dim=-1)
    elif kind == "mse":
        per = ((s.float() - te.float()) ** 2).mean(-1)
    else:
        raise ValueError(kind)
    return per[m].mean()


class StreamingStudent(nn.Module):
    """Student encoder + CTC head + Cohere decoder, with the combined distillation loss in `forward`."""

    def __init__(
        self,
        cohere_model: nn.Module,
        student_encoder: nn.Module,
        lookaheads: tuple[int, ...] = (0, 1, 3, 6, 13),
        lookahead_weights: tuple[float, ...] | None = None,
        weights: LossWeights = LossWeights(),
        kd_layers: tuple[int, ...] = (-1,),
        kd_kind: str = "cosine",
        freeze_decoder: bool = False,
        seed: int = 0,
        rank: int = 0,
    ):
        super().__init__()
        self.cohere = cohere_model  # holds the teacher encoder (frozen) + decoder/proj_out (student 2nd pass)
        self.encoder = student_encoder
        self.lookaheads = tuple(lookaheads)
        self.lookahead_weights = tuple(lookahead_weights) if lookahead_weights else None
        if self.lookahead_weights and len(self.lookahead_weights) != len(self.lookaheads):
            raise ValueError("lookahead_weights must match lookaheads")
        self.weights = weights
        self.kd_layers = tuple(kd_layers)
        self.kd_kind = kd_kind
        # Lookahead draws come from our own generator, not the global `random` module: `Trainer` seeds the global
        # one identically on every GPU, which would make all GPUs pick the same lookahead at every step. Adding
        # 1000 * rank gives each rank a different but reproducible sequence.
        self.rng = random.Random(seed + 1000 * rank)
        vocab = cohere_model.config.vocab_size
        self.blank_id = vocab  # CTC blank appended after the text vocabulary
        self.ctc_head = nn.Linear(student_encoder.config.hidden_size, vocab + 1)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        if freeze_decoder:
            for p in list(self.cohere.model.decoder.parameters()) + list(self.cohere.proj_out.parameters()):
                p.requires_grad_(False)

    @property
    def teacher(self) -> nn.Module:
        """The frozen offline encoder inside the Cohere model."""
        return self.cohere.model.encoder

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()  # frozen BatchNorm statistics, no dropout
        return self

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        decoder_targets: torch.Tensor,
        ctc_targets: list[list[int]],
        lookahead: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the student on a batch and return the losses ("aed", "ctc", "kd", "loss") and the "lookahead" used.

        Shapes: features (B, T, n_mels); attention_mask (B, T); decoder_* (B, N); ctc_targets is one list of
        token ids per sample. `lookahead=None` samples one from `self.lookaheads` while training and uses
        the largest (last) one in eval mode.
        """
        if lookahead is None:
            if self.training:
                lookahead = self.rng.choices(self.lookaheads, weights=self.lookahead_weights)[0]
            else:
                lookahead = self.lookaheads[-1]
        need_hidden = self.weights.kd > 0 and any(i != -1 for i in self.kd_layers)
        enc = self.encoder(
            student_features, attention_mask=attention_mask, num_lookahead_tokens=lookahead,
            output_hidden_states=need_hidden,
        )
        frame_mask = enc.attention_mask
        losses = {}

        # 1) AED second pass over student states
        dec = self.cohere.model.decoder(
            input_ids=decoder_input_ids, attention_mask=decoder_attention_mask,
            encoder_hidden_states=enc.last_hidden_state, encoder_attention_mask=frame_mask,
        )
        logits = self.cohere.proj_out(dec.last_hidden_state)
        losses["aed"] = F.cross_entropy(logits.float().flatten(0, 1), decoder_targets.flatten(), ignore_index=IGNORE)

        # 2) CTC first pass
        if self.weights.ctc > 0:
            log_probs = self.ctc_head(enc.last_hidden_state).float().log_softmax(-1).transpose(0, 1)  # (T, B, V)
            in_lens = frame_mask.sum(-1)
            tgt_lens = torch.tensor([len(t) for t in ctc_targets], device=log_probs.device)
            flat = torch.tensor([x for t in ctc_targets for x in t], dtype=torch.long, device=log_probs.device)
            losses["ctc"] = F.ctc_loss(log_probs, flat, in_lens, tgt_lens, blank=self.blank_id, zero_infinity=True)

        # 3) feature KD against the frozen offline teacher
        if self.weights.kd > 0:
            # The teacher may be stored in bf16 (see `main`) while the collator produces float32 features:
            # convert the input to whatever dtype the teacher uses.
            teacher_dtype = next(self.teacher.parameters()).dtype
            with torch.no_grad():
                t_out = self.teacher(teacher_features.to(teacher_dtype), attention_mask=attention_mask,
                                     output_hidden_states=need_hidden)
            kd = []
            for i in self.kd_layers:
                s_h = enc.last_hidden_state if i == -1 else enc.hidden_states[i]
                t_h = t_out.last_hidden_state if i == -1 else t_out.hidden_states[i]
                kd.append(masked_kd(s_h, t_h, frame_mask, self.kd_kind))
            losses["kd"] = torch.stack(kd).mean()

        w = self.weights
        losses["loss"] = w.aed * losses["aed"] + w.ctc * losses.get("ctc", 0.0) + w.kd * losses.get("kd", 0.0)
        losses["lookahead"] = torch.tensor(lookahead)
        return losses


def parse_int_list(text: str) -> tuple[int, ...]:
    """'0,3,13' -> (0, 3, 13)."""
    return tuple(int(x) for x in text.split(","))


def build_student(args: "DistillArguments", device, seed: int = 0, rank: int = 0) -> tuple[StreamingStudent, object, object]:
    """Load the teacher, build the streaming encoder and wrap everything as a `StreamingStudent`.

    `seed` and `rank` (the training seed and this process's GPU index) make each GPU draw its own lookaheads.
    Returns (student, processor, streaming encoder).
    """
    set_seed(seed)  # also seed newly initialised parameters (the CTC head), before Trainer is constructed
    processor = AutoProcessor.from_pretrained(COHERE_MODEL_ID)
    cohere = CohereAsrForConditionalGeneration.from_pretrained(COHERE_MODEL_ID, dtype=torch.float32).to(device)
    lookaheads = parse_int_list(args.lookaheads)
    encoder = build_streaming_encoder(cohere.model.encoder, cohere.config, conv_init=args.conv_init,
                                      left_context=args.left, lookahead=max(lookaheads))
    lookahead_weights = (tuple(float(x) for x in args.lookahead_weights.split(","))
                         if args.lookahead_weights else None)
    student = StreamingStudent(
        cohere, encoder, lookaheads=lookaheads, lookahead_weights=lookahead_weights,
        weights=LossWeights(aed=args.aed_weight, ctc=args.ctc_weight, kd=args.kd_weight),
        kd_layers=parse_int_list(args.kd_layers), freeze_decoder=args.freeze_decoder, seed=seed, rank=rank,
    ).to(device).train()
    return student, processor, encoder


def load_student_checkpoint(student: StreamingStudent, init_from: str) -> None:
    """Load student weights from a Hub repo or local directory (to continue a previous run)."""
    ckpt_dir = Path(init_from) if Path(init_from).exists() else Path(snapshot_download(init_from))
    state = {k: v.float() for k, v in load_file(str(ckpt_dir / "student.safetensors")).items()}
    missing, unexpected = student.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("cohere.model.encoder.")]  # checkpoints omit the teacher
    if missing or unexpected:
        raise RuntimeError(f"--init_from mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")


def estimate_training_mel_stats(dataset, processor, seed: int) -> dict:
    """Mel mean/std over ~300 random training utterances; the causal normaliser needs them at inference.

    Every rank computes this with the same seed on the same data, so no broadcast between ranks is needed.
    """
    picks = random.Random(seed).sample(range(len(dataset)), min(300, len(dataset)))
    audio = [torch.from_numpy(decode_flac(audio_bytes(a))) for a in dataset.select(picks)["audio"]]
    return estimate_stats(processor.feature_extractor, audio)


# ============================================================================================================
# 2. Command-line arguments
#
# Two groups, both parsed by `HfArgumentParser`: `DistillArguments` (what is specific to this recipe) and
# `DistillTrainingArguments` (all of `transformers.TrainingArguments`, with defaults suited to this recipe).
# ============================================================================================================


@dataclass
class DistillArguments:
    # data
    data: str = field(metadata={"help": "labelled dataset made by distill_dataset.py: a Hub dataset id, a folder "
                                        "(glob ok) or parquet files (local glob or hf://datasets/<repo>/<glob>); "
                                        "separate several with commas"})
    max_seconds: float = field(default=30.0, metadata={"help": "skip utterances longer than this"})
    # streaming student (what the model looks like)
    lookaheads: str = field(default="0,1,3,6,13", metadata={
        "help": "lookahead sizes (future frames the encoder may see) sampled during training"})
    lookahead_weights: Optional[str] = field(default=None, metadata={
        "help": "sampling weights for --lookaheads, e.g. 2,1.5,1,1,1 (default: uniform)"})
    left: int = field(default=70, metadata={"help": "left context: past frames each frame may attend to"})
    conv_init: Literal["shift", "truncate"] = field(default="truncate", metadata={
        "help": "how the encoder's convolutions are made causal when initialised from the teacher"})
    norm: Literal["causal", "global"] = field(default="causal", metadata={
        "help": "feature normalisation: causal (running statistics, streamable) or global statistics"})
    init_from: Optional[str] = field(default=None, metadata={
        "help": "start from a student checkpoint (Hub repo or local dir) instead of the teacher; to continue an "
                "interrupted run use --resume_from_checkpoint instead"})
    freeze_decoder: bool = field(default=False, metadata={"help": "train only the encoder (and CTC head)"})
    grad_ckpt: bool = field(default=True, metadata={
        "help": "gradient checkpointing of the student encoder: less memory, slower (--no_grad_ckpt to turn off)"})
    # losses
    aed_weight: float = field(default=1.0, metadata={"help": "weight of the decoder cross-entropy loss"})
    ctc_weight: float = field(default=0.3, metadata={"help": "weight of the CTC head loss (0 skips the CTC head; "
                                                             "DDP then runs with find_unused_parameters=True, "
                                                             "which no other zero weight needs)"})
    kd_weight: float = field(default=1.0, metadata={"help": "weight of the encoder hidden-state matching loss"})
    kd_layers: str = field(default="-1,12,24,36", metadata={
        "help": "encoder layers whose states are matched to the teacher"})
    # optimisation
    head_lr: float = field(default=1e-3, metadata={"help": "learning rate of the freshly initialised CTC head "
                                                           "(--learning_rate is for the pretrained weights). The "
                                                           "optimizer itself follows --optim, --adam_beta1/2, "
                                                           "--adam_epsilon and --weight_decay"})
    # augmentation (student input only)
    prior_range: Optional[str] = field(default=None, metadata={
        "help": "train-time causal-norm warm-up prior range in frames, e.g. 10,100"})
    gain_db: float = field(default=0.0, metadata={"help": "random volume change of +/- this many dB"})
    # evaluation (every --eval_steps; --eval_steps/--eval_strategy are TrainingArguments)
    eval_n: int = field(default=100, metadata={"help": "utterances per eval set"})
    eval_lookaheads: str = field(default="0,3,13", metadata={"help": "lookaheads to evaluate"})
    eval_sets: str = field(default="librispeech:test.other", metadata={
        "help": "comma-separated eval sets, e.g. librispeech:test.other,fleurs:fr,fleurs:de"})
    eval_prior: int = field(default=100, metadata={"help": "causal-norm warm-up prior (frames) used by evals"})


@dataclass
class DistillTrainingArguments(TrainingArguments):
    """`TrainingArguments` with this recipe's defaults (every other `TrainingArguments` flag is unchanged)."""

    output_dir: str = field(default="checkpoint/train", metadata={
        "help": "final model (student.safetensors in bf16 + config.json + mel_stats.json, loadable by "
                "CohereStreaming); intermediate checkpoint-N/ folders hold float32 weights for exact resume"})
    max_steps: int = field(default=2000, metadata={"help": "number of training steps"})
    learning_rate: float = field(default=5e-5, metadata={"help": "learning rate of the pretrained weights"})
    warmup_steps: int = field(default=200, metadata={"help": "steps of linear learning-rate warm-up"})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "decay after warm-up"})
    weight_decay: float = 0.01
    adam_beta2: float = 0.98
    per_device_train_batch_size: int = field(default=16, metadata={
        "help": "utterances per GPU per step; they are grouped by duration to keep padding low"})
    train_sampling_strategy: str = "group_by_length"
    length_column_name: str = "duration"
    dataloader_num_workers: int = field(default=8, metadata={
        "help": "DataLoader worker processes for audio decoding + features (0 = do it inline)"})
    bf16: bool = field(default_factory=lambda: torch.cuda.is_available() and
                       torch.cuda.is_bf16_supported(including_emulation=False),
                       metadata={"help": "bf16 mixed precision (on with a native bf16-capable GPU)"})
    remove_unused_columns: bool = False  # the collator needs every dataset column
    eval_strategy: str = "steps"
    eval_steps: float = field(default=250, metadata={"help": "evaluate every N steps (and at the end)"})
    eval_on_start: bool = field(default=True, metadata={"help": "evaluate before any update (the baseline)"})
    logging_steps: float = 10
    report_to: str = "trackio"
    project: str = "cohere-streaming"
    save_steps: float = 500
    save_total_limit: Optional[int] = 2
    ddp_timeout: int = 3600  # evals run on rank 0 only and take minutes: generous collective timeout
    hub_private_repo: Optional[bool] = True
    seed: int = 0


# ============================================================================================================
# 3. Evaluation
#
# Word error rate (WER) computed the way the Open ASR Leaderboard does, so numbers are comparable with the
# benchmark: reference and hypothesis are normalised first (case, punctuation, spelling variants).
# ============================================================================================================

# The evaluation code below (FLEURS, CHAR_LANGS, Utterance, load_eval_set, wer) intentionally mirrors
# benchmark/eval/data.py and benchmark/eval/metrics.py, so that training does not import benchmark/. Keep both in sync.
ESB_REPO = "hf-audio/esb-datasets-test-only-sorted"  # Open ASR Leaderboard test sets: "librispeech:test.other", ...
FLEURS = {"en": "en_us", "fr": "fr_fr", "de": "de_de", "es": "es_419", "it": "it_it", "pt": "pt_br", "nl": "nl_nl",
          "pl": "pl_pl", "el": "el_gr", "ar": "ar_eg", "ja": "ja_jp", "zh": "cmn_hans_cn", "ko": "ko_kr", "vi": "vi_vn"}
CHAR_LANGS = {"ja", "zh", "ko"}  # no reliable word boundaries: scored per character (CER) instead of per word


@dataclass
class Utterance:
    """One eval utterance: an id, its audio (16 kHz mono float32) and the reference transcript."""

    id: str
    audio: np.ndarray
    text: str

    @property
    def seconds(self) -> float:
        return len(self.audio) / 16000


def load_eval_set(spec: str, n: int | None = None, max_seconds: float = 30.0) -> tuple[list[Utterance], str]:
    """'librispeech:test.other' (Open ASR sets) or 'fleurs:fr' -> (`n` utterances spread evenly, language code)."""
    name, arg = spec.split(":")
    if name == "fleurs":
        files = f"hf://datasets/google/fleurs/parquet-data/{FLEURS[arg]}/test-*.parquet"
        ds, text_column, language = load_dataset("parquet", data_files=files, split="train"), "transcription", arg
    else:
        ds, text_column, language = load_dataset(ESB_REPO, name, split=arg), "text", "en"
        ds = ds.filter(lambda seconds: [s <= max_seconds for s in seconds], input_columns="audio_length_s", batched=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    if n is not None and n < len(ds):  # spread evenly over the (duration-sorted) split: keeps the length profile
        ds = ds.select(np.linspace(0, len(ds) - 1, n).round().astype(int).tolist())
    utterances = [Utterance(str(i), decode_flac(audio_bytes(row["audio"])), row[text_column])
                  for i, row in enumerate(ds)]
    return [u for u in utterances if u.seconds <= max_seconds], language


@lru_cache
def _normalizer(language: str):
    """Whisper's English normaliser (as the Open ASR Leaderboard) for English, its basic one otherwise."""
    if language != "en":
        return BasicTextNormalizer()
    from huggingface_hub import hf_hub_download

    return EnglishTextNormalizer(json.load(open(hf_hub_download("openai/whisper-large-v3", "normalizer.json"))))


def wer(refs: list[str], hyps: list[str], language: str = "en") -> float:
    """WER (CER for ja/zh/ko) in %, after the Open ASR Leaderboard normaliser (English) or Whisper's basic one."""
    norm = _normalizer(language)
    pairs = [(norm(r), norm(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r.strip()]
    if language in CHAR_LANGS:
        pairs = [(r.replace(" ", ""), h.replace(" ", "")) for r, h in pairs]
        return 100 * jiwer.cer([r for r, _ in pairs], [h for _, h in pairs])
    return 100 * jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])


def evaluate(student, processor, stats, norm_mode, eval_sets, lookaheads, device, dtype, prior_frames=100,
             autocast_dtype: torch.dtype | Literal["auto"] | None = "auto"):
    """Word error rate of the student's AED decoder and CTC head, per lookahead.

    `eval_sets` maps a name to (utterances, language). With a single set the result is {"R<la>": {...}};
    with several it is {name: {"R<la>": {...}}}.
    """
    student.eval()
    results = {}
    if autocast_dtype == "auto":
        autocast_dtype = (torch.bfloat16 if device == "cuda" and
                          torch.cuda.is_bf16_supported(including_emulation=False) else None)
    with torch.autocast("cuda", dtype=autocast_dtype or torch.bfloat16,
                        enabled=device == "cuda" and autocast_dtype is not None):
        for name, (utterances, language) in eval_sets.items():
            references = [u.text for u in utterances]
            per_lookahead = {}
            for r in lookaheads:
                aed = aed_transcribe(student.cohere, processor, student.encoder, utterances, norm_mode, stats, r,
                                     device, dtype, language=language, prior_frames=prior_frames)
                ctc = ctc_greedy(student.ctc_head, processor, student.encoder, utterances, norm_mode, stats, r,
                                 device, dtype, prior_frames=prior_frames)
                per_lookahead[f"R{r}"] = {"aed_wer": round(wer(references, aed, language), 2),
                                          "ctc_wer": round(wer(references, ctc, language), 2)}
            results[name] = per_lookahead
    student.train()
    return next(iter(results.values())) if len(results) == 1 else results


# ============================================================================================================
# 4. The trainer
#
# `DistillTrainer` is `transformers.Trainer` plus: the distillation loss (`compute_loss`), per-loss logging, WER
# evaluation (`evaluate`) and checkpoints in the layout `model.CohereStreaming` loads (`_save`).
# ============================================================================================================


class DistillTrainer(Trainer):
    """`Trainer` for a `StreamingStudent`. Extra keyword arguments: `distill_args`, `stats` (mel statistics)."""

    def __init__(self, *args, distill_args: DistillArguments, stats: dict, **kwargs):
        # Trainer insists on an eval_dataset whenever evaluation is on; ours (`evaluate` below) never reads it
        kwargs.setdefault("eval_dataset", kwargs.get("train_dataset"))
        # --ctc_weight 0 leaves the CTC head without a gradient (the other zero weights do not leave parameters unused,
        # see `needs_find_unused_parameters`). DDP raises an error for that unless `find_unused_parameters=True`; it is
        # read when `Trainer` is created,
        # so set it on the arguments before calling `super().__init__`. It does nothing on a single GPU.
        train_args = kwargs.get("args", args[1] if len(args) > 1 else None)
        if train_args is not None and needs_find_unused_parameters(distill_args) and not train_args.ddp_find_unused_parameters:
            train_args.ddp_find_unused_parameters = True
        super().__init__(*args, **kwargs)
        self.distill_args, self.stats = distill_args, stats
        self._eval_sets = None  # loaded on first use, on rank 0 only
        self._last_eval_step = None  # `global_step` of the latest `evaluate`, so `main` can skip a duplicate final eval
        # aed / ctc / kd loss sums since the last log. They stay (detached) tensors on the GPU: turning them into
        # Python floats on every step would force a GPU -> CPU sync, so that happens only in `log`.
        self._part_sums: dict[str, torch.Tensor] = {}
        self._part_count = 0

    # ---- loss ----------------------------------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Run the student (which also runs the frozen teacher encoder) and return the combined loss."""
        losses = model(**inputs)
        for name in ("aed", "ctc", "kd"):
            if name in losses:
                self._part_sums[name] = self._part_sums.get(name, 0.0) + losses[name].detach()
        self._part_count += 1
        return (losses["loss"], losses) if return_outputs else losses["loss"]

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Also log the mean of each loss term (aed_loss, ctc_loss, kd_loss) since the previous log."""
        if "loss" in logs and self._part_count:
            # `.item()` (the only GPU -> CPU sync) happens here, once per logging interval
            logs = {**logs, **{f"{k}_loss": round(float(v) / self._part_count, 4) for k, v in self._part_sums.items()}}
            self._part_sums, self._part_count = {}, 0
        super().log(logs, start_time)

    # ---- evaluation: WER of the AED decoder and the CTC head, per lookahead, on rank 0 -------------------------
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval") -> dict[str, float]:
        args, metrics = self.distill_args, {}
        self._last_eval_step = self.state.global_step
        if self.is_world_process_zero():
            if self._eval_sets is None:
                self._eval_sets = {spec: load_eval_set(spec, n=args.eval_n) for spec in args.eval_sets.split(",")}
            student = self.accelerator.unwrap_model(self.model)
            result = evaluate(student, self.processing_class, self.stats, args.norm, self._eval_sets,
                              list(parse_int_list(args.eval_lookaheads)), self.args.device.type, torch.float32,
                              prior_frames=args.eval_prior,
                              autocast_dtype=torch.bfloat16 if self.args.bf16 else
                              torch.float16 if self.args.fp16 else None)
            if len(self._eval_sets) == 1:
                result = {next(iter(self._eval_sets)): result}
            for set_name, per_lookahead in result.items():
                for lookahead, scores in per_lookahead.items():
                    for kind, value in scores.items():
                        metrics[f"{metric_key_prefix}_{set_name.replace(':', '_')}_{lookahead}_{kind}"] = value
        metrics = broadcast_object_list([metrics], from_process=0)[0]
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def evaluate_at_end(self) -> bool:
        """Evaluate once more after training unless evaluation is off or the last step was already an eval step.

        With `--eval_strategy steps` this transformers version already evaluates at the last step, but other
        versions, `--eval_strategy epoch` or `--eval_delay` can end a run without scoring the final model.
        `evaluate` records its step, so this never runs a duplicate. Returns whether an evaluation ran.
        """
        if self.args.eval_strategy == "no" or self._last_eval_step == self.state.global_step:
            return False
        self.evaluate()
        return True

    def save_and_push_final_model(self) -> None:
        """Write the final model to `--output_dir` and, with `--push_to_hub`, upload it exactly once.

        `Trainer.save_model()` itself pushes when `--push_to_hub` is set, and `Trainer.push_to_hub()` saves too (it
        also writes the model card), so calling both uploads twice. `push_to_hub()` alone does everything.
        """
        if self.args.push_to_hub:
            self.push_to_hub()
        else:
            self.save_model()

    # ---- optimizer: honours --optim / --adam_* / --weight_decay, plus a separate head learning rate -----------------
    def create_optimizer(self, model=None):
        """Build the optimizer once (unless one was passed in `optimizers=`) with `build_optimizer`."""
        if self.optimizer is None:
            self.optimizer = build_optimizer(self.accelerator.unwrap_model(model or self.model), self.distill_args,
                                             self.args)
        return self.optimizer

    # ---- checkpoints: the layout `model.CohereStreaming` reads ---------------------------------------------------
    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        """Write student weights (without the public teacher encoder), config, mel stats and log.

        Precision: intermediate `checkpoint-N/` folders are float32 so that resuming restores the exact weights
        (rounding to bf16 would silently change the model at every resume). The final model, saved to
        `--output_dir` itself by `trainer.save_model()`, is bf16: half the size, and `CohereStreaming` converts
        it back to float32 when loading.
        """
        ckpt = Path(output_dir or self.args.output_dir)
        ckpt.mkdir(parents=True, exist_ok=True)
        is_final = ckpt.resolve() == Path(self.args.output_dir).resolve()
        dtype = torch.bfloat16 if is_final else torch.float32
        student = self.accelerator.unwrap_model(self.model)
        state = {k: v.detach().to(dtype).contiguous().cpu() for k, v in student.state_dict().items()
                 if not k.startswith("cohere.model.encoder.")}  # the teacher is the public base model
        save_file(state, ckpt / "student.safetensors")
        config = {"encoder": student.encoder.config.to_dict(), "args": asdict(self.distill_args)}
        (ckpt / "config.json").write_text(json.dumps(config, indent=1))
        save_stats(self.stats, ckpt / "mel_stats.json")
        (ckpt / "log.json").write_text(json.dumps(self.state.log_history, indent=1))

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
        """Resume: read the weights back from `student.safetensors` (optimizer and step come from Trainer)."""
        load_student_checkpoint(self.accelerator.unwrap_model(model or self.model), resume_from_checkpoint)

    def _load_best_model(self) -> None:
        """Trainer's default loader only recognises model.safetensors/pytorch_model.bin."""
        self._load_from_checkpoint(self.state.best_model_checkpoint)

    def _save_rng_state(self, output_dir: str) -> None:
        """Save the rank's private lookahead generator as well as Trainer's global generators."""
        super()._save_rng_state(output_dir)
        student = self.accelerator.unwrap_model(self.model)
        (Path(output_dir) / f"lookahead_rng_{self.args.process_index}.json").write_text(
            json.dumps(student.rng.getstate()))

    def _load_rng_state(self, checkpoint: str | None) -> None:
        super()._load_rng_state(checkpoint)
        if checkpoint is None:
            return
        path = Path(checkpoint) / f"lookahead_rng_{self.args.process_index}.json"
        if path.exists():  # earlier checkpoints did not include this private generator
            version, internal, gauss = json.loads(path.read_text())
            self.accelerator.unwrap_model(self.model).rng.setstate((version, tuple(internal), gauss))

    def create_model_card(self, *args, **kwargs) -> None:
        try:  # the default card reads `model.config`, which a plain nn.Module does not have
            super().create_model_card(*args, **kwargs)
        except AttributeError:
            pass


def build_optimizer(student: StreamingStudent, distill: DistillArguments, args: TrainingArguments) -> torch.optim.Optimizer:
    """The optimizer chosen by the `TrainingArguments`, with three parameter groups.

    * the optimizer class and its settings come from `args` (`--optim`, `--adam_beta1/2`, `--adam_epsilon`),
      exactly as `Trainer` would pick them;
    * the CTC head (new, randomly initialised) learns at `--head_lr`, the pretrained weights at `--learning_rate`;
    * biases and normalisation weights get no weight decay, everything else gets `--weight_decay`.

    `DistillTrainer.create_optimizer` calls this, so you normally never need to; the notebook passes it explicitly.
    Optimizers that need extra parameter handling (GaLore, LOMO, ...) are not supported.
    """
    optimizer_cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(args, student)
    if {"params", "model", "optimizer_dict"} & set(kwargs):
        raise ValueError(f"--optim {args.optim} is not supported by this recipe; use an AdamW/SGD style optimizer")
    groups = {"head": [], "decay": [], "no_decay": []}
    for name, p in student.named_parameters():
        if not p.requires_grad:  # the frozen teacher (and the decoder with --freeze_decoder)
            continue
        if name.startswith("ctc_head."):
            groups["head"].append(p)
        elif p.ndim <= 1 or "bias" in name:  # biases and norm scales/shifts
            groups["no_decay"].append(p)
        else:
            groups["decay"].append(p)
    param_groups = [
        {"params": groups["decay"], "lr": args.learning_rate, "weight_decay": args.weight_decay},
        {"params": groups["no_decay"], "lr": args.learning_rate, "weight_decay": 0.0},
        {"params": groups["head"], "lr": distill.head_lr, "weight_decay": 0.0},
    ]
    return optimizer_cls([g for g in param_groups if g["params"]], **kwargs)


# ============================================================================================================
# 5. Main
# ============================================================================================================


def main():
    distill, args = HfArgumentParser((DistillArguments, DistillTrainingArguments)).parse_args_into_dataclasses()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.process_index == 0:
        (out / "args.json").write_text(json.dumps(asdict(distill) | {"world_size": args.world_size}, indent=2))

    # Model: frozen teacher inside a trainable streaming student
    student, processor, encoder = build_student(distill, args.device, seed=args.seed, rank=args.process_index)
    if distill.init_from:
        load_student_checkpoint(student, distill.init_from)
        print(f"initialised student from {distill.init_from}", flush=True)
    if distill.grad_ckpt:
        student.encoder.gradient_checkpointing_enable()
    if args.bf16:  # frozen teacher: bf16 halves its memory (the student's forward casts its inputs to match)
        student.teacher.to(torch.bfloat16)

    # Data
    dataset = load_distill_dataset(distill.data, distill.max_seconds)
    if args.process_index == 0:
        print(f"train rows={len(dataset)} hours={sum(dataset['duration']) / 3600:.1f} world_size={args.world_size}",
              flush=True)
    stats = estimate_training_mel_stats(dataset, processor, args.seed)
    if args.process_index == 0:
        save_stats(stats, out / "mel_stats.json")
    prior_range = tuple(float(x) for x in distill.prior_range.split(",")) if distill.prior_range else None
    collator = DistillCollator(processor, stats, distill.norm, prior_range=prior_range, gain_db=distill.gain_db,
                               seed=args.seed)

    trainer = DistillTrainer(model=student, args=args, train_dataset=dataset, data_collator=collator,
                             processing_class=processor, distill_args=distill, stats=stats)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.evaluate_at_end()  # before saving: the score lands in log.json
    trainer.save_and_push_final_model()
    print("done", flush=True)


if __name__ == "__main__":
    main()
