"""Train a streaming student from Cohere's offline transcripts and encoder states.

The student combines decoder (AED), frame-level (CTC), and feature-matching (KD)
losses. Transformers Trainer handles training, with Trackio logging by default.
"""

from __future__ import annotations

import json
import random
import shutil
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
from transformers import (
    AutoProcessor,
    CohereAsrForConditionalGeneration,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer

from distill_dataset import (
    IGNORE,
    DistillCollator,
    audio_bytes,
    build_decoder_io,  # noqa: F401 - re-exported for the training notebook
    decode_flac,
    load_distill_dataset,
)
from model import (
    COHERE_MODEL_ID,
    aed_transcribe,
    build_streaming_encoder,
    ctc_greedy,
    estimate_stats,
    save_processor,
    save_stats,
)


@dataclass
class LossWeights:
    """Contribution of each loss to the total training loss."""

    aed: float = 1.0
    ctc: float = 0.3
    kd: float = 1.0


def needs_find_unused_parameters(distill: "DistillArguments") -> bool:
    """A disabled CTC head needs unused-parameter support in multi-GPU training."""
    return distill.ctc_weight == 0


def masked_kd(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
    kind: str = "cosine",
) -> torch.Tensor:
    """Compare encoder states on valid frames, trimming unequal frame counts."""
    frame_count = min(student.shape[1], teacher.shape[1], mask.shape[1])
    student_states = student[:, :frame_count].float()
    teacher_states = teacher[:, :frame_count].detach().float()
    valid_frames = mask[:, :frame_count].bool()
    if kind == "cosine":
        frame_losses = 1.0 - F.cosine_similarity(student_states, teacher_states, dim=-1)
    elif kind == "mse":
        frame_losses = ((student_states - teacher_states) ** 2).mean(-1)
    else:
        raise ValueError(kind)
    return frame_losses[valid_frames].mean()


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
        self.cohere = cohere_model
        self.encoder = student_encoder
        self.lookaheads = tuple(lookaheads)
        self.lookahead_weights = tuple(lookahead_weights) if lookahead_weights else None
        if self.lookahead_weights and len(self.lookahead_weights) != len(self.lookaheads):
            raise ValueError("lookahead_weights must match lookaheads")
        self.weights = weights
        self.kd_layers = tuple(kd_layers)
        self.kd_kind = kd_kind
        # Each GPU gets its own reproducible lookahead sequence.
        self.rng = random.Random(seed + 1000 * rank)
        vocabulary_size = cohere_model.config.vocab_size
        self.blank_id = vocabulary_size
        self.ctc_head = nn.Linear(student_encoder.config.hidden_size, vocabulary_size + 1)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        if freeze_decoder:
            for module in (self.cohere.model.decoder, self.cohere.proj_out):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    @property
    def teacher(self) -> nn.Module:
        """The frozen offline encoder inside the Cohere model."""
        return self.cohere.model.encoder

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()  # Keep the frozen teacher's statistics and dropout unchanged.
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
        """Return loss terms and the chosen lookahead for one batch."""
        if lookahead is None:
            if self.training:
                lookahead = self.rng.choices(self.lookaheads, weights=self.lookahead_weights)[0]
            else:
                lookahead = self.lookaheads[-1]
        include_hidden_states = self.weights.kd > 0 and any(layer != -1 for layer in self.kd_layers)
        student_output = self.encoder(
            student_features,
            attention_mask=attention_mask,
            num_lookahead_tokens=lookahead,
            output_hidden_states=include_hidden_states,
        )
        # Keep the decoder in the gradient graph even when its loss weight is zero.
        losses = {
            "aed": self._decoder_loss(
                student_output, decoder_input_ids, decoder_attention_mask, decoder_targets
            )
        }
        if self.weights.ctc > 0:
            losses["ctc"] = self._ctc_loss(student_output, ctc_targets)
        if self.weights.kd > 0:
            losses["kd"] = self._distillation_loss(
                student_output, teacher_features, attention_mask, include_hidden_states
            )
        losses["loss"] = (
            self.weights.aed * losses["aed"]
            + self.weights.ctc * losses.get("ctc", 0.0)
            + self.weights.kd * losses.get("kd", 0.0)
        )
        losses["lookahead"] = torch.tensor(lookahead)
        return losses

    def _decoder_loss(self, student_output, input_ids, attention_mask, targets):
        decoder_output = self.cohere.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=student_output.last_hidden_state,
            encoder_attention_mask=student_output.attention_mask,
        )
        logits = self.cohere.proj_out(decoder_output.last_hidden_state)
        return F.cross_entropy(
            logits.float().flatten(0, 1), targets.flatten(), ignore_index=IGNORE
        )

    def _ctc_loss(self, student_output, targets):
        logits = self.ctc_head(student_output.last_hidden_state).float()
        log_probs = logits.log_softmax(-1).transpose(0, 1)  # CTC expects (frames, batch, vocabulary).
        input_lengths = student_output.attention_mask.sum(-1)
        target_lengths = torch.tensor([len(tokens) for tokens in targets], device=log_probs.device)
        target_tokens = torch.tensor(
            [token for tokens in targets for token in tokens],
            dtype=torch.long,
            device=log_probs.device,
        )
        return F.ctc_loss(
            log_probs,
            target_tokens,
            input_lengths,
            target_lengths,
            blank=self.blank_id,
            zero_infinity=True,
        )

    def _distillation_loss(self, student_output, teacher_features, attention_mask, include_hidden_states):
        teacher_dtype = next(self.teacher.parameters()).dtype
        with torch.no_grad():
            teacher_output = self.teacher(
                teacher_features.to(teacher_dtype),
                attention_mask=attention_mask,
                output_hidden_states=include_hidden_states,
            )
        layer_losses = []
        for layer in self.kd_layers:
            if layer == -1:
                student_states = student_output.last_hidden_state
                teacher_states = teacher_output.last_hidden_state
            else:
                student_states = student_output.hidden_states[layer]
                teacher_states = teacher_output.hidden_states[layer]
            layer_losses.append(
                masked_kd(student_states, teacher_states, student_output.attention_mask, self.kd_kind)
            )
        return torch.stack(layer_losses).mean()


def parse_int_list(text: str) -> tuple[int, ...]:
    """'0,3,13' -> (0, 3, 13)."""
    return tuple(int(value) for value in text.split(","))


def build_student(
    args: "DistillArguments", device, seed: int = 0, rank: int = 0
) -> tuple[StreamingStudent, object, object]:
    """Return the student, audio processor, and streaming encoder."""
    set_seed(seed)
    processor = AutoProcessor.from_pretrained(COHERE_MODEL_ID)
    cohere = CohereAsrForConditionalGeneration.from_pretrained(COHERE_MODEL_ID, dtype=torch.float32).to(device)
    lookaheads = parse_int_list(args.lookaheads)
    encoder = build_streaming_encoder(
        cohere.model.encoder,
        cohere.config,
        conv_init=args.conv_init,
        left_context=args.left,
        lookahead=max(lookaheads),
    )
    lookahead_weights = None
    if args.lookahead_weights:
        lookahead_weights = tuple(float(value) for value in args.lookahead_weights.split(","))
    student = StreamingStudent(
        cohere,
        encoder,
        lookaheads=lookaheads,
        lookahead_weights=lookahead_weights,
        weights=LossWeights(aed=args.aed_weight, ctc=args.ctc_weight, kd=args.kd_weight),
        kd_layers=parse_int_list(args.kd_layers),
        freeze_decoder=args.freeze_decoder,
        seed=seed,
        rank=rank,
    ).to(device).train()
    return student, processor, encoder


def load_student_checkpoint(student: StreamingStudent, init_from: str) -> None:
    """Load student weights from a Hub repo or local directory (to continue a previous run)."""
    checkpoint_dir = Path(init_from)
    if not checkpoint_dir.exists():
        checkpoint_dir = Path(snapshot_download(init_from))
    state = {
        name: tensor.float()
        for name, tensor in load_file(str(checkpoint_dir / "student.safetensors")).items()
    }
    missing, unexpected = student.load_state_dict(state, strict=False)
    # The teacher is loaded from Cohere, so student checkpoints omit it.
    missing = [name for name in missing if not name.startswith("cohere.model.encoder.")]
    if missing or unexpected:
        raise RuntimeError(f"--init_from mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")


def estimate_training_mel_stats(dataset, processor, seed: int) -> dict:
    """Estimate mel mean and standard deviation from at most 300 training examples."""
    sample_indices = random.Random(seed).sample(range(len(dataset)), min(300, len(dataset)))
    audio = [
        torch.from_numpy(decode_flac(audio_bytes(sample)))
        for sample in dataset.select(sample_indices)["audio"]
    ]
    return estimate_stats(processor.feature_extractor, audio)


@dataclass
class DistillArguments:
    """Dataset, streaming model, and distillation settings exposed by the CLI."""

    data: str = field(metadata={"help": "labelled dataset made by distill_dataset.py: a Hub dataset id, a folder "
                                        "(glob ok) or parquet files (local glob or hf://datasets/<repo>/<glob>); "
                                        "separate several with commas"})
    max_seconds: float = field(default=30.0, metadata={"help": "skip utterances longer than this"})
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
    aed_weight: float = field(default=1.0, metadata={"help": "weight of the decoder cross-entropy loss"})
    ctc_weight: float = field(default=0.3, metadata={"help": "weight of the CTC head loss (0 disables it)"})
    kd_weight: float = field(default=1.0, metadata={"help": "weight of the encoder hidden-state matching loss"})
    kd_layers: str = field(default="-1,12,24,36", metadata={
        "help": "encoder layers whose states are matched to the teacher"})
    head_lr: float = field(default=1e-3, metadata={"help": "learning rate of the new CTC head; "
                                                           "--learning_rate controls pretrained weights"})
    prior_range: Optional[str] = field(default=None, metadata={
        "help": "train-time causal-norm warm-up prior range in frames, e.g. 10,100"})
    gain_db: float = field(default=0.0, metadata={"help": "random volume change of +/- this many dB"})
    eval_n: int = field(default=100, metadata={"help": "utterances per eval set"})
    eval_lookaheads: str = field(default="0,3,13", metadata={"help": "lookaheads to evaluate"})
    eval_sets: str = field(default="librispeech:test.other", metadata={
        "help": "comma-separated eval sets, e.g. librispeech:test.other,fleurs:fr,fleurs:de"})
    eval_prior: int = field(default=100, metadata={"help": "causal-norm warm-up prior (frames) used by evals"})


@dataclass
class DistillTrainingArguments(TrainingArguments):
    """Transformers training arguments with defaults for this recipe."""

    output_dir: str = field(default="checkpoint/train", metadata={
        "help": "final bf16 model with config, processor, and code for AutoModel loading; "
                "intermediate checkpoint-N/ folders hold float32 weights for exact resume"})
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
    ddp_timeout: int = 3600  # Other GPUs wait while rank 0 evaluates.
    hub_private_repo: Optional[bool] = True
    seed: int = 0


ESB_REPO = "hf-audio/esb-datasets-test-only-sorted"
FLEURS = {
    "en": "en_us",
    "fr": "fr_fr",
    "de": "de_de",
    "es": "es_419",
    "it": "it_it",
    "pt": "pt_br",
    "nl": "nl_nl",
    "pl": "pl_pl",
    "el": "el_gr",
    "ar": "ar_eg",
    "ja": "ja_jp",
    "zh": "cmn_hans_cn",
    "ko": "ko_kr",
    "vi": "vi_vn",
}
CHAR_LANGS = {"ja", "zh", "ko"}  # Score these languages by character because word boundaries vary.


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
    """Load evaluation audio from a spec such as 'librispeech:test.other' or 'fleurs:fr'."""
    name, split_or_language = spec.split(":")
    if name == "fleurs":
        files = f"hf://datasets/google/fleurs/parquet-data/{FLEURS[split_or_language]}/test-*.parquet"
        dataset = load_dataset("parquet", data_files=files, split="train")
        text_column = "transcription"
        language = split_or_language
    else:
        dataset = load_dataset(ESB_REPO, name, split=split_or_language)
        text_column = "text"
        language = "en"
        dataset = dataset.filter(
            lambda durations: [duration <= max_seconds for duration in durations],
            input_columns="audio_length_s",
            batched=True,
        )
    dataset = dataset.cast_column("audio", Audio(decode=False))
    if n is not None and n < len(dataset):
        # Evenly spaced samples preserve the duration profile of the sorted split.
        sample_indices = np.linspace(0, len(dataset) - 1, n).round().astype(int).tolist()
        dataset = dataset.select(sample_indices)
    utterances = []
    for index, row in enumerate(dataset):
        audio = decode_flac(audio_bytes(row["audio"]))
        utterance = Utterance(str(index), audio, row[text_column])
        if utterance.seconds <= max_seconds:
            utterances.append(utterance)
    return utterances, language


@lru_cache
def _normalizer(language: str):
    """Use the same Whisper text normalization as the Open ASR Leaderboard."""
    if language != "en":
        return BasicTextNormalizer()
    from huggingface_hub import hf_hub_download

    normalizer_path = hf_hub_download("openai/whisper-large-v3", "normalizer.json")
    with open(normalizer_path) as normalizer_file:
        return EnglishTextNormalizer(json.load(normalizer_file))


def wer(refs: list[str], hyps: list[str], language: str = "en") -> float:
    """Return normalized word error rate in percent (character error rate for ja/zh/ko)."""
    normalize = _normalizer(language)
    pairs = [(normalize(reference), normalize(hypothesis)) for reference, hypothesis in zip(refs, hyps)]
    pairs = [(reference, hypothesis) for reference, hypothesis in pairs if reference.strip()]
    if language in CHAR_LANGS:
        pairs = [(reference.replace(" ", ""), hypothesis.replace(" ", "")) for reference, hypothesis in pairs]
    references = [reference for reference, _ in pairs]
    hypotheses = [hypothesis for _, hypothesis in pairs]
    error_rate = jiwer.cer if language in CHAR_LANGS else jiwer.wer
    return 100 * error_rate(references, hypotheses)


def evaluate(
    student,
    processor,
    stats,
    norm_mode,
    eval_sets,
    lookaheads,
    device,
    dtype,
    prior_frames=100,
    autocast_dtype: torch.dtype | Literal["auto"] | None = "auto",
):
    """Score the decoder and CTC head for each lookahead and evaluation dataset.

    A single dataset returns {"R<lookahead>": scores}; multiple datasets add an
    outer mapping from dataset name to those scores.
    """
    student.eval()
    results = {}
    if autocast_dtype == "auto":
        autocast_dtype = None
        if device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False):
            autocast_dtype = torch.bfloat16
    with torch.autocast(
        "cuda",
        dtype=autocast_dtype or torch.bfloat16,
        enabled=device == "cuda" and autocast_dtype is not None,
    ):
        for name, (utterances, language) in eval_sets.items():
            references = [utterance.text for utterance in utterances]
            per_lookahead = {}
            for lookahead in lookaheads:
                decoder_transcripts = aed_transcribe(
                    student.cohere, processor, student.encoder, utterances, norm_mode, stats,
                    lookahead, device, dtype, language=language, prior_frames=prior_frames,
                )
                ctc_transcripts = ctc_greedy(
                    student.ctc_head, processor, student.encoder, utterances, norm_mode, stats,
                    lookahead, device, dtype, prior_frames=prior_frames,
                )
                per_lookahead[f"R{lookahead}"] = {
                    "aed_wer": round(wer(references, decoder_transcripts, language), 2),
                    "ctc_wer": round(wer(references, ctc_transcripts, language), 2),
                }
            results[name] = per_lookahead
    student.train()
    if len(results) == 1:
        return next(iter(results.values()))
    return results


class DistillTrainer(Trainer):
    """Train with distillation losses, transcription scores, and streaming checkpoints."""

    def __init__(self, *args, distill_args: DistillArguments, stats: dict, **kwargs):
        # Trainer requires a dataset here; our evaluation loads its own audio.
        kwargs.setdefault("eval_dataset", kwargs.get("train_dataset"))
        train_args = kwargs.get("args")
        if train_args is None and len(args) > 1:
            train_args = args[1]
        if train_args is not None and needs_find_unused_parameters(distill_args):
            train_args.ddp_find_unused_parameters = True
        super().__init__(*args, **kwargs)
        self.distill_args = distill_args
        self.stats = stats
        self._eval_sets = None
        self._last_eval_step = None
        self._part_sums: dict[str, torch.Tensor] = {}
        self._part_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Return the combined loss and collect individual losses for logging."""
        losses = model(**inputs)
        for name in ("aed", "ctc", "kd"):
            if name in losses:
                self._part_sums[name] = self._part_sums.get(name, 0.0) + losses[name].detach()
        self._part_count += 1
        if return_outputs:
            return losses["loss"], losses
        return losses["loss"]

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Include mean AED, CTC, and KD losses since the previous training log."""
        if "loss" in logs and self._part_count:
            mean_losses = {
                f"{name}_loss": round(float(total) / self._part_count, 4)
                for name, total in self._part_sums.items()
            }
            logs = {**logs, **mean_losses}
            self._part_sums = {}
            self._part_count = 0
        super().log(logs, start_time)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval") -> dict[str, float]:
        """Evaluate on rank 0, then share the metrics with every training process."""
        self._last_eval_step = self.state.global_step
        metrics = {}
        if self.is_world_process_zero():
            metrics = self._evaluation_metrics(metric_key_prefix)
        metrics = broadcast_object_list([metrics], from_process=0)[0]
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def _evaluation_metrics(self, metric_key_prefix: str) -> dict[str, float]:
        distill_args = self.distill_args
        if self._eval_sets is None:
            self._eval_sets = {
                spec: load_eval_set(spec, n=distill_args.eval_n)
                for spec in distill_args.eval_sets.split(",")
            }
        autocast_dtype = None
        if self.args.bf16:
            autocast_dtype = torch.bfloat16
        elif self.args.fp16:
            autocast_dtype = torch.float16
        student = self.accelerator.unwrap_model(self.model)
        results = evaluate(
            student,
            self.processing_class,
            self.stats,
            distill_args.norm,
            self._eval_sets,
            list(parse_int_list(distill_args.eval_lookaheads)),
            self.args.device.type,
            torch.float32,
            prior_frames=distill_args.eval_prior,
            autocast_dtype=autocast_dtype,
        )
        if len(self._eval_sets) == 1:
            results = {next(iter(self._eval_sets)): results}
        metrics = {}
        for set_name, per_lookahead in results.items():
            set_name = set_name.replace(":", "_")
            for lookahead, scores in per_lookahead.items():
                for score_name, value in scores.items():
                    metrics[f"{metric_key_prefix}_{set_name}_{lookahead}_{score_name}"] = value
        return metrics

    def evaluate_at_end(self) -> bool:
        """Score the final step once, unless evaluation is disabled."""
        if self.args.eval_strategy == "no" or self._last_eval_step == self.state.global_step:
            return False
        self.evaluate()
        return True

    def save_and_push_final_model(self) -> None:
        """Save the final model locally, or save and upload it to the Hub."""
        if self.args.push_to_hub:
            self.push_to_hub()  # This also saves the model locally.
        else:
            self.save_model()

    def create_optimizer(self, model=None):
        if self.optimizer is None:
            student = self.accelerator.unwrap_model(model or self.model)
            self.optimizer = build_optimizer(student, self.distill_args, self.args)
        return self.optimizer

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        """Save weights, model settings, normalization statistics, and training logs."""
        checkpoint_dir = Path(output_dir or self.args.output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        is_final = checkpoint_dir.resolve() == Path(self.args.output_dir).resolve()
        # Resume checkpoints retain full precision; the final release uses smaller weights.
        dtype = torch.bfloat16 if is_final else torch.float32
        student = self.accelerator.unwrap_model(self.model)
        state = {}
        for name, tensor in student.state_dict().items():
            if not name.startswith("cohere.model.encoder."):
                state[name] = tensor.detach().to(dtype).contiguous().cpu()
        save_file(state, checkpoint_dir / "student.safetensors")
        config = {
            "model_type": "cohere_streaming",
            "architectures": ["CohereStreamingModel"],
            "auto_map": {
                "AutoConfig": "model.CohereStreamingConfig",
                "AutoModel": "model.CohereStreamingModel",
            },
            "encoder": student.encoder.config.to_dict(),
            "cohere_config": student.cohere.config.to_dict(),
            "args": asdict(self.distill_args),
        }
        (checkpoint_dir / "config.json").write_text(json.dumps(config, indent=1))
        save_stats(self.stats, checkpoint_dir / "mel_stats.json")
        (checkpoint_dir / "log.json").write_text(json.dumps(self.state.log_history, indent=1))
        save_processor(self.processing_class, checkpoint_dir)
        student.cohere.generation_config.save_pretrained(checkpoint_dir)
        shutil.copyfile(Path(__file__).with_name("model.py"), checkpoint_dir / "model.py")

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
        """Restore student weights; Trainer restores the optimizer and training step."""
        student = self.accelerator.unwrap_model(model or self.model)
        load_student_checkpoint(student, resume_from_checkpoint)

    def _load_best_model(self) -> None:
        """Trainer's default loader only recognises model.safetensors/pytorch_model.bin."""
        self._load_from_checkpoint(self.state.best_model_checkpoint)

    def _save_rng_state(self, output_dir: str) -> None:
        """Save the rank's private lookahead generator as well as Trainer's global generators."""
        super()._save_rng_state(output_dir)
        student = self.accelerator.unwrap_model(self.model)
        (Path(output_dir) / f"lookahead_rng_{self.args.process_index}.json").write_text(
            json.dumps(student.rng.getstate())
        )

    def _load_rng_state(self, checkpoint: str | None) -> None:
        super()._load_rng_state(checkpoint)
        if checkpoint is None:
            return
        path = Path(checkpoint) / f"lookahead_rng_{self.args.process_index}.json"
        if path.exists():  # Older checkpoints may omit the lookahead generator.
            version, internal, gaussian_state = json.loads(path.read_text())
            student = self.accelerator.unwrap_model(self.model)
            student.rng.setstate((version, tuple(internal), gaussian_state))

    def create_model_card(self, *args, **kwargs) -> None:
        try:  # the default card reads `model.config`, which a plain nn.Module does not have
            super().create_model_card(*args, **kwargs)
        except AttributeError:
            pass


def build_optimizer(
    student: StreamingStudent, distill: DistillArguments, args: TrainingArguments
) -> torch.optim.Optimizer:
    """Use Trainer's optimizer with a separate CTC learning rate and no decay for biases or norms."""
    optimizer_class, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, student)
    if {"params", "model", "optimizer_dict"} & set(optimizer_kwargs):
        raise ValueError(f"--optim {args.optim} is not supported by this recipe; use an AdamW/SGD style optimizer")
    head_parameters = []
    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("ctc_head."):
            head_parameters.append(parameter)
        elif parameter.ndim <= 1 or "bias" in name:
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)
    parameter_groups = [
        {"params": decay_parameters, "lr": args.learning_rate, "weight_decay": args.weight_decay},
        {"params": no_decay_parameters, "lr": args.learning_rate, "weight_decay": 0.0},
        {"params": head_parameters, "lr": distill.head_lr, "weight_decay": 0.0},
    ]
    nonempty_groups = [group for group in parameter_groups if group["params"]]
    return optimizer_class(nonempty_groups, **optimizer_kwargs)


def main():
    parser = HfArgumentParser((DistillArguments, DistillTrainingArguments))
    distill_args, training_args = parser.parse_args_into_dataclasses()
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if training_args.process_index == 0:
        settings = asdict(distill_args) | {"world_size": training_args.world_size}
        (output_dir / "args.json").write_text(json.dumps(settings, indent=2))

    student, processor, _ = build_student(
        distill_args, training_args.device, seed=training_args.seed, rank=training_args.process_index
    )
    if distill_args.init_from:
        load_student_checkpoint(student, distill_args.init_from)
        print(f"initialised student from {distill_args.init_from}", flush=True)
    if distill_args.grad_ckpt:
        student.encoder.gradient_checkpointing_enable()
    if training_args.bf16:
        student.teacher.to(torch.bfloat16)

    dataset = load_distill_dataset(distill_args.data, distill_args.max_seconds)
    if training_args.process_index == 0:
        hours = sum(dataset["duration"]) / 3600
        print(
            f"train rows={len(dataset)} hours={hours:.1f} world_size={training_args.world_size}",
            flush=True,
        )
    stats = estimate_training_mel_stats(dataset, processor, training_args.seed)
    if training_args.process_index == 0:
        save_stats(stats, output_dir / "mel_stats.json")
    prior_range = None
    if distill_args.prior_range:
        prior_range = tuple(float(value) for value in distill_args.prior_range.split(","))
    collator = DistillCollator(
        processor,
        stats,
        distill_args.norm,
        prior_range=prior_range,
        gain_db=distill_args.gain_db,
        seed=training_args.seed,
    )

    trainer = DistillTrainer(
        model=student,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
        distill_args=distill_args,
        stats=stats,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.evaluate_at_end()  # Include the final score in log.json before saving.
    trainer.save_and_push_final_model()
    print("done", flush=True)


if __name__ == "__main__":
    main()
