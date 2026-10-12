"""Streaming Cohere transcription. Call reset(), feed 16 kHz audio with step(), then finish()."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.dataclasses import strict
from safetensors.torch import load_file
from torch import nn
from transformers import (
    AutoProcessor,
    CohereAsrForConditionalGeneration,
    NemotronAsrStreamingEncoder,
    NemotronAsrStreamingEncoderConfig,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.cohere_asr.feature_extraction_cohere_asr import LOG_ZERO_GUARD_VALUE
from transformers.models.nemotron_asr_streaming import modeling_nemotron_asr_streaming as nas


COHERE_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
DEFAULT_CHECKPOINT = "Steveeeeeeen/cohere-streaming"
FULL_CONTEXT = 10**6
EPSILON = 1e-5
SR = 16000


def log_mel(fe, waveforms: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log-mel features (batch, time, mel bins) and a valid-frame mask."""
    lengths = torch.tensor([waveform.shape[0] for waveform in waveforms])
    audio = torch.zeros(len(waveforms), int(lengths.max()))
    for index, waveform in enumerate(waveforms):
        audio[index, : waveform.shape[0]] = waveform.float()

    audio = fe._apply_dither(audio, lengths)
    if fe.preemphasis is not None:
        sample_mask = torch.arange(audio.shape[1])[None, :] < lengths[:, None]
        emphasized = audio[:, 1:] - fe.preemphasis * audio[:, :-1]
        audio = torch.cat([audio[:, :1], emphasized], dim=1).masked_fill(~sample_mask, 0.0)

    features = fe._torch_extract_fbank_features(audio, "cpu")
    frame_counts = torch.floor_divide(lengths + fe.n_fft // 2 * 2 - fe.n_fft, fe.hop_length)
    frame_mask = torch.arange(features.shape[1])[None, :] < frame_counts[:, None]
    return features, frame_mask


def normalize(
    feats: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "utterance",
    stats: dict | None = None,
    prior_frames: int = 100,
) -> torch.Tensor:
    """Normalize mel bins using utterance, corpus, or running statistics."""
    frame_mask = mask.unsqueeze(-1).to(feats.dtype)
    if mode == "utterance":
        frame_counts = mask.sum(1, keepdim=True).to(feats.dtype).unsqueeze(-1)
        mean = (feats * frame_mask).sum(1, keepdim=True) / frame_counts
        variance = (((feats * frame_mask) - mean) ** 2 * frame_mask).sum(1, keepdim=True)
        variance = variance / (frame_counts - 1)
        normalized = (feats - mean) / (variance.sqrt() + EPSILON)
    elif mode == "global":
        mean, std = _stats(stats, feats)
        normalized = (feats - mean) / (std + EPSILON)
    elif mode == "causal":
        global_mean, global_std = _stats(stats, feats)
        prior = float(prior_frames)
        running_sum = torch.cumsum(feats * frame_mask, dim=1) + prior * global_mean
        running_squared_sum = torch.cumsum((feats * frame_mask) ** 2, dim=1)
        running_squared_sum = running_squared_sum + prior * (global_std**2 + global_mean**2)
        frame_counts = torch.cumsum(frame_mask, dim=1) + prior
        mean = running_sum / frame_counts
        variance = (running_squared_sum / frame_counts - mean**2).clamp_min(1e-8)
        normalized = (feats - mean) / (variance.sqrt() + EPSILON)
    else:
        raise ValueError(f"unknown normalisation mode {mode!r}")
    return normalized * frame_mask


def _stats(stats: dict | None, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if stats is None:
        raise ValueError("global/causal normalisation needs corpus statistics (see `estimate_stats`).")
    mean = torch.tensor(stats["mean"], dtype=like.dtype).view(1, 1, -1)
    std = torch.tensor(stats["std"], dtype=like.dtype).view(1, 1, -1)
    return mean, std


def estimate_stats(fe, waveforms: list[torch.Tensor]) -> dict:
    """Estimate corpus mean and standard deviation for each mel bin."""
    feature_sum = 0
    squared_feature_sum = 0
    frame_count = 0
    for waveform in waveforms:
        features, frame_mask = log_mel(fe, [waveform])
        valid_features = features[frame_mask]
        feature_sum = feature_sum + valid_features.sum(0)
        squared_feature_sum = squared_feature_sum + (valid_features.double() ** 2).sum(0)
        frame_count += valid_features.shape[0]
    mean = feature_sum.double() / frame_count
    std = (squared_feature_sum / frame_count - mean**2).sqrt()
    return {"mean": mean.tolist(), "std": std.tolist(), "frames": int(frame_count)}


def save_stats(stats: dict, path: str | Path):
    Path(path).write_text(json.dumps(stats))


def load_stats(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


class StreamingFeaturizer:
    """Convert incoming audio to normalized mel frames without revisiting old samples."""

    def __init__(self, fe, stats: dict, prior_frames: int = 100, seed: int = 0, norm_mode: str = "causal"):
        if norm_mode not in ("causal", "global"):
            raise ValueError(f"streaming features require causal or global normalisation, got {norm_mode!r}")
        self.fe = fe
        self.stats = stats
        self.prior_frames = float(prior_frames)
        self.norm_mode = norm_mode
        self.hop = fe.hop_length
        self.n_fft = fe.n_fft
        self.win_length = fe.win_length
        self.window = torch.hann_window(self.win_length, periodic=False)
        self.global_mean = torch.tensor(stats["mean"], dtype=torch.float64)
        self.global_std = torch.tensor(stats["std"], dtype=torch.float64)
        self.seed = seed
        self.reset()

    def reset(self):
        self.audio_buffer = torch.zeros(self.n_fft // 2)  # Initial STFT center padding.
        self.n_samples = 0
        self.frames_out = 0
        self.last_sample = None
        self.noise_generator = torch.Generator().manual_seed(self.seed)
        self.running_sum = self.prior_frames * self.global_mean
        self.running_squared_sum = self.prior_frames * (self.global_std**2 + self.global_mean**2)
        self.frame_count = self.prior_frames

    def _append(self, audio: torch.Tensor):
        samples = audio.float()
        if self.fe.dither > 0:
            noise = torch.randn(samples.shape[0], generator=self.noise_generator)
            samples = samples + self.fe.dither * noise
        if self.fe.preemphasis is not None:
            if self.last_sample is None:
                previous_sample = torch.zeros(1)
            else:
                previous_sample = self.last_sample.view(1)
            previous_samples = torch.cat([previous_sample, samples[:-1]])
            emphasized = samples - self.fe.preemphasis * previous_samples
        else:
            emphasized = samples
        if samples.shape[0]:
            self.last_sample = samples[-1]
        self.audio_buffer = torch.cat([self.audio_buffer, emphasized])
        self.n_samples += samples.shape[0]

    def _frames(self, upto: int) -> torch.Tensor:
        frame_count = upto - self.frames_out
        if frame_count <= 0:
            return torch.zeros(0, self.fe.feature_size)
        window_samples = (frame_count - 1) * self.hop + self.n_fft
        signal = self.audio_buffer[:window_samples]
        stft = torch.stft(
            signal,
            self.n_fft,
            hop_length=self.hop,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power = torch.view_as_real(stft).pow(2).sum(-1)
        mel = torch.log(self.fe.mel_filters.to(power.dtype) @ power + LOG_ZERO_GUARD_VALUE).T
        normalized = self._normalize_frames(mel)
        self.frames_out = upto
        # Clone the remainder so it releases the full incoming chunk's storage.
        self.audio_buffer = self.audio_buffer[frame_count * self.hop:].clone()
        return normalized

    def _normalize_frames(self, mel: torch.Tensor) -> torch.Tensor:
        if self.norm_mode == "global":
            return (mel - self.global_mean.float()) / (self.global_std.float() + EPSILON)

        # Float64 keeps running sums accurate across long recordings.
        features = mel.double()
        running_sum = torch.cumsum(features, 0) + self.running_sum
        running_squared_sum = torch.cumsum(features**2, 0) + self.running_squared_sum
        frame_counts = torch.arange(1, features.shape[0] + 1, dtype=torch.float64).unsqueeze(1)
        frame_counts = frame_counts + self.frame_count
        mean = running_sum / frame_counts
        variance = (running_squared_sum / frame_counts - mean**2).clamp_min(1e-8)
        normalized = ((features - mean) / (variance.sqrt() + EPSILON)).float()
        self.running_sum = running_sum[-1]
        self.running_squared_sum = running_squared_sum[-1]
        self.frame_count = float(frame_counts[-1, 0])
        return normalized

    def push(self, audio) -> torch.Tensor:
        """Return newly completed mel frames for a chunk of 16 kHz audio."""
        self._append(torch.as_tensor(audio))
        complete_frames = self.frames_out + max(0, (self.audio_buffer.shape[0] - self.n_fft) // self.hop + 1)
        valid_frames = self.n_samples // self.hop
        return self._frames(min(complete_frames, valid_frames))

    def flush(self) -> torch.Tensor:
        """Apply end padding and return the remaining mel frames."""
        self.audio_buffer = torch.cat([self.audio_buffer, torch.zeros(self.n_fft // 2)])
        return self._frames(self.n_samples // self.hop)


@strict
class CohereStreamingEncoderConfig(NemotronAsrStreamingEncoderConfig):
    """Cohere encoder sizes and streaming attention settings."""

    model_type = "cohere_streaming_encoder"

    causal: bool = True

    @property
    def subsampling_out_hidden_size(self) -> int:
        """Width of the flattened subsampling output (channels * remaining frequency bins)."""
        padding = self.subsampling_conv_kernel_size - 1
        frequency_bins = self.num_mel_bins
        subsampling_stages = self.subsampling_factor.bit_length() - 1
        for _ in range(subsampling_stages):
            frequency_bins = (
                (frequency_bins + padding - self.subsampling_conv_kernel_size) // self.subsampling_conv_stride + 1
            )
        return self.subsampling_conv_channels * frequency_bins


class ChannelAffine(nn.Module):
    """Per-channel scale and bias, used in place of inference-time BatchNorm."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias


class CohereCausalConv2D(nas.NemotronAsrStreamingEncoderCausalConv2D):
    """Subsampling with causal time padding and symmetric frequency padding."""

    @property
    def freq_pad(self) -> tuple[int, int]:
        padding = (self.kernel_size[1] - 1) // 2
        return (padding, padding)


class CohereSymmetricConv2D(CohereCausalConv2D):
    """Subsampling with symmetric padding for offline inference."""

    @property
    def time_pad(self) -> tuple[int, int]:
        padding = (self.kernel_size[0] - 1) // 2
        return (padding, padding)


class CohereSymmetricConv1d(nas.NemotronAsrStreamingEncoderCausalConv1d):
    """Offline convolution that clears padded frames before symmetric padding."""

    frame_mask: torch.Tensor | None = None

    def forward(self, x, padding_cache=None):
        if padding_cache is not None:
            raise ValueError("Symmetric (causal=False) convolutions cannot stream.")
        if self.frame_mask is not None:
            x = x.masked_fill(~self.frame_mask, 0.0)
        padding = (self.kernel_size[0] - 1) // 2
        return nn.Conv1d.forward(self, nn.functional.pad(x, (padding, padding)))


class CohereStreamingEncoder(NemotronAsrStreamingEncoder):
    """Cache-aware encoder with Cohere-compatible padding and normalization."""

    config: CohereStreamingEncoderConfig

    def __init__(self, config: CohereStreamingEncoderConfig):
        super().__init__(config)
        # Change padding without replacing the existing convolution parameters.
        subsampling_conv_class = CohereCausalConv2D if config.causal else CohereSymmetricConv2D
        self.subsampling.conv_in.__class__ = subsampling_conv_class
        for layer in self.subsampling.layers:
            layer.depthwise_conv.__class__ = subsampling_conv_class
        for block in self.layers:
            block.conv.norm = ChannelAffine(config.hidden_size)
            if not config.causal:
                block.conv.depthwise_conv.__class__ = CohereSymmetricConv1d

    def forward(self, input_features, attention_mask=None, **kwargs):
        if not self.config.causal:
            frame_mask = None
            if attention_mask is not None:
                input_length = torch.tensor([input_features.shape[1]])
                output_frames = int(self._get_subsampling_output_length(input_length)[0])
                frame_mask = self._get_output_attention_mask(attention_mask, target_length=output_frames).unsqueeze(1)
            for block in self.layers:
                block.conv.depthwise_conv.frame_mask = frame_mask
        return super().forward(input_features, attention_mask=attention_mask, **kwargs)

    def _get_subsampling_output_length(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Number of encoder frames produced from `input_lengths` mel frames (applies each strided conv in turn)."""
        config = self.config
        padding = (config.subsampling_conv_kernel_size - 1) + (config.subsampling_conv_stride - 1)
        if not config.causal:
            padding = config.subsampling_conv_kernel_size - 1
        lengths = input_lengths.to(torch.float)
        subsampling_stages = config.subsampling_factor.bit_length() - 1
        for _ in range(subsampling_stages):
            lengths = torch.floor(
                (lengths + padding - config.subsampling_conv_kernel_size) / config.subsampling_conv_stride + 1.0
            )
        return lengths.to(torch.int)

    def _resolve_attn_context(self, num_lookahead_tokens: int | None = None) -> tuple[int, int]:
        """(left, right) attention context in encoder frames; (-1, FULL_CONTEXT) means unrestricted."""
        if not self.config.causal:
            return -1, FULL_CONTEXT
        if num_lookahead_tokens is None:
            num_lookahead_tokens = self.config.default_num_lookahead_tokens
        return self.config.sliding_window - 1, num_lookahead_tokens


def streaming_config_from_cohere(
    cohere_config, left_context: int = 70, lookahead: int = 13, causal: bool = True
) -> CohereStreamingEncoderConfig:
    """Copy Cohere encoder sizes and choose its streaming context."""
    encoder_config = cohere_config.encoder_config
    return CohereStreamingEncoderConfig(
        hidden_size=encoder_config.hidden_size,
        num_hidden_layers=encoder_config.num_hidden_layers,
        num_attention_heads=encoder_config.num_attention_heads,
        intermediate_size=encoder_config.intermediate_size,
        hidden_act=encoder_config.hidden_act,
        attention_bias=encoder_config.attention_bias,
        convolution_bias=encoder_config.convolution_bias,
        conv_kernel_size=encoder_config.conv_kernel_size,
        subsampling_factor=encoder_config.subsampling_factor,
        subsampling_conv_channels=encoder_config.subsampling_conv_channels,
        subsampling_conv_kernel_size=encoder_config.subsampling_conv_kernel_size,
        subsampling_conv_stride=encoder_config.subsampling_conv_stride,
        num_mel_bins=encoder_config.num_mel_bins,
        scale_input=encoder_config.scale_input,
        max_position_embeddings=encoder_config.max_position_embeddings,
        dropout=0.0,
        dropout_positions=0.0,
        layerdrop=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        sliding_window=left_context + 1,
        default_num_lookahead_tokens=lookahead,
        causal=causal,
    )


def _map_key(key: str) -> str | None:
    """Offline ParakeetEncoder key -> streaming encoder key (None = BatchNorm, folded separately)."""
    subsampling_names = {
        "subsampling.layers.0.": "subsampling.conv_in.",
        "subsampling.layers.2.": "subsampling.layers.0.depthwise_conv.",
        "subsampling.layers.3.": "subsampling.layers.0.pointwise_conv.",
        "subsampling.layers.5.": "subsampling.layers.1.depthwise_conv.",
        "subsampling.layers.6.": "subsampling.layers.1.pointwise_conv.",
    }
    for offline_prefix, streaming_prefix in subsampling_names.items():
        if key.startswith(offline_prefix):
            return streaming_prefix + key[len(offline_prefix):]
    if ".conv.norm." in key:
        return None
    return key


def _truncate_to_causal(w: torch.Tensor, time_dim: int) -> torch.Tensor:
    """Keep past and center kernel taps aligned, and discard future taps."""
    kernel_size = w.shape[time_dim]
    center = (kernel_size - 1) // 2
    causal_weights = torch.zeros_like(w)
    past_weights = w.narrow(time_dim, 0, center + 1)
    causal_weights.narrow(time_dim, kernel_size - 1 - center, center + 1).copy_(past_weights)
    return causal_weights


@torch.no_grad()
def convert_encoder_state_dict(
    offline_sd: dict[str, torch.Tensor], eps: float = 1e-5, conv_init: str = "shift"
) -> dict[str, torch.Tensor]:
    """Map offline weights to the streaming encoder and fold BatchNorm."""
    if conv_init not in ("shift", "truncate"):
        raise ValueError(f"unknown conv_init {conv_init!r}")
    streaming_weights = {}
    for key, value in offline_sd.items():
        streaming_key = _map_key(key)
        if streaming_key is not None:
            streaming_weights[streaming_key] = value
    normalization_prefixes = {key.rsplit(".conv.norm.", 1)[0] for key in offline_sd if ".conv.norm." in key}
    for prefix in normalization_prefixes:
        weight = offline_sd[f"{prefix}.conv.norm.weight"]
        bias = offline_sd[f"{prefix}.conv.norm.bias"]
        mean = offline_sd[f"{prefix}.conv.norm.running_mean"]
        variance = offline_sd[f"{prefix}.conv.norm.running_var"]
        scale = weight.float() / torch.sqrt(variance.float() + eps)
        streaming_weights[f"{prefix}.conv.norm.weight"] = scale.to(weight.dtype)
        streaming_weights[f"{prefix}.conv.norm.bias"] = (bias.float() - mean.float() * scale).to(weight.dtype)
    if conv_init == "truncate":
        for key in list(streaming_weights):
            weights = streaming_weights[key]
            is_conformer_kernel = key.endswith("conv.depthwise_conv.weight") and weights.dim() == 3
            is_subsampling_kernel = (
                key.startswith("subsampling.")
                and key.endswith(".weight")
                and weights.dim() == 4
                and weights.shape[2] > 1
            )
            if is_conformer_kernel or is_subsampling_kernel:
                streaming_weights[key] = _truncate_to_causal(weights, time_dim=2)
    return streaming_weights


def build_streaming_encoder(
    offline_encoder: nn.Module, cohere_config, conv_init: str = "shift", **config_kwargs
) -> CohereStreamingEncoder:
    """Convert an offline encoder, keeping its device and dtype."""
    config = streaming_config_from_cohere(cohere_config, **config_kwargs)
    encoder = CohereStreamingEncoder(config)
    batch_norm_epsilon = next(module.eps for module in offline_encoder.modules() if isinstance(module, nn.BatchNorm1d))
    streaming_weights = convert_encoder_state_dict(
        offline_encoder.state_dict(),
        eps=batch_norm_epsilon,
        conv_init=conv_init if config.causal else "shift",
    )
    missing, unexpected = encoder.load_state_dict(streaming_weights, strict=False)
    missing = [key for key in missing if not key.endswith("inv_freq")]
    if missing or unexpected:
        raise RuntimeError(f"Weight mapping incomplete. missing={missing[:10]} unexpected={unexpected[:10]}")
    reference = next(offline_encoder.parameters())
    return encoder.to(device=reference.device, dtype=reference.dtype).eval()


def encode_batch(fe, encoder, audios, norm_mode, stats, lookahead, device, dtype, prior_frames=100):
    """Extract normalized features and encode a batch of audio."""
    features, frame_mask = log_mel(fe, [torch.as_tensor(audio) for audio in audios])
    features = normalize(features, frame_mask, norm_mode, stats, prior_frames=prior_frames)
    features = features.to(device, dtype)
    encoder_options = {}
    if lookahead is not None:
        encoder_options["num_lookahead_tokens"] = lookahead
    return encoder(features, attention_mask=frame_mask.long().to(device), **encoder_options)


@torch.no_grad()
def aed_transcribe(
    model,
    processor,
    encoder,
    utts,
    norm_mode="utterance",
    stats=None,
    lookahead=None,
    device="cuda",
    dtype=torch.float32,
    bs=16,
    language="en",
    prior_frames=100,
    num_beams=1,
    suppress_tokens=None,
):
    """Return one decoder transcript per utterance; each utterance supplies audio and seconds."""
    prompt = torch.tensor([processor.get_decoder_prompt_ids(language=language, punctuation=True)], device=device)
    transcripts = []
    for start in range(0, len(utts), bs):
        batch = utts[start: start + bs]
        encoder_output = encode_batch(
            processor.feature_extractor,
            encoder,
            [utterance.audio for utterance in batch],
            norm_mode,
            stats,
            lookahead,
            device,
            dtype,
            prior_frames=prior_frames,
        )
        longest_duration = max(utterance.seconds for utterance in batch)
        generated_tokens = model.generate(
            encoder_outputs=encoder_output,
            decoder_input_ids=prompt.expand(len(batch), -1),
            max_new_tokens=min(256, int(10 + 8 * longest_duration)),
            num_beams=num_beams,
            suppress_tokens=suppress_tokens,
        )
        transcripts += processor.batch_decode(generated_tokens, skip_special_tokens=True)
    return transcripts


@torch.no_grad()
def ctc_greedy(
    ctc_head,
    processor,
    encoder,
    utts,
    norm_mode="causal",
    stats=None,
    lookahead=None,
    device="cuda",
    dtype=torch.float32,
    bs=16,
    prior_frames=100,
):
    """Return one greedy CTC transcript per utterance."""
    blank_token = ctc_head.out_features - 1
    transcripts = []
    for start in range(0, len(utts), bs):
        batch = utts[start: start + bs]
        encoder_output = encode_batch(
            processor.feature_extractor,
            encoder,
            [utterance.audio for utterance in batch],
            norm_mode,
            stats,
            lookahead,
            device,
            dtype,
            prior_frames=prior_frames,
        )
        predicted_tokens = ctc_head(encoder_output.last_hidden_state).argmax(-1)
        frame_counts = encoder_output.attention_mask.sum(-1)
        for tokens, frame_count in zip(predicted_tokens.tolist(), frame_counts.tolist()):
            transcript_tokens = []
            previous_token = None
            # Collapse repeats before removing blanks, as required by CTC.
            for token in tokens[:frame_count]:
                if token != previous_token and token != blank_token:
                    transcript_tokens.append(token)
                previous_token = token
            transcripts.append(processor.tokenizer.decode(transcript_tokens, skip_special_tokens=True))
    return transcripts


def _common_prefix(a: list[int], b: list[int]) -> int:
    prefix_length = 0
    for first_token, second_token in zip(a, b):
        if first_token != second_token:
            break
        prefix_length += 1
    return prefix_length


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _checkpoint_weights(state_dict: dict, prefix: str) -> dict:
    return {
        name[len(prefix):]: weights.float()
        for name, weights in state_dict.items()
        if name.startswith(prefix)
    }


class CohereStreaming:
    """Stream 16 kHz audio through reset(), step(audio), and finish()."""

    def __init__(
        self,
        checkpoint: str | None = DEFAULT_CHECKPOINT,
        lookahead: int = 6,
        device: str | None = None,
        dtype=torch.bfloat16,
        stats: dict | None = None,
        left_context: int = 70,
        conv_init: str = "truncate",
        partials: str = "aed",
        aed_interval_ms: int = 320,
        show_tentative: bool = True,
        holdback_words: int = 1,
        min_commit_ms: int = 1000,
        final: str = "free",
        prior_frames: int | None = None,
        final_lookahead: int | None = 13,
        final_beams: int = 4,
        suppress_brackets: bool = True,
        final_reencode_max_s: float = 30.0,
    ):
        if partials not in ("aed", "ctc"):
            raise ValueError(partials)
        if final not in ("forced", "free"):
            raise ValueError(final)
        self.device = device or _default_device()
        self.dtype = dtype
        stats, norm_mode, checkpoint_prior = self._load_model(
            checkpoint, stats, left_context, lookahead, conv_init
        )

        self.stats = stats
        self.lookahead = lookahead
        self.partials = partials
        self.show_tentative = show_tentative
        self.holdback_words = holdback_words
        self.min_commit_ms = min_commit_ms
        self.final = final
        self.final_beams = final_beams
        self.final_lookahead = None
        if final == "free" and partials == "aed":
            self.final_lookahead = final_lookahead
        self._reencode_max_frames = int(final_reencode_max_s * SR / self.processor.feature_extractor.hop_length)

        self.hop_ms = 80
        chunk_ms = (lookahead + 1) * self.hop_ms
        self.aed_every = max(1, math.ceil(aed_interval_ms / chunk_ms))
        self._first = 1 + 8 * lookahead
        self._per = 8 * (lookahead + 1)
        self.name = self._model_name(chunk_ms, suppress_brackets)
        self._configure_tokens(suppress_brackets)

        if prior_frames is not None:
            self.prior_frames = prior_frames
        elif checkpoint_prior is not None:
            self.prior_frames = checkpoint_prior
        else:
            self.prior_frames = 100
        self.featurizer = StreamingFeaturizer(
            self.processor.feature_extractor,
            stats,
            prior_frames=self.prior_frames,
            norm_mode=norm_mode,
        )

    def _load_model(self, checkpoint, stats, left_context, lookahead, conv_init):
        self.processor = AutoProcessor.from_pretrained(COHERE_MODEL_ID)
        model = CohereAsrForConditionalGeneration.from_pretrained(COHERE_MODEL_ID, dtype=torch.float32)
        state_dict_path = None
        checkpoint_prior = None
        norm_mode = "causal"
        if checkpoint is not None:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.exists():
                checkpoint_path = Path(snapshot_download(checkpoint))
            config = json.loads((checkpoint_path / "config.json").read_text())
            training_args = config["args"]
            left_context = config["encoder"]["sliding_window"] - 1
            conv_init = training_args.get("conv_init", conv_init)
            checkpoint_prior = training_args.get("eval_prior")
            norm_mode = training_args.get("norm", "causal")
            stats_path = checkpoint_path / "mel_stats.json"
            if stats is None and stats_path.exists():
                stats = load_stats(stats_path)
            state_dict_path = checkpoint_path / "student.safetensors"

        self.encoder = build_streaming_encoder(
            model.model.encoder,
            model.config,
            conv_init=conv_init,
            left_context=left_context,
            lookahead=lookahead,
        )
        self.ctc_head = nn.Linear(self.encoder.config.hidden_size, model.config.vocab_size + 1)
        if state_dict_path is not None:
            self._load_checkpoint_weights(model, state_dict_path)
        del model.model.encoder
        model.model.encoder = self.encoder
        self.model = model.to(self.device, self.dtype).eval()
        self.encoder = self.model.model.encoder
        self.ctc_head = self.ctc_head.to(self.device, self.dtype).eval()
        if stats is None:
            raise ValueError(
                "causal normalisation needs mel statistics (shipped with the checkpoint as mel_stats.json)"
            )
        return stats, norm_mode, checkpoint_prior

    def _load_checkpoint_weights(self, model, path):
        state_dict = load_file(str(path))
        self.encoder.load_state_dict(_checkpoint_weights(state_dict, "encoder."))
        self.ctc_head.load_state_dict(_checkpoint_weights(state_dict, "ctc_head."))
        missing, unexpected = model.load_state_dict(_checkpoint_weights(state_dict, "cohere."), strict=False)
        if unexpected or any(not name.startswith("model.encoder.") for name in missing):
            raise RuntimeError(f"decoder checkpoint mismatch: {unexpected[:5]} {missing[:5]}")

    def _model_name(self, chunk_ms: int, suppress_brackets: bool) -> str:
        suffix = ""
        if self.partials != "aed" or not self.show_tentative:
            suffix += f"-{self.partials}"
            if not self.show_tentative:
                suffix += "-committed"
        if self.final == "free":
            suffix += "-freefinal"
        if self.final_lookahead is not None:
            suffix += f"-reenc{self.final_lookahead}"
        if self.final_beams > 1:
            suffix += f"-beam{self.final_beams}"
        if suppress_brackets:
            suffix += "-nobrackets"
        return f"cohere-streaming-2b{suffix}@{chunk_ms}ms"

    def _configure_tokens(self, suppress_brackets: bool):
        tokenizer = self.processor.tokenizer
        vocabulary = tokenizer.get_vocab()
        self._special = set(tokenizer.all_special_ids)
        self._suppress = None
        if suppress_brackets:
            self._suppress = [
                token_id
                for token, token_id in vocabulary.items()
                if any(bracket in token for bracket in "()[]")
            ]
        self._word_start = {token_id for token, token_id in vocabulary.items() if token.startswith("▁")}

    def reset(self, language: str = "en") -> None:
        """Clear the previous utterance and choose the transcription language."""
        self.language = language
        self.featurizer.reset()
        self._pending = torch.zeros(0, self.featurizer.fe.feature_size)
        self._mel_all: list[torch.Tensor] | None = [] if self.final_lookahead is not None else None
        self._mel_n = 0
        self._mel_used = 0
        self._key_value_cache = None
        self._convolution_cache = None
        self._states: list[torch.Tensor] = []
        self._chunks = 0
        self._ctc_ids: list[int] = []
        self._prev = None
        self._committed: list[int] = []
        self._last_hyp: list[int] = []
        self._shown: list[int] = []
        self._n_samples = 0
        self._prompt = self.processor.get_decoder_prompt_ids(language=language, punctuation=True)

    @torch.no_grad()
    def _encode(self, chunk: torch.Tensor):
        encoder_output = self.encoder(
            chunk.unsqueeze(0).to(self.device, self.dtype),
            past_key_values=self._key_value_cache,
            padding_cache=self._convolution_cache,
            use_cache=True,
            num_lookahead_tokens=self.lookahead,
        )
        self._key_value_cache = encoder_output.past_key_values
        self._convolution_cache = encoder_output.padding_cache
        hidden_states = encoder_output.last_hidden_state
        self._states.append(hidden_states)
        self._chunks += 1
        if self.partials == "ctc":
            blank_token = self.ctc_head.out_features - 1
            predicted_tokens = self.ctc_head(hidden_states).argmax(-1)[0].tolist()
            for token in predicted_tokens:
                if token != self._prev and token != blank_token:
                    self._ctc_ids.append(token)
                self._prev = token

    def _next_size(self) -> int:
        # Subsampling needs one initial frame before the regular eight-frame stride.
        return self._first if self._mel_used == 0 else self._per

    def _feed(self, frames: torch.Tensor) -> int:
        if self._mel_all is not None:
            self._mel_all.append(frames)
            self._mel_n += frames.shape[0]
            if self._mel_n > self._reencode_max_frames:
                self._mel_all = None  # Release the endpoint buffer for long recordings.
        self._pending = torch.cat([self._pending, frames])
        encoded_chunks = 0
        chunk_size = self._next_size()
        while self._pending.shape[0] >= chunk_size:
            self._encode(self._pending[:chunk_size])
            self._pending = self._pending[chunk_size:]
            self._mel_used += chunk_size
            encoded_chunks += 1
            chunk_size = self._next_size()
        return encoded_chunks

    @torch.no_grad()
    def _aed(self, states: torch.Tensor, max_new: int, num_beams: int = 1) -> list[int]:
        """Decode encoder states using the committed tokens as a prefix."""
        encoder_output = BaseModelOutput(last_hidden_state=states)
        encoder_output["attention_mask"] = torch.ones(states.shape[:2], dtype=torch.long, device=states.device)
        decoder_input = torch.tensor([self._prompt + self._committed], device=self.device)
        generated_tokens = self.model.generate(
            encoder_outputs=encoder_output,
            decoder_input_ids=decoder_input,
            max_new_tokens=max_new,
            num_beams=num_beams,
            suppress_tokens=self._suppress,
        )
        text_tokens = generated_tokens[0, len(self._prompt):].tolist()
        return [token for token in text_tokens if token not in self._special]

    def _hold_back(self, ids: list[int], words: int) -> list[int]:
        """Leave the last words uncommitted because their audio may be incomplete."""
        for _ in range(words):
            word_starts = [index for index, token in enumerate(ids) if token in self._word_start]
            if not word_starts:
                return []
            ids = ids[:word_starts[-1]]
        return ids

    def _partial_update(self):
        states = torch.cat(self._states, dim=1)
        hypothesis = self._aed(states, max_new=24)
        audio_duration_ms = self._n_samples * 1000 / SR
        if audio_duration_ms >= self.min_commit_ms:
            agreed_tokens = hypothesis[:_common_prefix(hypothesis, self._last_hyp)]
            agreed_tokens = self._hold_back(agreed_tokens, self.holdback_words)
            if len(agreed_tokens) > len(self._committed):
                self._committed = agreed_tokens
        self._last_hyp = hypothesis
        self._shown = hypothesis if self.show_tentative else self._committed

    def _text(self, ids: list[int]) -> str:
        return self.processor.tokenizer.decode(ids, skip_special_tokens=True)

    def step(self, audio: np.ndarray) -> str:
        """Feed an 80 ms chunk of 16 kHz audio and return the transcript so far."""
        self._n_samples += len(audio)
        previous_chunks = self._chunks
        new_frames = self.featurizer.push(audio)
        self._feed(new_frames)
        if self.partials == "ctc":
            return self._text(self._ctc_ids)
        has_new_output = self._chunks > previous_chunks
        decode_interval_reached = self._chunks // self.aed_every > previous_chunks // self.aed_every
        if has_new_output and decode_interval_reached:
            self._partial_update()
        return self._text(self._shown)

    @torch.no_grad()
    def finish(self) -> str:
        """Flush remaining audio and return the final transcript."""
        tail = self.featurizer.flush()
        token_limit = min(256, int(10 + 8 * self._n_samples / SR))
        can_reencode = self._mel_all is not None and self._mel_n + tail.shape[0] <= self._reencode_max_frames
        if can_reencode:
            return self._finish_with_reencode(tail, token_limit)
        return self._finish_streamed(tail, token_limit)

    def _finish_with_reencode(self, tail: torch.Tensor, token_limit: int) -> str:
        features = torch.cat(self._mel_all + [tail])
        if features.shape[0] == 0:
            return ""
        encoder_output = self.encoder(
            features.unsqueeze(0).to(self.device, self.dtype),
            num_lookahead_tokens=self.final_lookahead,
        )
        states = encoder_output.last_hidden_state
        if states.shape[1] == 0:
            return ""
        self._committed = []
        tokens = self._aed(states, max_new=token_limit, num_beams=self.final_beams)
        return self._text(tokens)

    def _finish_streamed(self, tail: torch.Tensor, token_limit: int) -> str:
        remaining_frames = torch.cat([self._pending, tail])
        total_frames = self._mel_used + remaining_frames.shape[0]
        self._pending = torch.zeros(0, remaining_frames.shape[1])
        while remaining_frames.shape[0]:
            chunk_size = self._next_size()
            chunk = remaining_frames[:chunk_size]
            remaining_frames = remaining_frames[chunk_size:]
            if chunk.shape[0] < chunk_size:
                chunk = nn.functional.pad(chunk, (0, 0, 0, chunk_size - chunk.shape[0]))
            self._encode(chunk)
            self._mel_used += chunk_size
        if not self._states:
            return ""
        valid_frames = int(self.encoder._get_subsampling_output_length(torch.tensor([total_frames]))[0])
        states = torch.cat(self._states, dim=1)[:, :valid_frames]
        if self.partials == "ctc" or self.final == "free":
            self._committed = []
        tokens = self._aed(states, max_new=token_limit, num_beams=self.final_beams)
        return self._text(tokens)
