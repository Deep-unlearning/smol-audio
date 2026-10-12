"""Cohere Transcribe, made streaming: feed audio 80 ms at a time, get live text.

HOW IT WORKS (a map of this file, in reading order)

    1. Constants      model id, default checkpoint, sample rate, ...
    2. Features       `log_mel`, `normalize`, `StreamingFeaturizer`: waveform -> normalised log-mel frames, computed
                      incrementally so no frame needs the future or the whole utterance.
    3. Encoder        `build_streaming_encoder`: converts Cohere's offline encoder into a cache-aware streaming one
                      (chunked attention, causal convolutions, folded BatchNorm).
    4. Decoding       `aed_transcribe` / `ctc_greedy`: batch helpers that run encoder + decoder (used by training / eval).
    5. CohereStreaming  the user-facing class: reset() / step() / finish().

QUICK START

    from model import CohereStreaming

    asr = CohereStreaming()                 # downloads Cohere Transcribe + the streaming weights
    asr.reset(language="en")                # start a new utterance
    for hop in audio_16khz_float32.reshape(-1, 1280):    # 80 ms = 1280 samples at 16 kHz
        print(asr.step(hop))                # transcript so far, updated live
    print(asr.finish())                     # final, most accurate transcript

`CohereStreaming` wraps the streaming Cohere Transcribe model behind three calls:

        asr.reset(language="en")   # start a new utterance
        text = asr.step(audio)     # feed 80 ms of 16 kHz audio, get the transcript so far
        text = asr.finish()        # the audio ended: get the final, most accurate transcript

WHY IT IS BUILT THIS WAY
    The offline Cohere model needs the whole recording before it can answer. Here the audio encoder is
    "cache-aware": it keeps its attention keys/values and convolution state from earlier chunks, so each new chunk
    costs the same no matter how long you have been talking. The (small) text decoder is then run every so often
    on all the encoder output so far to produce a live caption. That is why this is a "two-pass" system: a fast
    live pass while audio arrives, and one careful pass at the end.

THE LIFECYCLE
    reset()   clears every piece of state below.
    step()    is called once per 80 ms of audio. It
                1. turns audio into mel frames (`StreamingFeaturizer`, same numbers as the offline path),
                2. queues the frames and runs the encoder whenever a full chunk is available
                   (the first chunk has 1 + 8R frames, later ones 8(R+1), where R is the `lookahead`:
                   the caption lags the speaker by about (R + 1) * 80 ms),
                3. every `aed_interval_ms` runs the decoder over all encoder output so far and updates the
                   caption (see "LocalAgreement" below).
    finish()  is called once at the end of the audio and returns the final transcript.

STATE KEPT BETWEEN CHUNKS (all created in `reset()`)
    featurizer          running mel / normalisation state
    _pending            mel frames received but not yet big enough to make a chunk
    _pkv, _pc           the encoder's attention (key/value) cache and convolution cache
    _states             encoder outputs of every chunk so far (the decoder reads all of them)
    _committed          token ids we are confident in and will not change (LocalAgreement, below)
    _last_hyp           the previous decode, to compare against the next one
    _shown              what `step()` currently displays
    _mel_all            every mel frame so far, kept only for the endpoint re-encode (below)

LOCALAGREEMENT (how live text stays stable)
    A decode on partial audio can change its mind about the last word. So we decode repeatedly and only
    *commit* the longest token prefix on which two consecutive decodes agree, after also dropping the last
    `holdback_words` words (the ones cut off by the audio edge). Later decodes are *forced* to start with the
    committed tokens, so committed text never changes. The caption shown is committed text plus the newest
    unconfirmed tail (`show_tentative=True`) or committed text only.

THE ENDPOINT RE-ENCODE (`final="free"`, the default)
    Live chunks only see a short look into the future. At the end of the audio we have everything, so
    `finish()` throws the streamed encoder states away and encodes the whole utterance again in a single pass
    with a longer lookahead (`final_lookahead=13`, i.e. 1.12 s, a lookahead the model was trained with), then
    decodes with beam search. That gives the best accuracy, at the price that already-shown words may change at
    the very end. Because this needs the whole utterance in memory, it is capped at `final_reencode_max_s`; past
    that the buffer is dropped and the final decode uses the streamed states instead.
    `final="forced"` instead decodes the streamed states once more, keeping the committed words.

OTHER OPTIONS
    partials="ctc"     live text from the first-pass CTC head (greedy) instead of the decoder; for ablations.
    prior_frames       warm-up frames for the causal normalisation; defaults to the checkpoint's `eval_prior`
                       (30 for v2), else 100 (as trained).
    suppress_brackets  stop the decoder from writing "(...)" asides, which scoring normalisers would drop.
"""

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

# =====================================================================================================================
# 1. CONSTANTS
# =====================================================================================================================

COHERE_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"  # the offline teacher: tokenizer, decoder, feature extractor
DEFAULT_CHECKPOINT = "Steveeeeeeen/cohere-streaming"  # streaming student weights used by `CohereStreaming`
FULL_CONTEXT = 10**6  # "attend to everything": used as the lookahead when causal=False
EPSILON = 1e-5  # same constant as the reference feature extractor
SR = 16000  # sample rate (Hz) of all audio

# =====================================================================================================================
# 2. FEATURES: log-mel frames that can be computed while audio is still arriving
# =====================================================================================================================

#
# WHAT: turns waveforms into log-mel frames (T frames x n_mels) and normalises them so the model sees values
# with roughly zero mean and unit variance per mel bin.
# WHY: Cohere's feature extractor normalises each mel bin with the mean/std of the *whole utterance*, which a
# streaming model cannot know in advance. We reuse its STFT / dither / pre-emphasis and swap the normalisation:
#
# * ``utterance`` - original behaviour (reference; must match `CohereAsrFeatureExtractor` exactly)
# * ``global``    - fixed per-bin mean/std estimated once on a corpus (`estimate_stats`)
# * ``causal``    - running per-bin mean/var over frames <= t, warm-started from the global stats as if
#                   ``prior_frames`` frames with those stats had already been seen. Only running sums are
#                   needed, so it is fully streamable.
#
# `StreamingFeaturizer` computes exactly the ``causal`` features incrementally, one audio chunk at a time.


def log_mel(fe, waveforms: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Un-normalised log-mel `(B, T, n_mels)` and frame mask `(B, T)`, following `CohereAsrFeatureExtractor`.

    Waveforms of different lengths are zero-padded into one batch; the mask marks the real frames.
    """
    lengths = torch.tensor([w.shape[0] for w in waveforms])
    x = torch.zeros(len(waveforms), int(lengths.max()))
    for i, w in enumerate(waveforms):
        x[i, : w.shape[0]] = w.float()
    x = fe._apply_dither(x, lengths)
    if fe.preemphasis is not None:
        timemask = torch.arange(x.shape[1])[None, :] < lengths[:, None]
        x = torch.cat([x[:, :1], x[:, 1:] - fe.preemphasis * x[:, :-1]], dim=1).masked_fill(~timemask, 0.0)
    feats = fe._torch_extract_fbank_features(x, "cpu")
    n_frames = torch.floor_divide(lengths + fe.n_fft // 2 * 2 - fe.n_fft, fe.hop_length)
    mask = torch.arange(feats.shape[1])[None, :] < n_frames[:, None]
    return feats, mask


def normalize(
    feats: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "utterance",
    stats: dict | None = None,
    prior_frames: int = 100,
) -> torch.Tensor:
    """Normalise each mel bin of `feats` (B, T, n_mels); `mask` (B, T) marks real frames. Padding becomes 0.

    Args:
        mode: "utterance", "global" or "causal" (see the module docstring).
        stats: corpus {"mean", "std"} lists, required for "global" and "causal".
        prior_frames: weight of the global stats in the "causal" running average.
    """
    m = mask.unsqueeze(-1).to(feats.dtype)
    if mode == "utterance":
        n = mask.sum(1, keepdim=True).to(feats.dtype)
        mean = (feats * m).sum(1, keepdim=True) / n.unsqueeze(-1)
        var = (((feats * m) - mean) ** 2 * m).sum(1, keepdim=True) / (n - 1).unsqueeze(-1)
        out = (feats - mean) / (var.sqrt() + EPSILON)
    elif mode == "global":
        mean, std = _stats(stats, feats)
        out = (feats - mean) / (std + EPSILON)
    elif mode == "causal":
        g_mean, g_std = _stats(stats, feats)
        k = float(prior_frames)
        # Running sums over frames 0..t (cumsum along time), started from k "virtual" frames of global stats:
        # s1 = sum of x, s2 = sum of x^2, n = frame count; then mean = s1/n and var = s2/n - mean^2.
        s1 = torch.cumsum(feats * m, dim=1) + k * g_mean
        s2 = torch.cumsum((feats * m) ** 2, dim=1) + k * (g_std**2 + g_mean**2)
        n = torch.cumsum(m, dim=1) + k
        mean = s1 / n
        var = (s2 / n - mean**2).clamp_min(1e-8)
        out = (feats - mean) / (var.sqrt() + EPSILON)
    else:
        raise ValueError(f"unknown normalisation mode {mode!r}")
    return out * m


def _stats(stats: dict | None, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Corpus mean and std as tensors of shape (1, 1, n_mels), matching `like`'s dtype."""
    if stats is None:
        raise ValueError("global/causal normalisation needs corpus statistics (see `estimate_stats`).")
    mean = torch.tensor(stats["mean"], dtype=like.dtype).view(1, 1, -1)
    std = torch.tensor(stats["std"], dtype=like.dtype).view(1, 1, -1)
    return mean, std


def estimate_stats(fe, waveforms: list[torch.Tensor]) -> dict:
    """Per-mel-bin mean and std of the un-normalised log-mel over all valid frames of `waveforms`."""
    s1 = s2 = n = 0
    for w in waveforms:
        f, mk = log_mel(fe, [w])
        f = f[mk]
        s1, s2, n = s1 + f.sum(0), s2 + (f.double() ** 2).sum(0), n + f.shape[0]
    mean = s1.double() / n
    std = (s2 / n - mean**2).sqrt()
    return {"mean": mean.tolist(), "std": std.tolist(), "frames": int(n)}


def save_stats(stats: dict, path: str | Path):
    """Write the stats dict from `estimate_stats` to a JSON file."""
    Path(path).write_text(json.dumps(stats))


def load_stats(path: str | Path) -> dict:
    """Read a stats JSON file written by `save_stats`."""
    return json.loads(Path(path).read_text())


class StreamingFeaturizer:
    """Incremental log-mel + streamable normalisation (O(new audio) per call).

    Reproduces `log_mel` + `normalize(..., "causal")` frame for frame: the signal is dithered and pre-emphasised
    as it arrives, a virtual `n_fft // 2` zero pad stands in for STFT centring, and a frame is emitted only once
    its whole window is available, so no frame is ever recomputed. The running per-bin sums carry the causal
    statistics across calls. `flush()` applies the end-of-signal centre padding and emits the remaining valid
    frames (`len // hop`, as the reference feature mask). Dither uses a per-utterance sequential noise stream
    (the batch extractor seeds by total length, which a stream cannot know), so frames match exactly only with
    dither disabled; the noise itself is ~1e-5 and statistically identical.
    """

    def __init__(self, fe, stats: dict, prior_frames: int = 100, seed: int = 0, norm_mode: str = "causal"):
        """`fe` is Cohere's feature extractor, `stats` the corpus stats, `seed` seeds the dither noise."""
        if norm_mode not in ("causal", "global"):
            raise ValueError(f"streaming features require causal or global normalisation, got {norm_mode!r}")
        self.fe, self.stats, self.k = fe, stats, float(prior_frames)
        self.norm_mode = norm_mode
        self.hop, self.n_fft, self.win_length = fe.hop_length, fe.n_fft, fe.win_length
        self.window = torch.hann_window(self.win_length, periodic=False)
        self.g_mean = torch.tensor(stats["mean"], dtype=torch.float64)
        self.g_std = torch.tensor(stats["std"], dtype=torch.float64)
        self.seed = seed
        self.reset()

    def reset(self):
        """Forget all audio and start a new utterance."""
        self.y = torch.zeros(self.n_fft // 2)  # virtual centre pad at the start
        self.n_samples = 0
        self.frames_out = 0
        self.last_x = None
        self.gen = torch.Generator().manual_seed(self.seed)
        self.s1 = self.k * self.g_mean
        self.s2 = self.k * (self.g_std**2 + self.g_mean**2)
        self.n = self.k

    def _append(self, audio: torch.Tensor):
        """Dither and pre-emphasise new samples (1-D) and add them to the signal buffer `self.y`."""
        x = audio.float()
        if self.fe.dither > 0:
            x = x + self.fe.dither * torch.randn(x.shape[0], generator=self.gen)
        if self.fe.preemphasis is not None:
            first = torch.zeros(1) if self.last_x is None else self.last_x.view(1)  # y[0] = x[0] at the very start
            y = x - self.fe.preemphasis * torch.cat([first, x[:-1]])
        else:
            y = x
        if x.shape[0]:
            self.last_x = x[-1]
        self.y = torch.cat([self.y, y])
        self.n_samples += x.shape[0]

    def _frames(self, upto: int) -> torch.Tensor:
        """Emit frames [frames_out, upto): log-mel -> causal normalisation."""
        a, b = self.frames_out, upto
        if b <= a:
            return torch.zeros(0, self.fe.feature_size)
        # The buffer begins at frame `a`; samples preceding it were released after the previous call.
        seg = self.y[: (b - a - 1) * self.hop + self.n_fft]
        stft = torch.stft(seg, self.n_fft, hop_length=self.hop, win_length=self.win_length, window=self.window,
                          center=False, return_complex=True)
        mag = torch.view_as_real(stft).pow(2).sum(-1)  # |X|^2, as the reference
        mel = torch.log(self.fe.mel_filters.to(mag.dtype) @ mag + LOG_ZERO_GUARD_VALUE).T  # (frames, n_mels)
        if self.norm_mode == "global":
            out = (mel - self.g_mean.float()) / (self.g_std.float() + EPSILON)
        else:
            # Causal normalisation in float64 (long running sums lose precision in float32).
            m64 = mel.double()
            c1 = torch.cumsum(m64, 0) + self.s1
            c2 = torch.cumsum(m64**2, 0) + self.s2
            cn = torch.arange(1, m64.shape[0] + 1, dtype=torch.float64).unsqueeze(1) + self.n
            mean = c1 / cn
            var = (c2 / cn - mean**2).clamp_min(1e-8)
            out = ((m64 - mean) / (var.sqrt() + EPSILON)).float()
            self.s1, self.s2, self.n = c1[-1], c2[-1], float(cn[-1, 0])
        self.frames_out = b
        # Clone so the small remainder does not retain the storage of the full incoming chunk.
        self.y = self.y[(b - a) * self.hop:].clone()
        return out

    def push(self, audio) -> torch.Tensor:
        """Add a chunk of 16 kHz samples; return the newly completed normalised frames, shape (n_new, n_mels)."""
        self._append(torch.as_tensor(audio))
        complete = self.frames_out + max(0, (self.y.shape[0] - self.n_fft) // self.hop + 1)
        valid = self.n_samples // self.hop  # never emit past the reference mask
        return self._frames(min(complete, valid))

    def flush(self) -> torch.Tensor:
        """Signal end of audio; return the last frames, shape (n_last, n_mels)."""
        self.y = torch.cat([self.y, torch.zeros(self.n_fft // 2)])  # centre pad at the end
        return self._frames(self.n_samples // self.hop)

# =====================================================================================================================
# 3. STREAMING ENCODER: turn Cohere's offline encoder into a cache-aware streaming encoder
# =====================================================================================================================

#
# WHAT: `build_streaming_encoder` takes the offline Cohere Transcribe encoder and returns a
# `CohereStreamingEncoder` that can process audio a few frames at a time, carrying small caches between calls.
# WHY: the offline encoder needs the entire recording before it can emit anything; a streaming encoder lets us
# show text while the user is still talking.
#
# We reuse transformers' `NemotronAsrStreamingEncoder` (a cache-aware FastConformer) and adapt it so that the
# weights of Cohere's offline encoder load into it. Four ideas are worth knowing:
#
# * Chunked attention: each frame attends to `left_context` past frames and `lookahead` (R) future frames, not
#   to the whole utterance. Larger R means better accuracy but more latency (each unit is one encoder frame,
#   8 mel frames of audio).
# * Caches: between chunks we keep the attention keys/values of the last `left_context` frames (`past_key_values`)
#   and the last few inputs of every causal convolution (`padding_cache`), so a chunk never recomputes the past.
# * Causal convolutions: a convolution may only look at the past, so each output frame is final once its input
#   has arrived. Cohere's subsampling pads the frequency axis symmetrically (128 mels -> 16 bins, linear in=4096)
#   while NeMo's causal Conv2D pads it causally (-> 17 bins). We keep Cohere's frequency padding so the weights
#   transfer exactly; only the *time* axis becomes causal.
# * BatchNorm -> `ChannelAffine`: Cohere's conformer conv module uses BatchNorm, the streaming encoder uses
#   LayerNorm. At inference BatchNorm is a per-channel scale and shift (time-local, so streaming-safe), so we
#   replace the LayerNorm with a `ChannelAffine` initialised from the folded BatchNorm statistics.
#
# `causal=False` restores symmetric time padding and full attention. In that mode the converted encoder must
# reproduce the offline encoder exactly, which is how we test that the weight mapping is correct.


@strict
class CohereStreamingEncoderConfig(NemotronAsrStreamingEncoderConfig):
    """Nemotron streaming config plus a `causal` switch (False = offline-equivalent, used for parity tests)."""

    model_type = "cohere_streaming_encoder"

    causal: bool = True

    @property
    def subsampling_out_hidden_size(self) -> int:
        """Width of the flattened subsampling output (channels * remaining frequency bins)."""
        # Cohere pads frequency symmetrically: (k - 1) // 2 on each side.
        pad = self.subsampling_conv_kernel_size - 1
        out = self.num_mel_bins
        for _ in range(len(bin(self.subsampling_factor)) - 3):  # log2
            out = (out + pad - self.subsampling_conv_kernel_size) // self.subsampling_conv_stride + 1
        return self.subsampling_conv_channels * out


class ChannelAffine(nn.Module):
    """Per-channel scale and shift over the last dim; the folded form of inference-time BatchNorm."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., channels)
        return x * self.weight + self.bias


class CohereCausalConv2D(nas.NemotronAsrStreamingEncoderCausalConv2D):
    """Time-causal, frequency-symmetric subsampling conv."""

    @property
    def freq_pad(self) -> tuple[int, int]:
        """(left, right) padding on the frequency axis: symmetric, as in Cohere's offline subsampling."""
        p = (self.kernel_size[1] - 1) // 2
        return (p, p)


class CohereSymmetricConv2D(CohereCausalConv2D):
    """Offline (parity) variant: symmetric padding on time as well."""

    @property
    def time_pad(self) -> tuple[int, int]:
        """(left, right) padding on the time axis: symmetric, so the conv also sees the future."""
        p = (self.kernel_size[0] - 1) // 2
        return (p, p)


class CohereSymmetricConv1d(nas.NemotronAsrStreamingEncoderCausalConv1d):
    """Offline (parity) variant of the depthwise conv.

    The streaming encoder masks padded *keys* only, so padded frames are not zeroed before the conv. That is
    harmless for causal convs (right padding is never read) but leaks through a symmetric kernel, so here we
    zero padded frames explicitly, like the offline encoder does. `frame_mask` (B, 1, T) is set per forward.
    """

    frame_mask: torch.Tensor | None = None

    def forward(self, x, padding_cache=None):
        if padding_cache is not None:
            raise ValueError("Symmetric (causal=False) convolutions cannot stream.")
        if self.frame_mask is not None:
            x = x.masked_fill(~self.frame_mask, 0.0)
        p = (self.kernel_size[0] - 1) // 2
        return nn.Conv1d.forward(self, nn.functional.pad(x, (p, p)))


class CohereStreamingEncoder(NemotronAsrStreamingEncoder):
    """Nemotron streaming encoder with Cohere's subsampling padding and folded-BatchNorm conv modules."""

    config: CohereStreamingEncoderConfig

    def __init__(self, config: CohereStreamingEncoderConfig):
        super().__init__(config)
        # Swap the conv classes in place: the parameters stay, only the padding behaviour changes.
        conv2d_cls = CohereCausalConv2D if config.causal else CohereSymmetricConv2D
        self.subsampling.conv_in.__class__ = conv2d_cls
        for layer in self.subsampling.layers:
            layer.depthwise_conv.__class__ = conv2d_cls
        for block in self.layers:
            block.conv.norm = ChannelAffine(config.hidden_size)
            if not config.causal:
                block.conv.depthwise_conv.__class__ = CohereSymmetricConv1d

    def forward(self, input_features, attention_mask=None, **kwargs):
        """input_features: (B, T_mel, n_mels); attention_mask: (B, T_mel), 1 = real frame, 0 = padding."""
        if not self.config.causal:
            # Offline mode: tell every depthwise conv which (subsampled) frames are padding.
            frame_mask = None
            if attention_mask is not None:
                n_out = int(self._get_subsampling_output_length(torch.tensor([input_features.shape[1]]))[0])
                frame_mask = self._get_output_attention_mask(attention_mask, target_length=n_out).unsqueeze(1)
            for block in self.layers:
                block.conv.depthwise_conv.frame_mask = frame_mask
        return super().forward(input_features, attention_mask=attention_mask, **kwargs)

    def _get_subsampling_output_length(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Number of encoder frames produced from `input_lengths` mel frames (applies each strided conv in turn)."""
        c = self.config
        pad = (c.subsampling_conv_kernel_size - 1) + (c.subsampling_conv_stride - 1)
        if not c.causal:
            pad = c.subsampling_conv_kernel_size - 1
        lengths = input_lengths.to(torch.float)
        for _ in range(len(bin(c.subsampling_factor)) - 3):
            lengths = torch.floor((lengths + pad - c.subsampling_conv_kernel_size) / c.subsampling_conv_stride + 1.0)
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
    """Copy the sizes of Cohere's encoder config and add the streaming attention settings.

    Args:
        cohere_config: the Cohere model config (its `encoder_config` is read).
        left_context: past encoder frames each frame may attend to.
        lookahead: default number of future encoder frames (R) each frame may attend to.
        causal: False builds the offline-equivalent encoder.
    """
    ec = cohere_config.encoder_config
    return CohereStreamingEncoderConfig(
        hidden_size=ec.hidden_size,
        num_hidden_layers=ec.num_hidden_layers,
        num_attention_heads=ec.num_attention_heads,
        intermediate_size=ec.intermediate_size,
        hidden_act=ec.hidden_act,
        attention_bias=ec.attention_bias,
        convolution_bias=ec.convolution_bias,
        conv_kernel_size=ec.conv_kernel_size,
        subsampling_factor=ec.subsampling_factor,
        subsampling_conv_channels=ec.subsampling_conv_channels,
        subsampling_conv_kernel_size=ec.subsampling_conv_kernel_size,
        subsampling_conv_stride=ec.subsampling_conv_stride,
        num_mel_bins=ec.num_mel_bins,
        scale_input=ec.scale_input,
        max_position_embeddings=ec.max_position_embeddings,
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
    sub = {
        "subsampling.layers.0.": "subsampling.conv_in.",
        "subsampling.layers.2.": "subsampling.layers.0.depthwise_conv.",
        "subsampling.layers.3.": "subsampling.layers.0.pointwise_conv.",
        "subsampling.layers.5.": "subsampling.layers.1.depthwise_conv.",
        "subsampling.layers.6.": "subsampling.layers.1.pointwise_conv.",
    }
    for old, new in sub.items():
        if key.startswith(old):
            return new + key[len(old):]
    if ".conv.norm." in key:
        return None
    return key


def _truncate_to_causal(w: torch.Tensor, time_dim: int) -> torch.Tensor:
    """Centred kernel -> causal kernel that keeps the past + centre taps *aligned* and drops future taps.

    Plain causal re-padding reuses the centred kernel as-is, which shifts every tap `c` frames into the past
    (c = (k-1)//2); over 48 layers that misalignment destroys the representation. Truncation instead maps
    offset o in [-c, 0] from index o+c to index o+k-1 and zeroes the rest.

    Args:
        w: kernel weights of any shape; the kernel size k is `w.shape[time_dim]`.
        time_dim: which dimension of `w` is the time axis.
    """
    k = w.shape[time_dim]
    c = (k - 1) // 2
    out = torch.zeros_like(w)
    src = w.narrow(time_dim, 0, c + 1)
    out.narrow(time_dim, k - 1 - c, c + 1).copy_(src)
    return out


@torch.no_grad()
def convert_encoder_state_dict(
    offline_sd: dict[str, torch.Tensor], eps: float = 1e-5, conv_init: str = "shift"
) -> dict[str, torch.Tensor]:
    """Map offline encoder weights to the streaming encoder, folding BatchNorm into `ChannelAffine`.

    Args:
        offline_sd: state dict of the offline encoder.
        eps: BatchNorm epsilon of the offline encoder.
        conv_init: "shift" reuses the centred kernels as-is under causal padding; "truncate" re-aligns them
            (see `_truncate_to_causal`). Irrelevant for causal=False.

    Returns:
        State dict for `CohereStreamingEncoder`.
    """
    if conv_init not in ("shift", "truncate"):
        raise ValueError(f"unknown conv_init {conv_init!r}")
    out = {}
    for key, value in offline_sd.items():
        new = _map_key(key)
        if new is not None:
            out[new] = value
    # BatchNorm(x) = g * (x - mean) / sqrt(var + eps) + b  ==  x * scale + shift  with
    #   scale = g / sqrt(var + eps),  shift = b - mean * scale
    prefixes = {k.rsplit(".conv.norm.", 1)[0] for k in offline_sd if ".conv.norm." in k}
    for p in prefixes:
        g, b = offline_sd[f"{p}.conv.norm.weight"], offline_sd[f"{p}.conv.norm.bias"]
        mean, var = offline_sd[f"{p}.conv.norm.running_mean"], offline_sd[f"{p}.conv.norm.running_var"]
        scale = g.float() / torch.sqrt(var.float() + eps)
        out[f"{p}.conv.norm.weight"] = scale.to(g.dtype)
        out[f"{p}.conv.norm.bias"] = (b.float() - mean.float() * scale).to(g.dtype)
    if conv_init == "truncate":
        for key in list(out):
            if key.endswith("conv.depthwise_conv.weight") and out[key].dim() == 3:  # (C, 1, k) conformer conv
                out[key] = _truncate_to_causal(out[key], time_dim=2)
            elif key.startswith("subsampling.") and key.endswith(".weight") and out[key].dim() == 4 \
                    and out[key].shape[2] > 1:  # (out, in, k_time, k_freq) strided subsampling convs
                out[key] = _truncate_to_causal(out[key], time_dim=2)
    return out


def build_streaming_encoder(
    offline_encoder: nn.Module, cohere_config, conv_init: str = "shift", **config_kwargs
) -> CohereStreamingEncoder:
    """Create a streaming encoder and load the offline encoder's weights into it.

    Args:
        offline_encoder: the Cohere (Parakeet) encoder module to convert.
        cohere_config: the Cohere model config.
        conv_init: "shift" or "truncate", see `convert_encoder_state_dict`.
        **config_kwargs: `left_context`, `lookahead`, `causal` (see `streaming_config_from_cohere`).

    Returns:
        The streaming encoder in eval mode, on the offline encoder's device and dtype.
    """
    cfg = streaming_config_from_cohere(cohere_config, **config_kwargs)
    enc = CohereStreamingEncoder(cfg)
    bn_eps = next(m.eps for m in offline_encoder.modules() if isinstance(m, nn.BatchNorm1d))
    sd = convert_encoder_state_dict(offline_encoder.state_dict(), eps=bn_eps, conv_init=conv_init if cfg.causal else "shift")
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("inv_freq")]
    if missing or unexpected:
        raise RuntimeError(f"Weight mapping incomplete. missing={missing[:10]} unexpected={unexpected[:10]}")
    ref = next(offline_encoder.parameters())
    return enc.to(device=ref.device, dtype=ref.dtype).eval()

# =====================================================================================================================
# 4. DECODING HELPERS: batch audio -> text with a (streaming) encoder
# =====================================================================================================================

#
# Used by the zero-shot study, the training-time evaluation and the streaming model. Both helpers first run
# `encode_batch`, then decode in one of two ways:
#
# * `aed_transcribe` - second pass: Cohere's attention encoder-decoder (AED) generates text from the encoder states
# * `ctc_greedy`     - first pass: greedy CTC over the student's CTC head (cheap, good for live partial text)


def encode_batch(fe, encoder, audios, norm_mode, stats, lookahead, device, dtype, prior_frames=100):
    """Audio -> normalised log-mel -> encoder output (hidden states and frame mask).

    `lookahead=None` keeps the encoder's default lookahead (R).
    """
    feats, mask = log_mel(fe, [torch.as_tensor(a) for a in audios])
    feats = normalize(feats, mask, norm_mode, stats, prior_frames=prior_frames).to(device, dtype)
    kw = {} if lookahead is None else {"num_lookahead_tokens": lookahead}
    return encoder(feats, attention_mask=mask.long().to(device), **kw)


@torch.no_grad()
def aed_transcribe(model, processor, encoder, utts, norm_mode="utterance", stats=None, lookahead=None,
                   device="cuda", dtype=torch.float32, bs=16, language="en", prior_frames=100, num_beams=1,
                   suppress_tokens=None):
    """Transcribe `utts` (objects with `.audio` and `.seconds`) with the AED decoder; returns one string each."""
    prompt = torch.tensor([processor.get_decoder_prompt_ids(language=language, punctuation=True)], device=device)
    hyps = []
    for i in range(0, len(utts), bs):
        batch = utts[i: i + bs]
        enc = encode_batch(processor.feature_extractor, encoder, [u.audio for u in batch], norm_mode, stats,
                           lookahead, device, dtype, prior_frames=prior_frames)
        out = model.generate(
            encoder_outputs=enc,  # carries `attention_mask`, which the decoder's cross-attention reads
            decoder_input_ids=prompt.expand(len(batch), -1),
            max_new_tokens=min(256, int(10 + 8 * max(u.seconds for u in batch))),  # stops hallucination loops
            num_beams=num_beams,
            suppress_tokens=suppress_tokens,
        )
        hyps += processor.batch_decode(out, skip_special_tokens=True)
    return hyps


@torch.no_grad()
def ctc_greedy(ctc_head, processor, encoder, utts, norm_mode="causal", stats=None, lookahead=None,
               device="cuda", dtype=torch.float32, bs=16, prior_frames=100):
    """Transcribe `utts` with greedy CTC; returns one string each."""
    blank = ctc_head.out_features - 1
    hyps = []
    for i in range(0, len(utts), bs):
        batch = utts[i: i + bs]
        enc = encode_batch(processor.feature_extractor, encoder, [u.audio for u in batch], norm_mode, stats,
                           lookahead, device, dtype, prior_frames=prior_frames)
        ids = ctc_head(enc.last_hidden_state).argmax(-1)
        lens = enc.attention_mask.sum(-1)
        for row, n in zip(ids.tolist(), lens.tolist()):
            # CTC rule: collapse repeated labels, then drop blanks.
            toks, prev = [], None
            for t in row[:n]:
                if t != prev and t != blank:
                    toks.append(t)
                prev = t
            hyps.append(processor.tokenizer.decode(toks, skip_special_tokens=True))
    return hyps

# =====================================================================================================================
# 5. COHERE STREAMING: the user-facing class (see the module docstring for the full picture)
# =====================================================================================================================


def _common_prefix(a: list[int], b: list[int]) -> int:
    """Length of the longest shared prefix of two token lists."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class CohereStreaming:
    """Streaming Cohere Transcribe. See the module docstring for the full picture.

    Main arguments: `checkpoint` (Hub id or local folder of the streaming weights), `lookahead` R (live latency
    is (R + 1) * 80 ms; trained values 0, 1, 3, 6, 13), `partials` ("aed" or "ctc"), `final` ("free" or
    "forced"). The rest tune LocalAgreement and the endpoint decode.
    """

    def __init__(self, checkpoint: str | None = DEFAULT_CHECKPOINT, lookahead: int = 6, device: str | None = None, dtype=torch.bfloat16,
                 stats: dict | None = None, left_context: int = 70, conv_init: str = "truncate",
                 partials: str = "aed", aed_interval_ms: int = 320, show_tentative: bool = True,
                 holdback_words: int = 1, min_commit_ms: int = 1000, final: str = "free", prior_frames: int | None = None,
                 final_lookahead: int | None = 13, final_beams: int = 4, suppress_brackets: bool = True,
                 final_reencode_max_s: float = 30.0):
        if partials not in ("aed", "ctc"):
            raise ValueError(partials)
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(COHERE_MODEL_ID)
        model = CohereAsrForConditionalGeneration.from_pretrained(COHERE_MODEL_ID, dtype=torch.float32)
        state_dict_path = None
        ck_prior = None
        norm_mode = "causal"
        if checkpoint is not None:
            path = Path(checkpoint) if Path(checkpoint).exists() else Path(snapshot_download(checkpoint))
            ck_cfg = json.loads((path / "config.json").read_text())
            left_context = ck_cfg["encoder"]["sliding_window"] - 1
            conv_init = ck_cfg["args"].get("conv_init", conv_init)
            ck_prior = ck_cfg["args"].get("eval_prior")  # the warm-up the checkpoint was evaluated with
            norm_mode = ck_cfg["args"].get("norm", "causal")
            if stats is None and (path / "mel_stats.json").exists():
                stats = json.loads((path / "mel_stats.json").read_text())
            state_dict_path = path / "student.safetensors"
        self.encoder = build_streaming_encoder(model.model.encoder, model.config, conv_init=conv_init,
                                               left_context=left_context, lookahead=lookahead)
        self.ctc_head = torch.nn.Linear(self.encoder.config.hidden_size, model.config.vocab_size + 1)
        if state_dict_path is not None:
            sd = load_file(str(state_dict_path))
            self.encoder.load_state_dict({k[len("encoder."):]: v.float() for k, v in sd.items() if k.startswith("encoder.")})
            self.ctc_head.load_state_dict({k[len("ctc_head."):]: v.float() for k, v in sd.items() if k.startswith("ctc_head.")})
            missing, unexpected = model.load_state_dict(
                {k[len("cohere."):]: v.float() for k, v in sd.items() if k.startswith("cohere.")}, strict=False)
            if unexpected or any(not k.startswith("model.encoder.") for k in missing):
                raise RuntimeError(f"decoder checkpoint mismatch: {unexpected[:5]} {missing[:5]}")
        del model.model.encoder  # the offline teacher encoder is not needed at inference
        model.model.encoder = self.encoder
        self.model = model.to(self.device, dtype).eval()
        self.encoder = self.model.model.encoder
        self.ctc_head = self.ctc_head.to(self.device, dtype).eval()
        if stats is None:
            raise ValueError("causal normalisation needs mel statistics (shipped with the checkpoint as mel_stats.json)")
        self.stats = stats
        self.lookahead = lookahead
        self.partials, self.show_tentative = partials, show_tentative
        self.holdback_words, self.min_commit_ms = holdback_words, min_commit_ms
        if final not in ("forced", "free"):
            raise ValueError(final)
        self.final = final  # "free": full-context final decode (best WER), committed words may change at the endpoint
        # endpoint re-encode: with the whole utterance buffered, one full-sequence encoder pass at a longer lookahead
        # (== chunked streaming at that R; trained R in {0,1,3,6,13}) replaces the R=`lookahead` streamed states.
        # Only for a free final over AED partials (the forced prefix and CTC ids come from the streamed states).
        reencode = final_lookahead is not None and final == "free" and partials == "aed"
        self.final_lookahead = final_lookahead if reencode else None
        # one full-sequence pass is O(T) encoder work and O(T^2) attention mask: cap it, and the buffer, for long streams
        self._reencode_max_frames = int(final_reencode_max_s * SR / self.processor.feature_extractor.hop_length)
        self.final_beams = final_beams  # beam search for the one endpoint decode only (partials stay greedy)
        chunk_ms = (lookahead + 1) * 80
        self.aed_every = max(1, math.ceil(aed_interval_ms / chunk_ms))  # chunks between partial AED decodes
        self.hop_ms = 80
        tag = "" if partials == "aed" and show_tentative else f"-{partials}{'' if show_tentative else '-committed'}"
        tag += "-freefinal" if final == "free" else ""
        tag += f"-reenc{self.final_lookahead}" if self.final_lookahead is not None else ""
        tag += f"-beam{final_beams}" if final_beams > 1 else ""
        tag += "-nobrackets" if suppress_brackets else ""
        self.name = f"cohere-streaming-2b{tag}@{chunk_ms}ms"
        tok = self.processor.tokenizer
        self._special = set(tok.all_special_ids)
        # brackets: the decoder sometimes wraps spoken asides in (...), which Whisper-style normalisers (and many
        # downstream consumers) treat as non-speech and drop; suppressing bracket pieces keeps those words as speech
        self._suppress = [i for t, i in tok.get_vocab().items() if any(c in t for c in "()[]")] if suppress_brackets else None
        self._word_start = {i for t, i in tok.get_vocab().items() if t.startswith("▁")}  # SentencePiece word starts
        self._first, self._per = 1 + 8 * lookahead, 8 * (lookahead + 1)
        self.prior_frames = prior_frames if prior_frames is not None else (ck_prior if ck_prior is not None else 100)
        self.featurizer = StreamingFeaturizer(self.processor.feature_extractor, stats, prior_frames=self.prior_frames,
                                             norm_mode=norm_mode)

    # --- public API: reset / step / finish ---------------------------------------------------------------------
    def reset(self, language: str = "en") -> None:
        """Forget everything and get ready for a new utterance in `language`."""
        self.language = language
        self.featurizer.reset()
        self._pending = torch.zeros(0, self.featurizer.fe.feature_size)  # mel frames not yet fed to the encoder
        # every normalised mel frame so far, for the endpoint re-encode; None when it is off or the stream got too long
        self._mel_all: list[torch.Tensor] | None = [] if self.final_lookahead is not None else None
        self._mel_n = 0  # mel frames stored in _mel_all
        self._mel_used = 0  # mel frames already consumed by the streaming encoder
        self._pkv = self._pc = None
        self._states: list[torch.Tensor] = []
        self._chunks = 0
        self._ctc_ids: list[int] = []
        self._prev = None  # last CTC token, to collapse repeats across chunks
        self._committed: list[int] = []  # tokens that will never change again
        self._last_hyp: list[int] = []  # previous decode, for LocalAgreement
        self._shown: list[int] = []  # tokens currently displayed
        self._n_samples = 0
        self._prompt = self.processor.get_decoder_prompt_ids(language=language, punctuation=True)

    # --- encoder ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _encode(self, chunk: torch.Tensor):
        """Run one mel chunk through the streaming encoder; its caches carry over to the next chunk."""
        out = self.encoder(chunk.unsqueeze(0).to(self.device, self.dtype), past_key_values=self._pkv,
                           padding_cache=self._pc, use_cache=True, num_lookahead_tokens=self.lookahead)
        self._pkv, self._pc = out.past_key_values, out.padding_cache
        h = out.last_hidden_state
        self._states.append(h)
        self._chunks += 1
        if self.partials == "ctc":
            blank = self.ctc_head.out_features - 1
            for t in self.ctc_head(h).argmax(-1)[0].tolist():  # greedy CTC, carried across chunks
                if t != self._prev and t != blank:
                    self._ctc_ids.append(t)
                self._prev = t

    def _next_size(self) -> int:
        """Mel frames the next encoder chunk needs: the first chunk is 1 + 8R, later ones 8(R + 1)."""
        return self._first if self._mel_used == 0 else self._per

    def _feed(self, frames: torch.Tensor) -> int:
        """Queue new mel frames, encode every complete chunk, and return how many chunks were encoded."""
        if self._mel_all is not None:
            self._mel_all.append(frames)
            self._mel_n += frames.shape[0]
            if self._mel_n > self._reencode_max_frames:
                self._mel_all = None  # too long to re-encode at the endpoint: fall back to the streamed states
        self._pending = torch.cat([self._pending, frames])
        n = 0
        while self._pending.shape[0] >= (size := self._next_size()):
            self._encode(self._pending[:size])
            self._pending = self._pending[size:]
            self._mel_used += size
            n += 1
        return n

    # --- decoder ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _aed(self, states: torch.Tensor, max_new: int, num_beams: int = 1) -> list[int]:
        """Decode text from encoder `states`, forced to start with the committed tokens.

        Greedy, or beam search if `num_beams` > 1. Returns the text tokens (committed prefix + new ones)."""
        enc = BaseModelOutput(last_hidden_state=states)
        enc["attention_mask"] = torch.ones(states.shape[:2], dtype=torch.long, device=states.device)
        ids = torch.tensor([self._prompt + self._committed], device=self.device)
        out = self.model.generate(encoder_outputs=enc, decoder_input_ids=ids, max_new_tokens=max_new, num_beams=num_beams,
                                  suppress_tokens=self._suppress)
        return [t for t in out[0, len(self._prompt):].tolist() if t not in self._special]

    def _hold_back(self, ids: list[int], words: int) -> list[int]:
        """Drop the last `words` words (tokens from the last word-start onwards): the right-edge words are the ones
        still cut by the audio boundary, where two short-context decodes can agree on a wrong guess."""
        for _ in range(words):
            starts = [j for j, t in enumerate(ids) if t in self._word_start]
            if not starts:
                return []
            ids = ids[: starts[-1]]
        return ids

    def _partial_update(self):
        """LocalAgreement step: decode everything so far, commit what two consecutive decodes agree on."""
        states = torch.cat(self._states, dim=1)
        hyp = self._aed(states, max_new=24)
        if self._n_samples * 1000 / SR >= self.min_commit_ms:  # too little audio: decodes are unreliable
            agreed = hyp[: _common_prefix(hyp, self._last_hyp)]
            agreed = self._hold_back(agreed, self.holdback_words)
            if len(agreed) > len(self._committed):
                self._committed = agreed
        self._last_hyp = hyp
        self._shown = hyp if self.show_tentative else self._committed

    def _text(self, ids: list[int]) -> str:
        """Token ids to a string."""
        return self.processor.tokenizer.decode(ids, skip_special_tokens=True)

    def step(self, audio: np.ndarray) -> str:
        """Feed one hop (80 ms, 16 kHz float32) of audio and return the transcript so far."""
        self._n_samples += len(audio)
        before = self._chunks
        self._feed(self.featurizer.push(audio))
        if self.partials == "ctc":
            return self._text(self._ctc_ids)
        # decode only when the encoder has produced new output and we crossed an `aed_every`-chunk boundary
        if self._chunks > before and self._chunks // self.aed_every > before // self.aed_every:
            self._partial_update()
        return self._text(self._shown)

    @torch.no_grad()
    def finish(self) -> str:
        """The audio ended: return the final transcript (re-encode + beam search, or the streamed states)."""
        tail = self.featurizer.flush()
        max_new = min(256, int(10 + 8 * self._n_samples / SR))
        if self._mel_all is not None and self._mel_n + tail.shape[0] <= self._reencode_max_frames:
            # same causally normalised frames the stream saw, no zero padding: one pass at the longer lookahead
            feats = torch.cat(self._mel_all + [tail])
            if feats.shape[0] == 0:
                return ""
            states = self.encoder(feats.unsqueeze(0).to(self.device, self.dtype),
                                  num_lookahead_tokens=self.final_lookahead).last_hidden_state
            if states.shape[1] == 0:
                return ""
            self._committed = []
            return self._text(self._aed(states, max_new=max_new, num_beams=self.final_beams))
        # fallback: finish the streamed encoder run (also used when the utterance was too long to re-encode)
        rest = torch.cat([self._pending, tail])
        total = self._mel_used + rest.shape[0]
        self._pending = torch.zeros(0, rest.shape[1])
        while rest.shape[0]:  # zero-pad the last partial chunk
            size = self._next_size()
            chunk, rest = rest[:size], rest[size:]
            if chunk.shape[0] < size:
                chunk = torch.nn.functional.pad(chunk, (0, 0, 0, size - chunk.shape[0]))
            self._encode(chunk)
            self._mel_used += size
        if not self._states:
            return ""
        n_valid = int(self.encoder._get_subsampling_output_length(torch.tensor([total]))[0])
        states = torch.cat(self._states, dim=1)[:, :n_valid]
        if self.partials == "ctc" or self.final == "free":
            self._committed = []  # plain second pass (no prefix forcing)
        final = self._aed(states, max_new=max_new, num_beams=self.final_beams)
        return self._text(final)
