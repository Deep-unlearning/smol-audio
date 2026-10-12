"""Correctness gates for the Cohere -> cache-aware streaming encoder conversion.

1. Weight mapping: with causal=False the streaming encoder reproduces the offline encoder.
2. Padding invariance of the causal encoder (what batched training relies on).
3. Train/infer consistency: chunk-by-chunk cached streaming == one full-sequence forward with the
   chunked-limited mask (what we train with).
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import ParakeetEncoder, ParakeetEncoderConfig

from model import build_streaming_encoder

ATOL = 1e-4


def tiny_offline_encoder(seed=0):
    torch.manual_seed(seed)
    cfg = ParakeetEncoderConfig(
        hidden_size=64, num_hidden_layers=3, num_attention_heads=4, intermediate_size=128,
        num_mel_bins=128, subsampling_conv_channels=16, scale_input=False,
        dropout=0.0, dropout_positions=0.0, layerdrop=0.0, activation_dropout=0.0, attention_dropout=0.0,
    )
    enc = ParakeetEncoder(cfg).eval()
    with torch.no_grad():  # non-trivial BatchNorm statistics so the fold is actually exercised
        for m in enc.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.running_mean.normal_()
                m.running_var.uniform_(0.5, 2.0)
                m.weight.normal_(1.0, 0.1)
                m.bias.normal_()
    return enc, SimpleNamespace(encoder_config=cfg)


@pytest.mark.parametrize("n_frames", [200, 201, 357])
def test_noncausal_parity(n_frames):
    """With causal=False the converted encoder reproduces the offline encoder (weight mapping is right)."""
    offline, cohere_cfg = tiny_offline_encoder()
    streaming = build_streaming_encoder(offline, cohere_cfg, causal=False)
    x = torch.randn(2, n_frames, 128)
    with torch.no_grad():
        ref = offline(x).last_hidden_state
        out = streaming(x).last_hidden_state
    assert out.shape == ref.shape
    assert (out - ref).abs().max() < ATOL


@pytest.mark.parametrize("n_frames", [200, 357])
def test_noncausal_parity_padded(n_frames):
    """Non-causal parity also holds for padded batches, including the output frame mask."""
    offline, cohere_cfg = tiny_offline_encoder()
    streaming = build_streaming_encoder(offline, cohere_cfg, causal=False)
    x = torch.randn(2, n_frames, 128)
    mask = torch.ones(2, n_frames, dtype=torch.long)
    mask[1, n_frames - 37:] = 0
    with torch.no_grad():
        ref = offline(x, attention_mask=mask)
        out = streaming(x, attention_mask=mask)
    valid = ref.attention_mask.bool()
    assert torch.equal(out.attention_mask.bool(), valid)
    assert (out.last_hidden_state - ref.last_hidden_state).abs()[valid].max() < ATOL


def test_causal_padding_invariance():
    """Padding a sequence in a batch does not change the causal encoder output for its real frames."""
    offline, cohere_cfg = tiny_offline_encoder()
    enc = build_streaming_encoder(offline, cohere_cfg, left_context=14, lookahead=6)
    x = torch.randn(2, 300, 128)
    mask = torch.ones(2, 300, dtype=torch.long)
    mask[1, 231:] = 0
    with torch.no_grad():
        padded = enc(x, attention_mask=mask, num_lookahead_tokens=6)
        alone = enc(x[1:, :231], num_lookahead_tokens=6).last_hidden_state
    n = alone.shape[1]
    assert padded.attention_mask[1].sum() == n
    assert (padded.last_hidden_state[1, :n] - alone[0]).abs().max() < ATOL


@pytest.mark.parametrize("lookahead", [0, 1, 6])
def test_streaming_matches_masked_full_forward(lookahead):
    """Chunk-by-chunk cached streaming equals one full forward with the chunked attention mask (train/infer match)."""
    offline, cohere_cfg = tiny_offline_encoder()
    enc = build_streaming_encoder(offline, cohere_cfg, left_context=14, lookahead=lookahead)
    first, per = 1 + 8 * lookahead, 8 * (lookahead + 1)
    n_chunks = 6
    x = torch.randn(1, first + per * (n_chunks - 1), 128)
    with torch.no_grad():
        full = enc(x, num_lookahead_tokens=lookahead).last_hidden_state
        out, pkv, pc, t = [], None, None, 0
        for i in range(n_chunks):
            size = first if i == 0 else per
            o = enc(x[:, t:t + size], past_key_values=pkv, padding_cache=pc, use_cache=True,
                    num_lookahead_tokens=lookahead)
            pkv, pc, t = o.past_key_values, o.padding_cache, t + size
            out.append(o.last_hidden_state)
        streamed = torch.cat(out, dim=1)
    assert streamed.shape == full.shape
    assert (streamed - full).abs().max() < ATOL


def test_truncate_init_is_aligned():
    """A delta at time t through a truncated causal conv must land at t (not t - c) and see no future."""
    from model import _truncate_to_causal

    k, c = 9, 4
    w = torch.arange(1.0, k + 1).view(1, 1, k)  # centred kernel, tap for offset o at index o + c
    wt = _truncate_to_causal(w, time_dim=2)
    assert torch.equal(wt[0, 0, : k - 1 - c], torch.zeros(k - 1 - c))
    x = torch.zeros(1, 1, 20)
    x[0, 0, 10] = 1.0
    y_off = torch.nn.functional.conv1d(torch.nn.functional.pad(x, (c, c)), w)
    y_cau = torch.nn.functional.conv1d(torch.nn.functional.pad(x, (k - 1, 0)), wt)
    # causal output keeps the offline response for t >= 10 (past + centre taps) and nothing before
    assert torch.allclose(y_cau[0, 0, 10:], y_off[0, 0, 10:])
    assert torch.all(y_cau[0, 0, :10] == 0)


def test_truncate_encoder_streams_consistently():
    """An encoder converted with conv_init="truncate" still streams identically to its full forward."""
    offline, cohere_cfg = tiny_offline_encoder()
    enc = build_streaming_encoder(offline, cohere_cfg, conv_init="truncate", left_context=14, lookahead=1)
    first, per = 1 + 8, 16
    x = torch.randn(1, first + per * 4, 128)
    with torch.no_grad():
        full = enc(x, num_lookahead_tokens=1).last_hidden_state
        out, pkv, pc, t = [], None, None, 0
        for i in range(5):
            size = first if i == 0 else per
            o = enc(x[:, t:t + size], past_key_values=pkv, padding_cache=pc, use_cache=True, num_lookahead_tokens=1)
            pkv, pc, t = o.past_key_values, o.padding_cache, t + size
            out.append(o.last_hidden_state)
    assert (torch.cat(out, 1) - full).abs().max() < ATOL
