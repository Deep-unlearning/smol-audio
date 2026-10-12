"""Streaming features must match the batch causal-normalised features frame for frame."""

import numpy as np
import pytest
import torch
from transformers import CohereAsrFeatureExtractor

from model import StreamingFeaturizer, log_mel, normalize

STATS = {"mean": [-10.0] * 128, "std": [3.0] * 128}


@pytest.fixture(scope="module")
def fe():
    fe = CohereAsrFeatureExtractor()
    fe.dither = 0.0  # exact comparison
    return fe


@pytest.mark.parametrize("n_samples,hop", [(16000 * 3 + 123, 1280), (8000, 160), (40000, 4999)])
@pytest.mark.parametrize("norm_mode", ["causal", "global"])
def test_streaming_featurizer_matches_batch(fe, n_samples, hop, norm_mode):
    """Chunked streaming features equal the batch causal-normalised features for any chunk size."""
    torch.manual_seed(0)
    audio = (0.1 * torch.randn(n_samples)).numpy().astype(np.float32)
    feats, mask = log_mel(fe, [torch.from_numpy(audio)])
    ref = normalize(feats, mask, norm_mode, STATS)[0, : int(mask.sum())]
    sf = StreamingFeaturizer(fe, STATS, norm_mode=norm_mode)
    outs = [sf.push(audio[i: i + hop]) for i in range(0, n_samples, hop)] + [sf.flush()]
    got = torch.cat(outs)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert (got - ref).abs().max() < 1e-4


def test_streaming_featurizer_emits_only_complete_frames(fe):
    """A frame is emitted only once its whole STFT window has arrived."""
    sf = StreamingFeaturizer(fe, STATS)
    assert sf.push(np.zeros(255, dtype=np.float32)).shape[0] == 0  # first window needs n_fft/2 real samples
    got = sf.push(np.zeros(1, dtype=np.float32)).shape[0]
    assert got == 1


def test_streaming_featurizer_releases_consumed_samples(fe):
    """The waveform buffer remains a single STFT window even for a long utterance."""
    sf = StreamingFeaturizer(fe, STATS)
    for _ in range(500):
        sf.push(np.zeros(1280, dtype=np.float32))
        assert sf.y.numel() < fe.n_fft
        assert sf.y.untyped_storage().nbytes() == sf.y.numel() * sf.y.element_size()
    assert sf.n_samples == 500 * 1280
