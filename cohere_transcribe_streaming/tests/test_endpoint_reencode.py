"""Endpoint logic of the streaming adapter (no model download): the re-encode sees exactly the streamed mel frames,
and long streams drop the buffer and fall back to the streamed states."""

import types

import numpy as np
import torch

from model import CohereStreaming

N_MEL, HOP = 4, 160


class StubFeaturizer:
    fe = types.SimpleNamespace(feature_size=N_MEL)

    def __init__(self):
        self.n = 0

    def reset(self):
        self.n = 0

    def push(self, audio):  # one mel frame per 160 samples, value = frame index
        k = len(audio) // HOP
        out = torch.arange(self.n, self.n + k, dtype=torch.float32)[:, None].repeat(1, N_MEL)
        self.n += k
        return out

    def flush(self):  # two trailing frames, as the real featurizer emits at the end
        out = torch.arange(self.n, self.n + 2, dtype=torch.float32)[:, None].repeat(1, N_MEL)
        self.n += 2
        return out


class StubEncoder:
    def __init__(self):
        self.full_calls = []

    def __call__(self, x, past_key_values=None, padding_cache=None, use_cache=False, num_lookahead_tokens=None):
        if not use_cache:
            self.full_calls.append((x.clone(), num_lookahead_tokens))
        return types.SimpleNamespace(last_hidden_state=torch.zeros(1, max(1, x.shape[1] // 8), 3),
                                     past_key_values=None, padding_cache=None)

    def _get_subsampling_output_length(self, lengths):
        return lengths // 8


def make(max_frames: int, lookahead: int = 1):
    m = object.__new__(CohereStreaming)
    m.device, m.dtype = "cpu", torch.float32
    m.encoder, m.featurizer = StubEncoder(), StubFeaturizer()
    m.final_lookahead, m.final_beams, m.final = 13, 4, "free"
    m.partials, m.show_tentative, m.lookahead = "aed", True, lookahead
    m._reencode_max_frames = max_frames
    m._first, m._per = 1 + 8 * lookahead, 8 * (lookahead + 1)
    m.aed_every = 10**9  # no partial decodes
    m.processor = types.SimpleNamespace(get_decoder_prompt_ids=lambda **k: [])
    m.aed_calls = []
    m._aed = lambda states, max_new, num_beams=1: m.aed_calls.append((states.shape[1], num_beams)) or [1, 2]
    m._text = lambda ids: "ok"
    m.reset()
    return m


def feed(m, n_frames):
    for _ in range(n_frames // 8):
        m.step(np.zeros(8 * HOP, dtype=np.float32))


def test_reencode_sees_all_streamed_frames():
    """The endpoint re-encode receives exactly the streamed mel frames plus the flushed tail, in order."""
    m = make(max_frames=10_000)
    feed(m, 96)
    assert m.finish() == "ok"
    (x, la), = m.encoder.full_calls
    assert la == 13
    assert x.shape[1] == 96 + 2  # every pushed frame + the flushed tail, in order, no padding
    assert torch.equal(x[0, :, 0], torch.arange(98, dtype=torch.float32))
    assert m.aed_calls == [(98 // 8, 4)]


def test_long_stream_drops_buffer_and_uses_streamed_states():
    """Very long streams release the mel buffer and skip the quadratic re-encode, keeping memory bounded."""
    m = make(max_frames=40)
    feed(m, 96)
    assert m._mel_all is None  # bounded memory: the buffer is released once the cap is passed
    assert m.finish() == "ok"
    assert m.encoder.full_calls == []  # no O(T^2) full-sequence pass at the endpoint
    assert m.aed_calls and m.aed_calls[-1][1] == 4  # still the beam final, over the streamed states


def test_tail_pushing_past_cap_falls_back():
    """If only the flushed tail pushes the stream past the cap, the endpoint still falls back safely."""
    m = make(max_frames=97)  # 96 streamed frames fit, the 2 flushed ones do not
    feed(m, 96)
    assert m._mel_all is not None
    m.finish()
    assert m.encoder.full_calls == []
