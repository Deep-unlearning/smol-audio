"""Deploy tiny real checkpoints through the inference constructor without Hub downloads."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import (CohereAsrConfig, CohereAsrFeatureExtractor, CohereAsrForConditionalGeneration,
                          ParakeetEncoderConfig)

import model as asr_module
from model import CohereStreaming, build_streaming_encoder, log_mel, normalize


@pytest.mark.parametrize("norm_mode,eval_prior", [("global", 30), ("causal", 0)])
def test_checkpoint_normalisation_matches_training_features(tmp_path, monkeypatch, norm_mode, eval_prior):
    """A deployed checkpoint feeds its saved normalisation and warm-up into the actual streaming encoder."""
    torch.manual_seed(7)
    encoder_config = ParakeetEncoderConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64,
        num_mel_bins=128, subsampling_conv_channels=8, scale_input=False,
        dropout=0.0, dropout_positions=0.0, layerdrop=0.0, activation_dropout=0.0, attention_dropout=0.0,
    )
    config = CohereAsrConfig(
        encoder_config=encoder_config.to_dict() | {"model_type": "parakeet_encoder"},
        vocab_size=16, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        decoder_start_token_id=4,
    )
    base = CohereAsrForConditionalGeneration(config).eval()
    encoder = build_streaming_encoder(base.model.encoder, config, conv_init="truncate", left_context=14, lookahead=1)
    ctc_head = torch.nn.Linear(32, 17)
    state = {"encoder." + k: v.detach().clone() for k, v in encoder.state_dict().items()}
    state.update({"ctc_head." + k: v.detach().clone() for k, v in ctc_head.state_dict().items()})
    state.update({"cohere." + k: v.detach().clone() for k, v in base.state_dict().items()
                  if not k.startswith("model.encoder.")})
    save_file(state, tmp_path / "student.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({
        "encoder": encoder.config.to_dict(),
        "args": {"conv_init": "truncate", "norm": norm_mode, "eval_prior": eval_prior},
    }))
    stats = {"mean": [-10.0] * 128, "std": [3.0] * 128}
    (tmp_path / "mel_stats.json").write_text(json.dumps(stats))

    fe = CohereAsrFeatureExtractor(dither=0.0)
    tokenizer = SimpleNamespace(all_special_ids=[1, 2, 3, 4], get_vocab=lambda: {"▁hello": 5},
                                decode=lambda *a, **kw: "")
    processor = SimpleNamespace(feature_extractor=fe, tokenizer=tokenizer,
                                get_decoder_prompt_ids=lambda **kw: [4])
    monkeypatch.setattr(asr_module.AutoProcessor, "from_pretrained", lambda *a, **kw: processor)
    monkeypatch.setattr(asr_module.CohereAsrForConditionalGeneration, "from_pretrained",
                        lambda *a, **kw: copy.deepcopy(base))
    deployed = CohereStreaming(str(tmp_path), lookahead=1, device="cpu", dtype=torch.float32,
                               partials="ctc", final_lookahead=None)
    assert deployed.prior_frames == eval_prior
    assert torch.equal(deployed.ctc_head.weight, ctc_head.weight)
    assert torch.equal(deployed.encoder.layers[0].conv.norm.weight, encoder.layers[0].conv.norm.weight)

    # Observe actual encoder inputs, including the padded final chunk, rather than only inspecting attributes.
    seen = []
    deployed.encoder.register_forward_pre_hook(lambda _module, args: seen.append(args[0].detach().clone()))
    monkeypatch.setattr(deployed, "_aed", lambda *a, **kw: [])
    audio = (0.1 * np.random.default_rng(3).standard_normal(3000)).astype(np.float32)
    deployed.reset()
    for start in range(0, len(audio), 1280):
        deployed.step(audio[start:start + 1280])
    deployed.finish()

    features, mask = log_mel(fe, [torch.from_numpy(audio)])
    expected = normalize(features.double(), mask, norm_mode, stats, prior_frames=eval_prior)[0, mask[0]].float()
    actual = torch.cat(seen, dim=1)[0, :len(expected)]
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
