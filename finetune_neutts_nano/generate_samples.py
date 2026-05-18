"""Generate voice-cloned samples from a fine-tuned NeuTTS Nano checkpoint.

Pairs with `finetune_neutts_nano.py`: uses the same phonemizer setup,
chat-template grammar, and (optional) language token, so the inference prompt
exactly matches the training distribution.

Example:
    python generate_samples.py \
        --model-checkpoint ./runs/yodas-150k-official-trainer-it \
        --encoded-dataset Steveeeeeeen/yodas-granary-it-neucodec-150k \
        --reference-index 0 \
        --phonemizer-lang it --language-token "" \
        --prompts "Ciao mondo." "Oggi piove molto." \
        --output-dir ./samples
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import phonemizer
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

NEUCODEC_SAMPLE_RATE = 24_000
SPEECH_TOKEN_RE = re.compile(r"<\|speech_(\d+)\|>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-checkpoint", required=True, help="Path or Hub id of the fine-tuned model.")
    parser.add_argument("--codec-checkpoint", default="neuphonic/neucodec")
    parser.add_argument("--encoded-dataset", required=True, help="Dataset with `text` + `codes` to source the reference voice.")
    parser.add_argument("--reference-split", default="train")
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--phonemizer-lang", default="it", help="eSpeak language code (matches training config).")
    parser.add_argument("--language-token", default="", help='Special token to prepend, e.g. "<|IT|>". Empty if model was not trained with one.')
    parser.add_argument("--prompts", nargs="+", required=True, help="One or more target texts to synthesize.")
    parser.add_argument("--output-dir", default="./samples")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--min-new-tokens", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    return parser.parse_args()


def build_phonemizer(language: str):
    return phonemizer.backend.EspeakBackend(
        language=language,
        preserve_punctuation=True,
        with_stress=True,
        words_mismatch="ignore",
        language_switch="remove-flags",
    )


def phonemize(g2p, text: str) -> str:
    phones = g2p.phonemize([text])
    if not phones or not phones[0]:
        raise ValueError(f"Empty phonemization for text={text!r}")
    return " ".join(phones[0].split())


def build_voice_clone_prompt(tokenizer, ref_phonemes: str, target_phonemes: str, ref_codes: list[int], language_token: str) -> torch.Tensor:
    prompt_text = f"{ref_phonemes} {target_phonemes}"
    if language_token:
        prompt_text = f"{language_token} {prompt_text}"
    speech_token_text = "".join(f"<|speech_{idx}|>" for idx in ref_codes)
    template = (
        f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{prompt_text}<|TEXT_PROMPT_END|>\n"
        f"assistant:<|SPEECH_GENERATION_START|>{speech_token_text}"
    )
    ids = tokenizer.encode(template)
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def extract_speech_ids(tokenizer, token_ids: list[int], speech_end_id: int) -> list[int]:
    out: list[int] = []
    for tid in token_ids:
        if tid == speech_end_id:
            break
        piece = tokenizer.convert_ids_to_tokens(int(tid))
        match = SPEECH_TOKEN_RE.match(piece)
        if match:
            out.append(int(match.group(1)))
    return out


def decode_codes_to_wav(codec, codes: list[int], device: torch.device):
    codes_tensor = torch.tensor(codes, dtype=torch.long, device=device).view(1, 1, -1)
    with torch.no_grad():
        return codec.decode_code(codes_tensor).detach().cpu().squeeze().float().numpy()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    print(f"Device: {device}; dtype: {dtype}")

    print(f"Loading reference row {args.reference_split}[{args.reference_index}] from {args.encoded_dataset} ...")
    row = load_dataset(args.encoded_dataset, split=args.reference_split)[args.reference_index]
    ref_text, ref_codes = row["text"], [int(c) for c in row["codes"]]

    print(f"Loading tokenizer + model from {args.model_checkpoint} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint, torch_dtype=dtype).to(device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading NeuCodec ...")
    from neucodec import NeuCodec
    codec = NeuCodec.from_pretrained(args.codec_checkpoint).eval().to(device)

    g2p = build_phonemizer(args.phonemizer_lang)
    ref_phonemes = phonemize(g2p, ref_text)
    speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

    print(f"Reference text: {ref_text!r}")
    print(f"Reference codec tokens: {len(ref_codes)}")

    for idx, target_text in enumerate(args.prompts, start=1):
        torch.manual_seed(args.seed + idx - 1)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + idx - 1)

        target_phonemes = phonemize(g2p, target_text)
        input_ids = build_voice_clone_prompt(
            tokenizer, ref_phonemes, target_phonemes, ref_codes, args.language_token
        ).to(device)

        max_length = min(args.max_seq_length, input_ids.shape[-1] + args.max_new_tokens)
        if max_length <= input_ids.shape[-1]:
            raise ValueError(f"Prompt is {input_ids.shape[-1]} tokens, already over max_seq_length={args.max_seq_length}.")

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                min_new_tokens=args.min_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_k=args.top_k,
                eos_token_id=speech_end_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        new_tokens = output[0, input_ids.shape[-1]:].detach().cpu().tolist()
        speech_ids = extract_speech_ids(tokenizer, new_tokens, speech_end_id)
        if not speech_ids:
            raise RuntimeError(f"Sample {idx} produced no <|speech_*|> tokens.")

        wav = decode_codes_to_wav(codec, speech_ids, device)
        audio_path = output_dir / f"sample_{idx:02d}.wav"
        sf.write(audio_path, wav, NEUCODEC_SAMPLE_RATE)
        print(f"[{idx}] '{target_text}' -> {audio_path} ({len(wav) / NEUCODEC_SAMPLE_RATE:.2f}s, {len(speech_ids)} codec tokens)")


if __name__ == "__main__":
    main()
