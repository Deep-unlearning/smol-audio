# /// script
# requires-python = ">=3.10"
# dependencies = [
#     # Audio SFT support from https://github.com/huggingface/trl/pull/5830. We pin a fork that
#     # includes that branch plus a one-line fix (stale `self._is_vision_dataset` in
#     # `_reject_skip_prepare_without_labels`). Switch to `trl>=...` once the PR is merged.
#     "trl @ git+https://github.com/Deep-unlearning/trl.git@audio-fix",
#     "transformers>=4.57.0",
#     "datasets>=3.0.0",
#     "accelerate>=1.0.0",
#     "peft>=0.13.0",
#     "soundfile",
#     "librosa",
#     "numpy",
# ]
# ///
"""Teach Qwen2-Audio to *recognize sounds* with TRL's `SFTTrainer` (audio SFT support).

Rather than plain ASR (transcribe speech -> text), this fine-tunes Qwen2-Audio on
**environmental sound classification** with ESC-50: the model hears a 5 s clip and answers
with the sound category (`dog`, `sea waves`, `chainsaw`, `crying baby`, ...). It's a nicer
showcase of an audio-*language* model — it has to understand *what* it hears, not transcribe
words — and it comes with a clear, verifiable answer, so we can measure accuracy afterwards.

Audio SFT support comes from https://github.com/huggingface/trl/pull/5830, which teaches
`SFTTrainer` (via `DataCollatorForMultimodalLanguageModeling`) to accept an `audio` column
next to `messages`. Any audio-language model with an `AutoProcessor` works.

Dataset format expected by TRL
------------------------------
Each example is a conversation plus its audio (a raw `np.ndarray`, a `datasets.Audio` object,
or the `{"array", "sampling_rate"}` dict we build below)::

    {
        "audio": {"array": np.ndarray, "sampling_rate": 16000},
        "messages": [
            {"role": "user", "content": "What is the main sound in this clip?"},
            {"role": "assistant", "content": "sea waves"},
        ],
    }

You don't add an audio placeholder to the text: TRL injects `{"type": "audio"}` into the
first user turn, and the collator turns the waveform into `input_features` on the fly.

Quick start
-----------
Local smoke test (few samples, one epoch)::

    uv run finetune_qwen2_audio_trl.py --max_samples 64 --num_train_epochs 1 --eval_samples 16

Full run on Hugging Face Jobs (LoRA fits on a single 48 GB L40S / 24 GB A10G)::

    hf jobs uv run --flavor l40sx1 --secrets HF_TOKEN finetune_qwen2_audio_trl.py \
        --num_train_epochs 3 --push_to_hub --hub_model_id <your-user>/qwen2-audio-esc50

See the bottom of this file for more `hf jobs` recipes.
"""

from __future__ import annotations

import io
import types
from dataclasses import dataclass, field

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Features, Sequence, Value, load_dataset
from peft import LoraConfig
from transformers import AutoProcessor, HfArgumentParser, Qwen2AudioForConditionalGeneration

from trl import SFTConfig, SFTTrainer


# Qwen2-Audio's feature extractor (Whisper-style log-mel) expects 16 kHz mono audio.
TARGET_SR = 16_000

# Store the decoded waveform as a plain struct (NOT a `datasets.Audio` feature). datasets 4.x
# requires torchcodec to *encode* an Audio column, which we deliberately avoid; TRL's collator
# happily accepts this `{"array", "sampling_rate"}` dict.
AUDIO_FEATURE = {"array": Sequence(Value("float32")), "sampling_rate": Value("int64")}
MESSAGES_FEATURE = [{"role": Value("string"), "content": Value("string")}]


@dataclass
class ScriptArguments:
    """Command-line arguments for the fine-tuning script."""

    model_id: str = field(
        default="Qwen/Qwen2-Audio-7B-Instruct",
        metadata={"help": "Audio-language model to fine-tune. Any Qwen2-Audio / Voxtral checkpoint works."},
    )
    dataset_id: str = field(
        default="ashraq/esc50",
        metadata={"help": "Hub dataset with an audio column plus either a `messages` column or a target/label column."},
    )
    dataset_config: str = field(
        default="default",
        metadata={"help": "Dataset config name. Pass an empty string for datasets without configs."},
    )
    dataset_split: str = field(default="train", metadata={"help": "Split to load (we make our own train/eval split)."})
    audio_column: str = field(default="audio", metadata={"help": "Name of the audio column."})
    target_column: str = field(
        default="category",
        metadata={"help": "Column holding the answer (e.g. ESC-50 `category`, or a transcript for ASR). "
                          "Used only when the dataset is not already conversational. Underscores become spaces."},
    )
    instruction: str = field(
        default="You are an expert at recognizing environmental sounds. Listen to the clip and name the sound "
                "in a few words.",
        metadata={"help": "User instruction paired with each clip (ignored if the dataset has a `messages` column)."},
    )
    test_size: float = field(default=0.1, metadata={"help": "Fraction held out for evaluation."})
    max_samples: int | None = field(
        default=None, metadata={"help": "Optionally cap the number of samples (before splitting) for quick tests."}
    )
    use_lora: bool = field(default=True, metadata={"help": "Train a LoRA adapter (default) or full fine-tune."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Number of training epochs."})
    eval_samples: int = field(default=64, metadata={"help": "How many held-out clips to score after training (0=skip)."})
    push_to_hub: bool = field(default=False, metadata={"help": "Push the trained model to the Hub."})
    hub_model_id: str | None = field(default=None, metadata={"help": "Target repo id when pushing to the Hub."})


def build_dataset(args: ScriptArguments):
    """Load a dataset and shape it into TRL's conversational + audio format.

    We decode the audio ourselves with `soundfile` (+`librosa` for resampling) and hand TRL an
    `{"array", "sampling_rate"}` dict. This keeps the example dependency-light: `soundfile`
    decodes wav/flac/ogg natively, so no system `ffmpeg`/`torchcodec` is required at runtime.
    """
    config = args.dataset_config or None
    dataset = load_dataset(args.dataset_id, config, split=args.dataset_split)
    if args.max_samples is not None:
        dataset = dataset.shuffle(seed=42).select(range(min(args.max_samples, len(dataset))))

    # `decode=False` yields the raw {"bytes"/"path"} without triggering datasets' decoder.
    dataset = dataset.cast_column(args.audio_column, Audio(decode=False))
    already_conversational = "messages" in dataset.column_names

    # Force the output schema so the `audio` column is a plain struct rather than inheriting the
    # source `Audio` feature (which would try to re-encode via torchcodec on write).
    if already_conversational:
        out_features = Features({"messages": dataset.features["messages"], "audio": AUDIO_FEATURE})
        keep = ["messages"]
    else:
        out_features = Features({"messages": MESSAGES_FEATURE, "audio": AUDIO_FEATURE})
        keep = []

    def prepare(example):
        audio = example[args.audio_column]
        raw_bytes = audio.get("bytes")
        if raw_bytes is not None:
            wav, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        else:
            wav, sr = sf.read(audio["path"], dtype="float32")
        if wav.ndim > 1:  # downmix stereo to mono
            wav = wav.mean(axis=1)
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)

        out = {"audio": {"array": wav.astype(np.float32), "sampling_rate": TARGET_SR}}
        if not already_conversational:
            target = str(example[args.target_column]).replace("_", " ")  # "sea_waves" -> "sea waves"
            # Plain string content: TRL adds the audio placeholder to the first user turn for us.
            out["messages"] = [
                {"role": "user", "content": args.instruction},
                {"role": "assistant", "content": target},
            ]
        return out

    remove = [c for c in dataset.column_names if c not in keep]
    return dataset.map(prepare, remove_columns=remove, features=out_features, desc="Building audio SFT dataset")


@torch.no_grad()
def evaluate(model, processor, eval_dataset, max_new_tokens: int = 24) -> None:
    """Generate a prediction per clip and report how often the true label appears in the output."""
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    correct, shown = 0, []
    for example in eval_dataset:
        instruction = example["messages"][0]["content"]
        true_label = example["messages"][1]["content"]
        audio = np.asarray(example["audio"]["array"], dtype=np.float32)

        conversation = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": instruction}]}]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audio=[audio], sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        inputs = {k: (v.to(model.dtype) if v.is_floating_point() else v).to(model.device) for k, v in inputs.items()}

        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prediction = processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()

        if true_label.lower() in prediction.lower():
            correct += 1
        if len(shown) < 10:
            shown.append((true_label, prediction))

    n = len(eval_dataset)
    print(f"\n=== Post-training evaluation on {n} held-out clips ===")
    print(f"accuracy (true label found in prediction): {correct}/{n} = {correct / max(n, 1):.1%}\n")
    print(f"{'true label':<22} | prediction")
    print("-" * 60)
    for true_label, prediction in shown:
        print(f"{true_label:<22} | {prediction}")


def main():
    args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]

    dataset = build_dataset(args)
    split = dataset.train_test_split(test_size=args.test_size, seed=42)
    train_dataset, eval_dataset = split["train"], split["test"]
    if args.eval_samples:
        eval_dataset = eval_dataset.select(range(min(args.eval_samples, len(eval_dataset))))
    print(f"train: {len(train_dataset)} | eval: {len(eval_dataset)}")

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",  # swap for "flash_attention_2" if installed
    )

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            # Attention + MLP projections. peft matches these names across the language model
            # (and the audio tower's attention), leaving the audio feature extractor frozen.
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    training_args = SFTConfig(
        output_dir="qwen2-audio-esc50",
        per_device_train_batch_size=2,  # audio features are large; keep the per-device batch small
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=1e-4 if args.use_lora else 1e-5,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        # CRITICAL for audio: truncation can cut the audio placeholder tokens and crash the
        # processor, so disable it. Batches are padded to the longest sequence instead.
        max_length=None,
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        # Note: packing / padding_free / assistant_only_loss are not supported for multimodal SFT.
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub()

    if args.eval_samples:
        # Prepare a clean model for generation:
        #  1. Merge the LoRA adapter back so we generate with a plain Qwen2-Audio model.
        #  2. Rebind the class's `forward`: Trainer/accelerate replaces the *instance* forward with
        #     a mixed-precision wrapper whose generic `(*args, **kwargs)` signature hides
        #     `input_features`, which makes `generate`'s kwargs-validation reject the audio inputs.
        gen_model = trainer.model
        if hasattr(gen_model, "merge_and_unload"):
            gen_model = gen_model.merge_and_unload()
        gen_model.forward = types.MethodType(type(gen_model).forward, gen_model)
        try:
            evaluate(gen_model, processor, eval_dataset)
        except Exception as exc:  # eval is a bonus; never fail the job after training + save
            import traceback

            traceback.print_exc()
            print(f"[warn] evaluation skipped: {exc}")


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Running on Hugging Face Jobs (https://huggingface.co/docs/huggingface_hub/guides/cli)
# -----------------------------------------------------------------------------
# `hf jobs uv run` reads the inline `# /// script` dependencies above, so no extra setup is
# needed. `--secrets HF_TOKEN` forwards your local token (for model download + push).
#
#   # Cheap smoke test on the tiny test model (verifies the whole pipeline in ~1 min):
#   hf jobs uv run --flavor t4-small --secrets HF_TOKEN finetune_qwen2_audio_trl.py \
#       --model_id trl-internal-testing/tiny-Qwen2AudioForConditionalGeneration \
#       --max_samples 32 --num_train_epochs 1 --eval_samples 0
#
#   # Full LoRA fine-tune of Qwen2-Audio-7B-Instruct on ESC-50 (single 48 GB L40S or 24 GB A10G):
#   hf jobs uv run --flavor l40sx1 --secrets HF_TOKEN finetune_qwen2_audio_trl.py \
#       --num_train_epochs 3 --push_to_hub --hub_model_id <your-user>/qwen2-audio-esc50
#
#   # Full fine-tune instead of LoRA (needs more VRAM, e.g. an 80 GB A100):
#   hf jobs uv run --flavor a100-large --secrets HF_TOKEN finetune_qwen2_audio_trl.py \
#       --use_lora False --num_train_epochs 3
#
# Monitor a detached run (`-d`) with `hf jobs logs <job_id>` / `hf jobs ps`.
