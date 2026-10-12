---
title: "Streaming speech recognition with Transformers"
authors:
- user: Steveeeeeeen
---

# Streaming speech recognition with Transformers

A live caption should appear while someone is talking. That changes the speech recognition problem: the model must make useful predictions from audio that is still arriving, then revise them as it hears more.

This tutorial follows one concrete implementation: adapting the offline 2B [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) model into a streaming student. We will run a released checkpoint, follow audio through its encoder and decoder, and train a small demonstration with the Transformers stack.

The [smol-audio example](https://github.com/Deep-unlearning/smol-audio/tree/master/cohere_transcribe_streaming) contains the code. You can also follow the [inference notebook](https://github.com/Deep-unlearning/smol-audio/blob/master/cohere_transcribe_streaming/Stream_Cohere_Transcribe.ipynb) and [distillation notebook](https://github.com/Deep-unlearning/smol-audio/blob/master/cohere_transcribe_streaming/Distill_Streaming_Student.ipynb).

Try the [live microphone Space](https://huggingface.co/spaces/Steveeeeeeen/cohere-streaming-demo) first: choose a language, click Record, and speak for up to 25 seconds. Captions appear as audio arrives; Stop flushes the remaining audio and produces a final transcript. The [demo source](https://github.com/Deep-unlearning/smol-audio/tree/master/cohere_transcribe_streaming/space) shows how the browser sends audio chunks to a streaming GPU worker.

## How streaming recognition works

An offline encoder can use the entire recording to interpret every frame. A streaming encoder has to limit which future frames each prediction can see. It also needs to carry context from one chunk to the next.

Here is the path through our implementation:

```text
16 kHz mono audio
        |
        v
log-mel frames -> running normalization
        |
        v
causal convolutions + chunk-limited attention
        |                         ^
        |                         |
        +------------------> cached past context
        |
        v
accumulated encoder states -> decoder -> partial captions
        |
        v
end of utterance -> optional re-encoding -> final transcript
```

Transformers already includes a native cache-aware [Nemotron ASR Streaming encoder](https://huggingface.co/docs/transformers/model_doc/nemotron_asr_streaming). Our Cohere adapter reuses that encoder implementation and maps compatible Cohere weights into it. It retains Cohere's text decoder and adds the conversion, feature normalization, and caption logic needed for this model.

This conversion depends on the model architecture. Setting `chunk_length_s` in an ASR pipeline can split a long file into windows; it does not give an arbitrary offline model causal attention, convolution caches, or streaming training.

## Run a checkpoint first

Use Python 3.10 or newer in a fresh environment. Run commands and Python examples from the example directory:

```bash
git clone https://github.com/Deep-unlearning/smol-audio.git
cd smol-audio/cohere_transcribe_streaming
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf auth login
```

Accept access to the [base Cohere model](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) first. The adapter loads its processor and decoder alongside the streaming checkpoint.

The requirements pin `transformers==5.14.1` because the adapter uses streaming encoder internals. A GPU is recommended for inference. Distillation keeps both the frozen teacher and trainable student in memory; use an A100 80 GB or H100 as the starting point for this 2B recipe.

Place a short recording at `speech.wav`. Load it as mono `float32` and resample it to 16 kHz:

```python
import librosa
import soundfile as sf
import torch

from model import CohereStreaming

audio, sample_rate = sf.read("speech.wav", dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)
if sample_rate != 16000:
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

device = "cuda" if torch.cuda.is_available() else "cpu"
recognizer = CohereStreaming(
    checkpoint="Steveeeeeeen/cohere-streaming",
    lookahead=6,
    device=device,
    dtype=torch.float32,
)
recognizer.reset(language="en")

previous_text = ""
for start in range(0, len(audio), 1280):
    text = recognizer.step(audio[start:start + 1280])
    if text != previous_text:
        print("Partial:", text)
        previous_text = text

print("Final:", recognizer.finish())
```

At 16,000 samples per second, 1,280 samples are 80 ms of audio. The loop feeds those chunks immediately, so this file example runs as quickly as the computer allows. For paced playback, use `python stream_file.py speech.wav --language en --lookahead 6 --realtime`. A microphone application would call `step()` as each chunk arrives.

`reset()` clears the previous utterance. `step()` returns the current transcript, which may be unchanged while the encoder waits for context. `finish()` flushes the remaining audio and produces the endpoint transcript.

## From waveform to features

The waveform is a sequence of amplitudes. A log-mel spectrogram summarizes its frequency content over short overlapping windows. Nearby frames help the model recognize changing sounds, while mel bins group frequencies on a scale useful for speech.

The [Cohere processor](https://huggingface.co/docs/transformers/model_doc/cohere_asr) supplies the feature extractor and tokenizer. In this adapter, `StreamingFeaturizer` keeps the samples needed for the next analysis window and emits newly completed frames. Its state also preserves the previous sample for pre-emphasis and the running normalization statistics.

Normalization needs special care. Offline utterance normalization computes a mean and standard deviation from the whole clip, including future audio. Live features cannot depend on that future. The student therefore uses causal running statistics, initialized from corpus statistics saved in `mel_stats.json`.

The initialization acts as a prior: it stops the first few frames from defining unstable statistics on their own. As more audio arrives, the running estimates adapt. Training uses the same causal normalization for the student; the frozen offline teacher receives utterance-normalized features.

## Attention and convolution with lookahead

Attention lets an encoder frame consult other frames. Here, a chunk can attend to a bounded region of earlier audio and the future frames allowed within its current chunk. After processing, attention keys and values are cached for the next chunk.

The central encoder call passes both caches into the next forward pass. This small function shows the same pattern used by `CohereStreaming._encode()`:

```python
@torch.no_grad()
def encode_chunk(encoder, mel_chunk, lookahead, attention_cache=None, convolution_cache=None):
    output = encoder(
        mel_chunk.unsqueeze(0),
        past_key_values=attention_cache,
        padding_cache=convolution_cache,
        use_cache=True,
        num_lookahead_tokens=lookahead,
    )
    return output.last_hidden_state, output.past_key_values, output.padding_cache
```

`mel_chunk` contains feature frames on the encoder's device and in its dtype. Both caches start as `None`; each call returns their updated values. The wrapper handles feature grouping and saves these values between calls. The attention cache carries past keys and values, while the convolution cache carries the earlier inputs needed by a causal kernel. Starting a new utterance clears both.

Convolutions also mix neighboring time steps. A symmetric offline kernel reads past and future samples. The conversion makes those convolutions causal and caches the past values needed at chunk boundaries. For the default training initialization, it removes future kernel taps and re-aligns the retained taps to the causal kernel's time positions. Simply copying weights without respecting that alignment changes which time steps they describe.

There are also model-specific details: the adapter preserves symmetric padding across frequency bins and initializes a per-channel scale and shift from the offline encoder's BatchNorm statistics. Those affine parameters can train with the student. These changes let the compatible weights transfer while the time axis becomes causal. The conversion lives in [`build_streaming_encoder()`](https://github.com/Deep-unlearning/smol-audio/blob/master/cohere_transcribe_streaming/model.py).

These changes remove information that the offline model used during training. The converted weights are an initialization; distillation teaches the student to recognize speech with its new context restrictions.

We express lookahead as `R`, the number of future **subsampled encoder frames**. Each represents 80 ms. The nominal chunk wait is `(R + 1) * 80 ms`, including the current frame:

| `R` | Nominal chunk wait |
|---|---|
| 0 | 80 ms |
| 1 | 160 ms |
| 3 | 320 ms |
| 6 | 560 ms |
| 13 | 1,120 ms |

The training recipe samples these five settings. A larger value offers more context before emitting a chunk. The table describes audio accumulation, rather than a measured caption delay: feature-window startup, decoder scheduling, computation, and application buffering also contribute.

## Why partial captions can change

The default attention-based decoder periodically reads the encoder states accumulated so far. It predicts text tokens autoregressively: each new token depends on preceding text and the available audio representation.

A partial hypothesis can change when more speech disambiguates an unfinished word. The wrapper compares consecutive hypotheses, commits an agreed prefix after sufficient audio, and holds back the final word. With `show_tentative=True`, `step()` also exposes tentative text. An application should replace its displayed partial caption with the latest return value.

At the endpoint, the default free-final mode can reconsider the utterance. For recordings within the default 30-second re-encoding limit, it runs the saved normalized features through the student encoder using final lookahead 13, then decodes again. This pass still uses the student, rather than the offline teacher.

For longer recordings, finalization uses the streaming states instead. Final generation has a duration-dependent budget capped at 256 new tokens, so long utterances can produce incomplete output. Segment at speech boundaries and call `reset()` for the next utterance.

The encoder's attention cache is bounded, but the wrapper accumulates encoder states for decoding. Whole-utterance memory and repeated decoder work therefore grow with duration. Bounded encoder caches alone do not make this complete recognition pipeline constant-memory.

## Prepare teacher labels

Distillation trains the student to imitate a teacher. First, the offline Cohere model transcribes the training clips. Generate a tiny dataset to exercise the workflow:

```bash
pip install -r requirements-train.txt
python distill_dataset.py \
    --dataset hf-internal-testing/librispeech_asr_dummy \
    --config clean --split validation --language en \
    --limit 16 --no-streaming --out work/relabel
```

The saved Hugging Face dataset embeds each clip as 16 kHz FLAC bytes. It also stores its language, duration, teacher token IDs, decoded teacher text, reference transcript, and ID. Training reads that audio directly from the labelled dataset.

The labeller gives longer clips a larger generation budget, within the decoder's limits. A generation that fails to emit its end token gets an empty label. `load_distill_dataset()` removes empty labels and clips outside the permitted duration range, so the student does not learn to stop at an incomplete teacher transcript.

For a real run, choose training audio that represents the intended languages, recording conditions, and speaking styles. Keep separate held-out audio with human references. Teacher labels carry the teacher's mistakes; measuring against those labels alone cannot establish recognition quality.

## Build and train the student

[`StreamingStudent`](https://github.com/Deep-unlearning/smol-audio/blob/master/cohere_transcribe_streaming/train.py) combines the converted encoder, Cohere decoder, and a new CTC head. Its loss has three parts:

- Decoder cross-entropy teaches the student to generate the teacher transcript.
- CTC provides a frame-level transcription objective with a blank token for frames without text.
- Hidden-state matching encourages selected student encoder layers to resemble the frozen teacher's representations.

The default weights are 1.0, 0.3, and 1.0. Each training batch samples a lookahead setting. Teacher parameters stay frozen, while the student learns from the combined loss.

`DistillCollator` decodes FLAC, builds features and masks, and prepares decoder inputs and targets. [`DistillTrainer`](https://github.com/Deep-unlearning/smol-audio/blob/master/cohere_transcribe_streaming/train.py) extends [Transformers Trainer](https://huggingface.co/docs/transformers/trainer) with those losses, transcription evaluation, and the custom checkpoint format. Trainer manages the training loop, optimizer updates, and scheduling.

Release the inference model before training so it does not occupy GPU memory. Then run this single-GPU demonstration using the labels above:

```python
import gc
import torch

del recognizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

from distill_dataset import DistillCollator, load_distill_dataset
from train import (
    DistillArguments, DistillTrainer, DistillTrainingArguments,
    build_student, estimate_training_mel_stats,
)

recipe = DistillArguments(data="work/relabel", freeze_decoder=True)
settings = DistillTrainingArguments(
    output_dir="work/tiny-student",
    max_steps=20,
    warmup_steps=2,
    per_device_train_batch_size=1,
    dataloader_num_workers=0,
    eval_strategy="no",
    eval_on_start=False,
    report_to="trackio",
    project="cohere-streaming",
    run_name="tutorial-tiny-run",
)

student, processor, encoder = build_student(recipe, settings.device, seed=settings.seed)
student.encoder.gradient_checkpointing_enable()
if settings.bf16:
    student.teacher.to(torch.bfloat16)
dataset = load_distill_dataset(recipe.data, recipe.max_seconds)
stats = estimate_training_mel_stats(dataset, processor, settings.seed)
collator = DistillCollator(processor, stats, recipe.norm, seed=settings.seed)

trainer = DistillTrainer(
    model=student,
    args=settings,
    train_dataset=dataset,
    data_collator=collator,
    processing_class=processor,
    distill_args=recipe,
    stats=stats,
)
trainer.train()
trainer.save_model()
```

The demonstration freezes the pretrained decoder and updates the streaming encoder and CTC head. Sixteen clips and twenty updates demonstrate the workflow; they will not produce a useful streaming recognizer. The example disables evaluation to keep that first run focused. The training notebook then evaluates held-out clips, and `train.py` exposes evaluation sets and lookahead settings for longer runs.

## Follow the run with Trackio

The [Trackio integration](https://huggingface.co/docs/trackio/transformers_integration) is enabled by `report_to="trackio"`. The trainer logs the total loss and its AED, CTC, and distillation components alongside training metrics. Compare these curves with held-out transcription scores before deciding whether training helps.

Open the local dashboard with:

```bash
trackio show --project cohere-streaming
```

This command follows the [Trackio dashboard workflow](https://huggingface.co/docs/trackio/quickstart). Local logging is enough for the tutorial; a shared Space can be configured for a longer experiment.

## Load your checkpoint

The final directory contains `student.safetensors`, `config.json`, `mel_stats.json`, and `log.json`. Release the training objects and load it through the same streaming wrapper:

```python
del trainer, student, encoder
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

from model import CohereStreaming

recognizer = CohereStreaming(
    checkpoint="work/tiny-student",
    lookahead=6,
    device=device,
    dtype=torch.float32,
)
recognizer.reset(language="en")
for start in range(0, len(audio), 1280):
    recognizer.step(audio[start:start + 1280])
print(recognizer.finish())
```

This snippet continues the training example and reuses `audio` and `device` from the inference example. The checkpoint format belongs to this adapter, so use `CohereStreaming` to load it. Loading also downloads the base Cohere model weights and processor if they are not already cached.

For larger experiments, `train.py` supports intermediate resume checkpoints and multi-GPU launch through Accelerate. Evaluate final accuracy and partial-caption behavior separately: a good endpoint transcript can coexist with unstable or delayed live text. Use the inference notebook to compare lookahead settings, and the distillation notebook to develop a training run with sufficient data and held-out evaluation.
