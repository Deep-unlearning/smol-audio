# Stream and distill Cohere Transcribe

Two walkthroughs for running [Cohere Streaming](https://huggingface.co/Steveeeeeeen/cohere-streaming) and adapting the offline [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) teacher into a streaming student.

| Notebook | What you will do | Hardware |
|---|---|---|
| [Stream Cohere Transcribe](Stream_Cohere_Transcribe.ipynb) · [Open in Colab](https://colab.research.google.com/github/Deep-unlearning/smol-audio/blob/master/cohere_transcribe_streaming/Stream_Cohere_Transcribe.ipynb) | Load the released checkpoint, caption a clip in 80 ms chunks, compare lookahead settings, and run a small WER check. | GPU recommended; CPU works more slowly. |
| [Distill a streaming student](Distill_Streaming_Student.ipynb) · [Open in Colab](https://colab.research.google.com/github/Deep-unlearning/smol-audio/blob/master/cohere_transcribe_streaming/Distill_Streaming_Student.ipynb) | Teacher-label a small slice of audio, train with `transformers.Trainer`, evaluate held-out clips, and stream with the saved checkpoint. | A100 80 GB or H100 recommended for training the 2B model. |

The scripts and tests live alongside the notebooks so both walkthroughs use the implementation in this repository.

## Run locally

Use Python 3.10 or newer in a fresh environment. Install dependencies and run the scripts from this example's directory:

```bash
git clone https://github.com/Deep-unlearning/smol-audio.git
cd smol-audio/cohere_transcribe_streaming
pip install -r requirements.txt
python stream_file.py speech.wav --language en --lookahead 6 --realtime
```

The requirements pin `transformers==5.14.1` because the encoder conversion uses that version's streaming encoder internals. Model weights are downloaded when you load the model. The input is mono float32 audio at 16 kHz; the CLI resamples other rates.

```python
from model import CohereStreaming

asr = CohereStreaming(lookahead=6)
asr.reset(language="en")
for chunk in audio_chunks:  # 1280 samples (80 ms) per chunk at 16 kHz
    print(asr.step(chunk))
print(asr.finish())
```

The checkpoint supports en, fr, de, es, it, pt, nl, and pl. Choose lookahead 0, 1, 3, 6, or 13 for a nominal encoder wait of 80 ms, 160 ms, 320 ms, 560 ms, or 1.12 s. Decoder updates and computation add to that wait. For utterances up to the default 30-second re-encode limit, the final pass uses lookahead 13 regardless of the live setting. Longer utterances use the accumulated streaming states, so their final text can depend on live lookahead. Final decoding has a 256-token budget and can truncate long recordings. Call `finish()` and `reset()` at speech boundaries; this is an utterance-level captioning example.

## Train a student

Install the training dependencies, then label audio and train on the resulting dataset:

```bash
pip install -r requirements-train.txt
python distill_dataset.py \
    --dataset hf-internal-testing/librispeech_asr_dummy --config clean --split validation \
    --language en --limit 16 --no-streaming --out work/relabel
python train.py --data work/relabel --max_steps 20 \
    --per_device_train_batch_size 4 --eval_strategy no --eval_on_start false \
    --report_to none --output_dir work/tiny-student
```

The notebook explains the teacher, student, feature normalization, three loss terms, and checkpoint format. A few steps on 16 clips demonstrate the pipeline; they do not produce a useful speech model. This dataset format embeds FLAC audio and teacher token IDs. It cannot read the older pointer-style labels used by the original large training run; generate new labels with `distill_dataset.py`.

| File | Purpose |
|---|---|
| [model.py](model.py) | Feature extraction, cache-aware encoder conversion, decoding, and `CohereStreaming`. |
| [stream_file.py](stream_file.py) | File streaming CLI, run with `python stream_file.py`. |
| [distill_dataset.py](distill_dataset.py) | Teacher relabelling, encoded dataset loading, and training batches. |
| [train.py](train.py) | Student losses, `DistillTrainer`, evaluation, and checkpoint saving. |
| [run_hf_job.sh](run_hf_job.sh) | Submit a command to Hugging Face Jobs and upload `results/` and logs. |
| [tests/](tests/) | CPU checks with tiny random models and processor configuration. |

## Hugging Face Jobs

Jobs rent GPUs and incur charges. Run them explicitly after `hf auth login`, with `HF_TOKEN` available for the job secret. The wrapper clones smol-audio at `master`, enters `cohere_transcribe_streaming`, installs `requirements-train.txt`, and uploads `results/` to a Hub dataset under `runs/<run-name>/`. Set `GIT_REF` to a pushed branch or commit to run changes before they are merged.

```bash
./run_hf_job.sh distill a100-large 6h python train.py \
    --data your-user/teacher-labels --max_steps 2000 \
    --output_dir results/train --push_to_hub --hub_model_id your-user/streaming-student
```

For multiple GPUs use `accelerate launch --multi_gpu train.py ...`. The wrapper uploads only `results/`; write checkpoints there or use `--push_to_hub`. Run `./run_hf_job.sh --help` and inspect the script header for repository and result destination settings.

## Validation and source

```bash
pip install -r requirements-train.txt pytest
python -m pytest tests -q
```

Tests check encoder weight mapping, cached/full-sequence parity, padding invariance, incremental features, endpoint re-encoding, teacher labels, loss gradients, and checkpoint saving/resumption. They fetch the public processor configuration; they do not download the 2B weights.

Adapted from [cohere-transcribe-streaming PR #3](https://github.com/Deep-unlearning/cohere-transcribe-streaming/pull/3), revision [`1e1b41c`](https://github.com/Deep-unlearning/cohere-transcribe-streaming/tree/1e1b41ca259b85d0e2b257b6a794d1b9c1e1ef2d). Installation, repository paths, documentation, and integration fixes are maintained here. Benchmark scripts, results, and release records remain in the [source benchmark directory](https://github.com/Deep-unlearning/cohere-transcribe-streaming/tree/1e1b41ca259b85d0e2b257b6a794d1b9c1e1ef2d/benchmark).

Apache-2.0, like the base model. Training data retain their own licenses. See [LICENSE](LICENSE).
