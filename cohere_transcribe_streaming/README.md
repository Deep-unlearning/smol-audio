# Stream and distill Cohere Transcribe

Two walkthroughs for running [Cohere Streaming](https://huggingface.co/Steveeeeeeen/cohere-streaming) and adapting the offline [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) teacher into a streaming student.

| Notebook | What you will do | Hardware |
|---|---|---|
| [Stream Cohere Transcribe](Stream_Cohere_Transcribe.ipynb) · [Open in Colab](https://colab.research.google.com/github/Deep-unlearning/smol-audio/blob/master/cohere_transcribe_streaming/Stream_Cohere_Transcribe.ipynb) | Load the released checkpoint, caption a clip in 80 ms chunks, compare lookahead settings, and run a small WER check. | GPU recommended; CPU works more slowly. |
| [Distill a streaming student](Distill_Streaming_Student.ipynb) · [Open in Colab](https://colab.research.google.com/github/Deep-unlearning/smol-audio/blob/master/cohere_transcribe_streaming/Distill_Streaming_Student.ipynb) | Teacher-label a small slice of audio, train with `transformers.Trainer`, evaluate held-out clips, and stream with the saved checkpoint. | A100 80 GB or H100 recommended for training the 2B model. |

The companion [Hugging Face blog draft](streaming-asr-with-transformers.md) explains the streaming design and walks through the Transformers recipe.

Try the [live microphone demo](https://huggingface.co/spaces/Steveeeeeeen/cohere-streaming-demo): choose a language, record up to 25 seconds, and watch the transcript update as you speak. Stop finalizes the last words. The [Space source](space/) includes instructions for running locally or deploying your own copy.

## Run locally

Use Python 3.10 or newer in a fresh environment. Install dependencies and run the scripts from this example's directory:

```bash
git clone https://github.com/Deep-unlearning/smol-audio.git
cd smol-audio/cohere_transcribe_streaming
pip install -r requirements.txt
hf auth login
python stream_file.py speech.wav --language en --lookahead 6 --realtime
```

Accept access to the [base Cohere model](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) before loading it.

`transformers==5.14.1` is pinned for its streaming encoder internals. The CLI resamples files to 16 kHz mono. The notebooks explain `CohereStreaming.reset()`, `step()`, and `finish()`, supported languages, and lookahead settings. Final re-encoding defaults to a 30-second limit; longer recordings can hit the 256-token decoding budget. Reset at speech boundaries.

## Train a student

Install the training dependencies, then label audio and train on the resulting dataset:

```bash
pip install -r requirements-train.txt
python distill_dataset.py \
    --dataset hf-internal-testing/librispeech_asr_dummy --config clean --split validation \
    --language en --limit 16 --no-streaming --out work/relabel
python train.py --data work/relabel --max_steps 20 \
    --per_device_train_batch_size 1 --eval_strategy no --eval_on_start false \
    --report_to trackio --output_dir work/tiny-student
```

Training metrics use [Trackio](https://huggingface.co/docs/trackio/transformers_integration) under the `cohere-streaming` project. This is a toy run. The notebook explains the losses, evaluation, and checkpoint format. Generate labels with `distill_dataset.py`; older pointer-style labels from the original training run are incompatible.

## Hugging Face Jobs

After `hf auth login`, with `HF_TOKEN` available for the job secret, [run_hf_job.sh](run_hf_job.sh) can submit the scripts to rented GPUs. It installs the training requirements and uploads `results/`, including Trackio metrics in `results/trackio`. Set `GIT_REF` to a pushed branch or commit to preview changes before merge.

```bash
./run_hf_job.sh distill a100-large 6h python train.py \
    --data your-user/teacher-labels --max_steps 2000 \
    --output_dir results/train --push_to_hub --hub_model_id your-user/streaming-student
```

For multiple GPUs use `accelerate launch --multi_gpu train.py ...`. Write checkpoints under `results/` or use `--push_to_hub`. See the wrapper header for settings.

## Source

Adapted from [cohere-transcribe-streaming PR #3](https://github.com/Deep-unlearning/cohere-transcribe-streaming/pull/3), revision [`1e1b41c`](https://github.com/Deep-unlearning/cohere-transcribe-streaming/tree/1e1b41ca259b85d0e2b257b6a794d1b9c1e1ef2d). Benchmarks and release records remain in the [source repository](https://github.com/Deep-unlearning/cohere-transcribe-streaming/tree/1e1b41ca259b85d0e2b257b6a794d1b9c1e1ef2d/benchmark).

Apache-2.0, like the base model. Training data retain their own licenses. See [LICENSE](../LICENSE).
