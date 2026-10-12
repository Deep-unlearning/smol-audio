#!/usr/bin/env bash
# Run a command from this example on a Hugging Face Job (a rented GPU).
#
#   ./run_hf_job.sh <run-name> <flavor> <timeout> <command...>
#
# Examples:
#   ./run_hf_job.sh relabel a100-large 6h python distill_dataset.py --help
#   ./run_hf_job.sh distill a100-large 12h accelerate launch train.py --help
#   ./run_hf_job.sh inspect a100-large 1h python stream_file.py --help
#
# What happens: the job clones smol-audio, enters this example, installs it, runs your command, and uploads everything written to
# `results/` (plus the log) to a Hub dataset under runs/<run-name>/.
#
# One-time setup:  pip install -U huggingface_hub && hf auth login      (Jobs need a Pro / Team / Enterprise plan)
# Settings (environment variables, all optional):
#   REPO_URL      git repo to clone      (default: this project on GitHub)
#   GIT_REF       branch / tag / commit  (default: smol-audio's `master`). The job clones from GitHub, so
#                 push your changes first; uncommitted local edits never reach the job.
#   RESULTS_REPO  Hub dataset for outputs (default: <your-username>/cohere-streaming-results, created if missing)
#   REPO_SUBDIR   example within repo    (default: cohere_transcribe_streaming)
set -euo pipefail

usage() {
  echo "Usage: $0 <run-name> <flavor> <timeout> <command...>" >&2
}
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 4 ]]; then
  usage
  exit 2
fi

RUN="$1"; FLAVOR="$2"; TIMEOUT="$3"; shift 3
REPO_URL="${REPO_URL:-https://github.com/Deep-unlearning/smol-audio}"
GIT_REF="${GIT_REF:-master}"
REPO_SUBDIR="${REPO_SUBDIR-cohere_transcribe_streaming}"
RESULTS_REPO="${RESULTS_REPO:-$(hf auth whoami | awk 'NR == 1 {print $NF}')/cohere-streaming-results}"

# Only `results/` is uploaded when the job ends, and the container is deleted afterwards. A trained checkpoint
# written anywhere else is lost, so make train.py push it to the Hub.
if [[ " $* " == *" train.py "* && " $* " != *"push_to_hub"* ]]; then
  echo "WARNING: train.py without --push_to_hub --hub_model_id <you>/<name>: the checkpoint will be lost when the job ends." >&2
  echo "         (multi-GPU flavors: use 'accelerate launch --multi_gpu train.py ...' or 'torchrun --nproc_per_node N train.py ...')" >&2
fi

# Everything below runs INSIDE the job container.
JOB_SCRIPT=$(cat <<'INNER'
set -uo pipefail
git clone -q "$REPO_URL" /work && cd /work && git checkout -q "$GIT_REF" || { echo "git clone/checkout of $REPO_URL @ $GIT_REF failed" >&2; exit 1; }
cd "/work/$REPO_SUBDIR" || { echo "example directory $REPO_SUBDIR is missing" >&2; exit 1; }
mkdir -p results
# No `set -e` from here on: whatever happens, we still upload results/ (with job.log) at the end.
{
  echo "== $(date -u) run=$RUN cmd=$*"
  nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "no GPU"
  uv pip install --system -q -e ".[train]"
} 2>&1 | tee results/job.log
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  echo "== dependency install failed, not running the command" | tee -a results/job.log
  status=1
else
  "$@" 2>&1 | tee -a results/job.log
  status=${PIPESTATUS[0]}
fi
echo "== exit status $status" | tee -a results/job.log
hf repo create "$RESULTS_REPO" --repo-type dataset --exist-ok
hf upload "$RESULTS_REPO" results "runs/$RUN" --repo-type dataset --commit-message "$RUN (exit $status)"
exit $status
INNER
)

hf jobs run --detach --flavor "$FLAVOR" --timeout "$TIMEOUT" \
  --secrets HF_TOKEN \
  -e REPO_URL="$REPO_URL" -e GIT_REF="$GIT_REF" -e REPO_SUBDIR="$REPO_SUBDIR" -e RESULTS_REPO="$RESULTS_REPO" -e RUN="$RUN" \
  -e HF_XET_HIGH_PERFORMANCE=1 \
  --label project=cohere-streaming --label run="$RUN" \
  ghcr.io/astral-sh/uv:python3.12-bookworm \
  bash -c "$JOB_SCRIPT" _ "$@"
