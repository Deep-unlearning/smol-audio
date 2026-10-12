#!/usr/bin/env bash
# Run a command on a rented Hugging Face GPU and upload results/ with its logs.
# Usage: ./run_hf_job.sh <run-name> <flavor> <timeout> <command...>
# Setup: pip install -U huggingface_hub && hf auth login
# Optional environment settings:
#   REPO_URL: git repository; GIT_REF: pushed branch, tag, or commit
#   REPO_SUBDIR: example directory within the repository
#   RESULTS_REPO: output dataset (defaults to <your-user>/cohere-streaming-results)
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

RUN="$1"
FLAVOR="$2"
TIMEOUT="$3"
shift 3
REPO_URL="${REPO_URL:-https://github.com/Deep-unlearning/smol-audio}"
GIT_REF="${GIT_REF:-master}"
REPO_SUBDIR="${REPO_SUBDIR-cohere_transcribe_streaming}"
if [[ -z "${RESULTS_REPO:-}" ]]; then
  HF_USERNAME=$(hf auth whoami | awk 'NR == 1 {print $NF}')
  RESULTS_REPO="$HF_USERNAME/cohere-streaming-results"
fi

if [[ " $* " == *" train.py "* && " $* " != *"push_to_hub"* ]]; then
  echo "Only results/ is uploaded. Save with --output_dir results/train or --push_to_hub --hub_model_id <you>/<name>." >&2
fi

# This script runs inside the job container.
JOB_SCRIPT=$(cat <<'INNER'
set -uo pipefail
if ! git clone -q "$REPO_URL" /work; then
  echo "Could not clone $REPO_URL" >&2
  exit 1
fi
cd /work || exit 1
if ! git checkout -q "$GIT_REF"; then
  echo "Could not check out $GIT_REF" >&2
  exit 1
fi
if ! cd "/work/$REPO_SUBDIR"; then
  echo "Example directory $REPO_SUBDIR is missing" >&2
  exit 1
fi
mkdir -p results
export TRACKIO_DIR="${TRACKIO_DIR:-$PWD/results/trackio}"
# Keep going after a failure so the log can still be uploaded.
{
  echo "== $(date -u) run=$RUN cmd=$*"
  nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "no GPU"
  uv pip install --system -q -r requirements-train.txt
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
