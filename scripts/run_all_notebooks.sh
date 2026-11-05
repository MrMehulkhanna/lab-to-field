#!/usr/bin/env bash
# Execute notebooks in order and save the outputs inside each .ipynb.
# Usage: scripts/run_all_notebooks.sh                 (all notebooks)
#        scripts/run_all_notebooks.sh notebooks/03*.ipynb notebooks/04*.ipynb
set -euo pipefail
export HF_HUB_VERBOSITY=error  # hide the "unauthenticated requests" notice
cd "$(dirname "$0")/.."
if [ $# -eq 0 ]; then set -- notebooks/0*.ipynb; fi
for nb in "$@"; do
  echo "=== $(date +%H:%M:%S) running $nb"
  uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$nb"
  echo "=== $(date +%H:%M:%S) finished $nb"
done
echo "=== ALL DONE"
