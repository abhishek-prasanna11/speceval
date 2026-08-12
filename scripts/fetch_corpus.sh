#!/usr/bin/env bash
# Fetch the PEP corpus. Shallow clone -- history is not part of the study.
# The corpus is gitignored: it is input data, not source.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d peps/.git ]; then
  echo "peps/ already present; pulling latest"
  git -C peps pull --depth 1 --ff-only
else
  git clone --depth 1 https://github.com/python/peps.git peps
fi
echo "PEP files: $(ls peps/peps/pep-*.rst | wc -l | tr -d ' ')"
