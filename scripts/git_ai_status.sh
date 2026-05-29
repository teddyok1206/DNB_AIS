#!/usr/bin/env sh
set -eu
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

echo "repo: $ROOT"
echo "branch: $(git branch --show-current)"
echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo none)"
echo

echo "tracked changes:"
git status --short --untracked-files=no

echo

echo "untracked lightweight candidates:"
git status --short --untracked-files=normal -- \
  . \
  ':(exclude)outputs/**' \
  ':(exclude)artifacts/**' \
  ':(exclude)**/*.tif' \
  ':(exclude)**/*.tiff' \
  ':(exclude)**/*.nc' \
  ':(exclude)**/*.db' \
  ':(exclude)**/*.sqlite' \
  ':(exclude)**/*.pt' \
  ':(exclude)**/*.npy' \
  ':(exclude)**/*.npz' \
  ':(exclude)**/*.png' \
  ':(exclude)**/*.pdf' | sed -n '1,200p'
