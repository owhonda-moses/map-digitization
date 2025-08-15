#!/usr/bin/env bash
set -euo pipefail


if [ -f pat.env ]; then
  chmod 600 pat.env
  source ./pat.env
  echo "PAT loaded."
else
  echo "pat.env not found." >&2
  exit 1
fi


# git config
echo "Configuring git"
git config --global user.name  "owhonda-moses"
git config --global user.email "owhondamoses7@gmail.com"
git config --global init.defaultBranch main

# ~/.netrc for https auth
cat > "$HOME/.netrc" <<EOF
machine github.com
  login x-access-token
  password $GITHUB_TOKEN
EOF
chmod 600 "$HOME/.netrc"

# bootstrap
echo "Bootstrapping git repo…"
git remote remove origin 2>/dev/null || true
if [ ! -d .git ]; then
  git init
fi
git remote add origin \
  https://x-access-token:${GITHUB_TOKEN}@github.com/owhonda-moses/map-digitization.git
git fetch origin main --depth=1 2>/dev/null || true
git checkout main 2>/dev/null || git checkout -b main



echo "Installing Tesseract (OCR) and GDAL"
apt-get update -y >/dev/null 2>&1
apt-get install -y --no-install-recommends \
  tesseract-ocr \
  libgdal-dev \
  >/dev/null 2>&1
echo "System dependencies installed."


echo "Setting up Python with Poetry and PyTorch"
pip install --upgrade pip poetry >/dev/null 2>&1
poetry env use python3.11 >/dev/null 2>&1
poetry install --no-interaction --no-ansi >/dev/null 2>&1
echo "Python env ready: $(poetry run python -V)"


echo "Setup complete"
