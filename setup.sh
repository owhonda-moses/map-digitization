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
git config --global user.name  "         " # set your git identity
git config --global user.email "         "
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


CONDA_INSTALL_PATH="$HOME/miniconda"
CONDA_ENV_NAME="map-cv"


if [ -d "$CONDA_INSTALL_PATH" ]; then
    echo "Miniconda is installed."
else
    echo "Installing Miniconda"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$CONDA_INSTALL_PATH"
    rm miniconda.sh
    echo "Miniconda installed."
fi

source "$CONDA_INSTALL_PATH/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "Installing Mamba"
conda install -n base -c conda-forge mamba -y

if conda env list | grep -q "^${CONDA_ENV_NAME}\s"; then
  echo "Environment '${CONDA_ENV_NAME}' exists."
else
  echo "Creating environment '${CONDA_ENV_NAME}'"
  yes | mamba env create -f environment.yml
  echo "Environment created."
fi

echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc

echo "Setup complete"