#!/bin/bash
rm -rf .venv
rm -rf mmdetection
rm -rf models

uv venv .venv
source .venv/bin/activate
uv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
uv pip install -r requirements.txt
uv run mim install "mmcv>=2.0.0rc4,<2.1.0"
uv pip install -r requirements.txt
mkdir -p models
mkdir -p output
mkdir -p results
./evaluation/download_models.sh models

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout 3.x
uv pip install -v -e . --no-build-isolation
cd ..
