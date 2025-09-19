#!/bin/bash
# ========================================
# ProtFlap Installation Script
# Sets up Python environment and installs dependencies
# ========================================

ENV_NAME=protflap
PYTHON_VER=3.8

echo "🔹 Creating Conda environment: $ENV_NAME"
conda create --name $ENV_NAME python=$PYTHON_VER -y

echo "🔹 Activating environment"
source $(conda info --base)/bin/activate $ENV_NAME

echo "🔹 Installing Python dependencies"
pip install -r requirements.txt

echo "✅ Installation complete!"
echo "   Activate environment with: conda activate $ENV_NAME"

