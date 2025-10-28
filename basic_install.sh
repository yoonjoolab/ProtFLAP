#!/bin/bash
# ========================================
# ProtFlap Installation Script
# Installs Python dependencies and DSSP
# ========================================

echo "🔹 Installing Python dependencies"
pip install -r requirements.txt

echo "🔹 Installing DSSP (system dependency)"
if ! command -v mkdssp &> /dev/null
then
    sudo apt update
    sudo apt install -y dssp
else
    echo "   DSSP is already installed ✅"
fi

echo "✅ Installation complete!"

