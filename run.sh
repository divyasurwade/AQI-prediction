#!/bin/bash

echo "======================================================================"
echo "         Air Quality Prediction Comparative Analysis System"
echo "======================================================================"
echo ""

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    exit 1
fi

# Create virtual environment if missing
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

# Activate & install dependencies
source .venv/bin/activate
echo "Installing dependencies..."
pip install -r requirements.txt

# Create folders
mkdir -p data/raw data/processed models

# Launch app
echo ""
echo "Starting Flask web server on http://localhost:5000..."
python app.py
