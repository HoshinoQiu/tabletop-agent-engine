#!/bin/bash
# Quick start script for Tabletop-Agent-Engine

set -e

echo "🚀 Tabletop-Agent-Engine Quick Start"
echo "====================================="

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Initialize rulebook
echo "📚 Initializing rulebook..."
if [ -f "data/rules/example_rules.txt" ]; then
    python init_rulebook.py --input data/rules/example_rules.txt
else
    echo "⚠️  Example rulebook not found. Please add a rulebook file to data/rules/"
fi

# Start the server
echo ""
echo "✅ Setup complete!"
echo ""
echo "Starting API server..."
echo "🚀 Server will be available at http://localhost:8000"
echo ""
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
