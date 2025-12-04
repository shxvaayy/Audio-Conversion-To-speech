#!/bin/bash

echo "🚀 Starting Arali.ai Video Analyzer..."
echo ""

cd "/Volumes/SHIVAY DATA/Writory-branch/without ai/video-analyzer"

# Activate virtual environment
source venv/bin/activate

# Start server
echo "Starting Flask server on http://localhost:5000"
echo "Press Ctrl+C to stop"
echo ""

python app_simple.py
