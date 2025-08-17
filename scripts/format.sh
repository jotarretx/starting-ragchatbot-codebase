#!/bin/bash

echo "🎨 Running Black code formatter..."
uv run black backend/ main.py

echo "✅ Code formatting complete!"