#!/bin/bash

echo "🔍 Running code quality checks..."

echo "📝 Checking code formatting with Black..."
if uv run black --check --diff backend/ main.py; then
    echo "✅ Code formatting is correct"
else
    echo "❌ Code formatting issues found. Run './scripts/format.sh' to fix."
    exit 1
fi

echo "🧪 Running tests..."
if uv run pytest backend/tests/ -v; then
    echo "✅ All tests passed"
else
    echo "❌ Tests failed"
    exit 1
fi

echo "🎉 All quality checks passed!"