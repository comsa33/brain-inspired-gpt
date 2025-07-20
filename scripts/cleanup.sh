#!/bin/bash

# Cleanup script for CortexGPT project

echo "🧹 Cleaning up unnecessary files..."

# Remove temporary test files
rm -f check_data_simple.py
rm -f check_prepared_data.py
rm -f create_test_data.py
rm -f test_tokenization.py
rm -f TRAINING_FIX_SUMMARY.md

# Keep organized demo scripts in scripts folder
mkdir -p scripts/demos
mv -f learning_effect_demo.py scripts/demos/ 2>/dev/null || true
mv -f natural_language_demo.py scripts/demos/ 2>/dev/null || true
mv -f minimal_demo.py scripts/demos/ 2>/dev/null || true

# Keep test files in tests folder
mkdir -p tests
mv -f test_overfit.py tests/ 2>/dev/null || true
mv -f demo_tokenizer.py tests/ 2>/dev/null || true

# Move data setup scripts
mkdir -p scripts/data
mv -f create_demo_data.py scripts/data/ 2>/dev/null || true
mv -f setup_datasets.py scripts/data/ 2>/dev/null || true

# Clean wandb cache (keep only latest run)
rm -rf wandb/run-*/tmp
find wandb -name "*.log" -not -path "*/latest-run/*" -delete 2>/dev/null || true

# Clean pycache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Keep documentation
mkdir -p docs
mv -f training_tips.md docs/ 2>/dev/null || true

echo "✅ Cleanup complete!"
echo ""
echo "📁 New structure:"
echo "   scripts/"
echo "      ├── data/        # Data preparation scripts"
echo "      ├── demos/       # Demo scripts"
echo "      └── cleanup.sh   # This cleanup script"
echo "   tests/              # Test scripts"
echo "   docs/               # Documentation"
echo "   cortexgpt/          # Main package"
echo "   data/               # Training data"
echo ""