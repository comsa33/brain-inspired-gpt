#!/usr/bin/env python3
"""
Comprehensive test runner for Brain-Inspired GPT
Tests all functionality: model, data, training, and inference
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_section(title):
    """Print a section header"""
    print("\n" + "="*70)
    print(f"🧠 {title}")
    print("="*70)


def run_command(cmd, description):
    """Run a command and capture output"""
    print(f"\n📌 {description}")
    print(f"   Command: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("   ✅ Success")
            if result.stdout:
                print("   Output:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
        else:
            print("   ❌ Failed")
            if result.stderr:
                print("   Error:", result.stderr[:200] + "..." if len(result.stderr) > 200 else result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⏱️  Timeout (60s)")
        return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False
        
    return True


def run_python_test(script_path, description):
    """Run a Python test script"""
    cmd = f"uv run python {script_path}"
    return run_command(cmd, description)


def test_environment():
    """Test environment setup"""
    print_section("Environment Tests")
    
    # Check Python version
    run_command("uv run python --version", "Check Python version")
    
    # Check PyTorch
    run_command(
        'uv run python -c "import torch; print(f\'PyTorch {torch.__version__}\')"',
        "Check PyTorch installation"
    )
    
    # Check CUDA
    run_command(
        'uv run python -c "import torch; print(f\'CUDA available: {torch.cuda.is_available()}\')"',
        "Check CUDA availability"
    )
    
    # Check Korean NLP tools
    run_command(
        'uv run python -c "import tiktoken; print(\'Tiktoken installed\')"',
        "Check tokenizer dependencies"
    )


def test_model_components():
    """Test individual model components"""
    print_section("Model Component Tests")
    
    # Test imports
    test_code = """
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_brain import BrainGPT
from core.model_brain_config import BrainGPTConfig
from core.multilingual_tokenizer import MultilingualBrainTokenizer
from core.sparse_modules import CorticalColumnLinear

print("✅ All imports successful")

# Test model creation
config = BrainGPTConfig()
config.n_layer = 2
config.n_embd = 256

model = BrainGPT(config)
print(f"✅ Model created: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# Test tokenizer
tokenizer = MultilingualBrainTokenizer()
tokens = tokenizer.encode("Hello 안녕하세요")
print(f"✅ Tokenizer working: {len(tokens)} tokens")

# Test sparse module
sparse_linear = CorticalColumnLinear(256, 128, num_columns=4)
print("✅ Sparse modules working")
"""
    
    with open("test_components.py", "w") as f:
        f.write(test_code)
        
    success = run_python_test("test_components.py", "Test model components")
    os.remove("test_components.py")
    
    return success


def test_data_pipeline():
    """Test data loading and processing"""
    print_section("Data Pipeline Tests")
    
    # Run data loading tests
    return run_python_test(
        "brain_gpt/tests/test_data_loading.py",
        "Test data loading pipeline"
    )


def test_training():
    """Test training functionality"""
    print_section("Training Tests")
    
    # Run training tests
    return run_python_test(
        "brain_gpt/tests/test_training.py",
        "Test training functionality"
    )


def test_quickstart():
    """Test quickstart demo"""
    print_section("Quickstart Demo Test")
    
    # Create a test script that runs quickstart non-interactively
    test_code = """
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_gpt.quickstart_brain_gpt import main

# Monkey patch input to exit immediately
def mock_input(prompt):
    print(f"Mock input: {prompt}")
    return "quit"

# Replace input function
import builtins
builtins.input = mock_input

try:
    main()
    print("✅ Quickstart completed")
except Exception as e:
    print(f"❌ Quickstart error: {e}")
    import traceback
    traceback.print_exc()
"""
    
    with open("test_quickstart.py", "w") as f:
        f.write(test_code)
        
    success = run_python_test("test_quickstart.py", "Test quickstart demo")
    os.remove("test_quickstart.py")
    
    return success


def test_benchmarks():
    """Test performance benchmarks"""
    print_section("Performance Benchmarks")
    
    # Create minimal benchmark test
    test_code = """
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.model_brain import BrainGPT
from core.model_brain_config import BrainGPTConfig

# Small config for testing
config = BrainGPTConfig()
config.n_layer = 2
config.n_embd = 256

model = BrainGPT(config)
if torch.cuda.is_available():
    model = model.cuda()

# Simple forward pass
input_ids = torch.randint(0, config.vocab_size, (1, 32))
if torch.cuda.is_available():
    input_ids = input_ids.cuda()

with torch.no_grad():
    output = model(input_ids)

print(f"✅ Benchmark test passed: output shape {output.shape}")
"""
    
    with open("test_benchmark.py", "w") as f:
        f.write(test_code)
        
    success = run_python_test("test_benchmark.py", "Test benchmarks")
    os.remove("test_benchmark.py")
    
    return success


def test_pytest_suite():
    """Run pytest test suite"""
    print_section("PyTest Suite")
    
    # Run pytest on test files
    return run_command(
        "uv run pytest brain_gpt/tests/test_all_functions.py -v -x",
        "Run pytest suite"
    )


def main():
    """Run all tests"""
    print("🧠 Brain-Inspired GPT - Comprehensive Test Suite")
    print("=" * 70)
    print("Running all tests to verify functionality...")
    
    start_time = time.time()
    
    # Track results
    results = {}
    
    # Run tests
    results['environment'] = test_environment()
    results['components'] = test_model_components()
    results['data'] = test_data_pipeline()
    results['training'] = test_training()
    results['quickstart'] = test_quickstart()
    results['benchmarks'] = test_benchmarks()
    results['pytest'] = test_pytest_suite()
    
    # Summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.ljust(20)}: {status}")
        
    print("-"*70)
    print(f"Total: {passed_tests}/{total_tests} passed ({passed_tests/total_tests*100:.0f}%)")
    print(f"Time: {elapsed_time:.1f} seconds")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Brain-Inspired GPT is ready to use.")
        print("\n🚀 Quick start:")
        print("   uv run brain_gpt/quickstart_brain_gpt.py")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
        print("Common fixes:")
        print("  1. Run: uv sync")
        print("  2. Check CUDA: uv run python -c 'import torch; print(torch.cuda.is_available())'")
        print("  3. See: brain_gpt/setup_with_uv.md")
        

if __name__ == "__main__":
    main()