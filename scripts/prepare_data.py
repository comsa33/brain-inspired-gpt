#!/usr/bin/env python3
"""
Wrapper script for data preparation
Redirects to the actual prepare_datasets.py with proper path handling
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import and run the actual prepare_datasets module
from cortexgpt.data.prepare_datasets import main

if __name__ == "__main__":
    main()