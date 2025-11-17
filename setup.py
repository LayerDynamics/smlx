"""
SMLX (Smol MLX) Setup Configuration

This setup.py provides backwards compatibility and enables editable installs.
Modern configuration is in pyproject.toml - add new dependencies there.

Install for development:
    pip install -e .
    pip install -e ".[dev]"
    pip install -e ".[all]"
"""

from setuptools import setup

# All configuration is in pyproject.toml
# This file exists for:
# 1. Backwards compatibility with older pip versions
# 2. Enabling editable installs (pip install -e .)
# 3. Custom build logic if needed in the future

if __name__ == "__main__":
    setup()
