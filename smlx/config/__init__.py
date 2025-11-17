#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Configuration package for SMLX.

Provides centralized configuration management for various aspects
of the SMLX framework including memory management, model settings,
and runtime behavior.
"""

from .memory import MemoryConfig, get_default_config, reset_default_config

__all__ = [
    'MemoryConfig',
    'get_default_config',
    'reset_default_config',
]
