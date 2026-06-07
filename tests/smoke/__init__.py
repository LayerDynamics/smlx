"""Smoke tests: per-model load + run + real-output assertions.

This package is the enforcement mechanism for the "no placeholder output"
requirement. Each model is loaded with its real weights, run on a known input,
and its output is checked for *correctness* (not merely non-emptiness) by the
modality-specific assertions in :mod:`tests.smoke.assertions`. A model running
on random/placeholder weights fails these assertions.
"""
