"""Pytest configuration for the unifiedwvalticolocs test suite."""

import os

# Use a non-interactive matplotlib backend for the plotting tests.
os.environ.setdefault("MPLBACKEND", "Agg")
