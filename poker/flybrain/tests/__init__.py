"""Tests for the fly-brain module.

Deliberately a separate package from poker/tests/: that package's __init__ imports
poker.main (PyQt6, matplotlib, tensorflow) and reads a MongoDB URL out of config.ini at
import time, so it cannot run without the full GUI stack and a configured database.
These tests need only numpy and scipy, and run on the synthetic connectome, so they stay
runnable on their own and in CI.
"""
