import pytest
import os
import sys

# Ensure module import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.scaling import scale_lap_time

def test_scaling_parity():
    # Test that the shared scale_lap_time function applies correct logic.
    assert scale_lap_time(100000.0) == 1.0
    assert scale_lap_time(50000.0) == 0.5
    assert scale_lap_time("85000.0") == 0.85
    assert scale_lap_time(0.0) == 0.0
