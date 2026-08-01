SCALING_CONFIG = {
  "lap_time_unit": "milliseconds",
  "lap_time_scale": 100000.0,
  "scaled_range_expected": [0.0, 2.0]
}

def scale_lap_time(lap_time_ms):
    """
    Scales a raw lap time in milliseconds to the expected normalized range.
    Uses the SCALING_CONFIG contract.
    """
    return float(lap_time_ms) / SCALING_CONFIG["lap_time_scale"]
