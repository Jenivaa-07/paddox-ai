import os
import json
import time
import platform
import psutil
import pandas as pd
import numpy as np
import requests
import torch
from services.race_predictor import RacePredictorService
from services.fantasy_predictor import FantasyPredictorService

def measure_function(func, args, warmup=20, measured=100):
    # Warmup
    for _ in range(warmup):
        try:
            func(*args)
        except Exception:
            pass
            
    # Measure
    latencies = []
    for _ in range(measured):
        start = time.perf_counter_ns()
        try:
            func(*args)
        except Exception:
            pass
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1e6) # ms
        
    return latencies

def measure_api(endpoint, payload, warmup=20, measured=100):
    url = f"http://127.0.0.1:8000/{endpoint}"
    for _ in range(warmup):
        try:
            requests.post(url, json=payload, timeout=2)
        except:
            pass
            
    latencies = []
    for _ in range(measured):
        start = time.perf_counter_ns()
        try:
            requests.post(url, json=payload, timeout=2)
        except:
            pass
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1e6) # ms
        
    return latencies

def calc_stats(latencies):
    arr = np.array(latencies)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "measurements": len(arr)
    }

def run_benchmarks():
    print("Initializing services for model-only benchmarking...")
    race_svc = RacePredictorService()
    fant_svc = FantasyPredictorService()
    
    race_req = {
        "recent_laps": [90000]*10,
        "features": {"grid_position": 1, "rolling_avg_finish": 1.5, "field_size": 20}
    }
    
    fant_req = {
        "drivers": [
            {
                "driver_id": "VER",
                "constructor_id": "Red Bull",
                "qualifying_position": 1,
                "features": {"rolling_avg_finish": 1.0}
            }
        ]
    }
    
    print("Measuring Model-Only LSTM...")
    race_model_lats = measure_function(race_svc.predict, (race_req,))
    print("Measuring Model-Only RF...")
    fant_model_lats = measure_function(fant_svc.predict_batch, (fant_req,))
    
    print("Measuring End-to-End API LSTM (Make sure FastAPI is running!)...")
    race_api_lats = measure_api("predict-race", race_req)
    print("Measuring End-to-End API RF...")
    fant_api_lats = measure_api("predict-fantasy", fant_req)
    
    os.makedirs("artifacts/benchmarks", exist_ok=True)
    
    df = pd.DataFrame({
        "race_model_ms": race_model_lats,
        "fant_model_ms": fant_model_lats,
        "race_api_ms": race_api_lats,
        "fant_api_ms": fant_api_lats
    })
    df.to_csv("artifacts/benchmarks/raw_latency_samples.csv", index=False)
    
    metrics = {
        "race_model": calc_stats(race_model_lats),
        "fant_model": calc_stats(fant_model_lats),
        "race_api": calc_stats(race_api_lats),
        "fant_api": calc_stats(fant_api_lats),
        "environment": {
            "os": platform.system(),
            "cpu": platform.processor() or "Unknown CPU",
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        }
    }
    
    with open("artifacts/benchmarks/latency_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Benchmarking complete. Saved to artifacts/benchmarks/")

if __name__ == "__main__":
    run_benchmarks()
