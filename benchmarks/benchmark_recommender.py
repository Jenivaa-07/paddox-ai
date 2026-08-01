import time
import json
import statistics
import os
import platform
import pandas as pd
from services.recommendation_service import recommendation_service

def run_latency_benchmark():
    print("Running Recommendation Latency Benchmark...")
    warm_up_count = 50
    measured_call_count = 500
    
    # Warm up
    for _ in range(warm_up_count):
        recommendation_service.get_recommendations("test_user", {}, 10, [])
        
    latencies = []
    
    for _ in range(measured_call_count):
        start = time.perf_counter()
        recommendation_service.get_recommendations("test_user", {}, 10, [])
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
        
    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.5)]
    p95 = latencies[int(len(latencies) * 0.95)]
    mean = statistics.mean(latencies)
    stddev = statistics.stdev(latencies)
    
    report = {
        "warm_up_count": warm_up_count,
        "measured_call_count": measured_call_count,
        "p50_latency_ms": round(p50, 4),
        "p95_latency_ms": round(p95, 4),
        "mean_latency_ms": round(mean, 4),
        "stddev_ms": round(stddev, 4),
        "environment": {
            "os": platform.system(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count()
        },
        "raw_sample_path": "reports/recommendations/latency_samples.csv"
    }
    
    os.makedirs("reports/recommendations", exist_ok=True)
    df = pd.DataFrame({"latency_ms": latencies})
    df.to_csv("reports/recommendations/latency_samples.csv", index=False)
    
    with open("reports/recommendations/benchmark.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print(json.dumps(report, indent=4))
    
if __name__ == "__main__":
    run_latency_benchmark()
