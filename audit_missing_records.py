import pandas as pd
import json
import os
import glob
from pathlib import Path

def audit_missing_records():
    print("Auditing missing result records...")
    
    # We want to iterate over all results.csv files
    results_files = glob.glob("data/raw/predictive/*/*/R/results.csv")
    
    records = []
    
    for f in results_files:
        path = Path(f)
        season = path.parent.parent.parent.name
        rnd_str = path.parent.parent.name
        race_id = f"{season}_{rnd_str}"
        
        try:
            results = pd.read_csv(f)
        except Exception:
            continue
            
        if results.empty:
            continue
            
        # raw drivers in results
        results_drivers = set(results['DriverNumber'].dropna().unique())
        field_size = len(results)
        
        laps_path = path.parent / "laps.csv"
        try:
            laps = pd.read_csv(laps_path, low_memory=False)
            laps_drivers = set(laps['DriverNumber'].dropna().unique())
        except Exception:
            laps_drivers = set()
            
        # Count checkpoints
        eligible_chk = set()
        dropped = []
        
        # Mimic build_race_sequences
        # laps dropping
        if not laps.empty:
            laps_clean = laps.dropna(subset=['DriverNumber', 'LapNumber', 'LapTime'])
            if not laps_clean.empty:
                max_lap = laps_clean['LapNumber'].max()
                if max_lap >= 10:
                    for d, group in laps_clean.groupby('DriverNumber'):
                        if group['LapNumber'].max() >= 10:
                            eligible_chk.add(d)
                        else:
                            dropped.append((d, "Laps", "DNF before lap 10"))
                else:
                    for d in results_drivers:
                        dropped.append((d, "Laps", "Race too short (<10 laps)"))
            else:
                for d in results_drivers:
                    dropped.append((d, "Laps", "All laps dropped due to missing LapTime"))
        else:
            for d in results_drivers:
                dropped.append((d, "Laps", "Laps file empty or unreadable"))
                
        # Drop logic for fantasy: none explicitly in process_race, but let's check
        # results rows with no DriverNumber are dropped natively (but shouldn't exist)
        
        for d in results_drivers:
            if d not in eligible_chk and not any(x[0] == d for x in dropped):
                dropped.append((d, "Laps", "Missing from laps or DNF before lap 10"))
                
        # For fantasy targets, they should all be included.
        fantasy_target_count = len(results_drivers)
        
        records.append({
            "race_id": race_id,
            "event_name": race_id, # not in results, approximation
            "expected_field_size": field_size,
            "results_row_count": len(results),
            "unique_drivers_results": len(results_drivers),
            "unique_drivers_laps": len(laps_drivers),
            "unique_drivers_features": len(results_drivers), # we don't drop on feature joins
            "eligible_checkpoint_drivers": len(eligible_chk),
            "fantasy_target_drivers": fantasy_target_count,
            "dropped_driver_ids": [x[0] for x in dropped],
            "drop_stage": "Laps",
            "drop_reason": "Various lap omissions" if dropped else "None"
        })
        
    df_audit = pd.DataFrame(records)
    
    os.makedirs("reports/predictive", exist_ok=True)
    df_audit.to_csv("reports/predictive/driver_coverage_by_stage.csv", index=False)
    
    print(f"Total results records: {df_audit['results_row_count'].sum()}")
    print(f"Total fantasy records simulated: {df_audit['fantasy_target_drivers'].sum()}")
    
if __name__ == "__main__":
    audit_missing_records()
