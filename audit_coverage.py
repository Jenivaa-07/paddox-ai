import pandas as pd
import os
import glob
from pathlib import Path

def audit_coverage():
    print("Auditing checkpoint-specific coverage...")
    results_files = glob.glob("data/raw/predictive/*/*/R/results.csv")
    
    records = []
    
    total_results_rows = 0
    total_lap_10 = 0
    total_dist_25 = 0
    total_dist_50 = 0
    total_dist_75 = 0
    
    for f in results_files:
        path = Path(f)
        season = path.parent.parent.parent.name
        rnd_str = path.parent.parent.name
        race_id = f"{season}_{rnd_str}"
        
        try: results = pd.read_csv(f)
        except: continue
        if results.empty: continue
            
        results_drivers = set(results['DriverNumber'].dropna().unique())
        total_results_rows += len(results_drivers)
        
        laps_path = path.parent / "laps.csv"
        try: laps = pd.read_csv(laps_path, low_memory=False)
        except: laps = pd.DataFrame()
        
        eligible_10 = set()
        eligible_25 = set()
        eligible_50 = set()
        eligible_75 = set()
        dropped = []
        
        if not laps.empty:
            laps_clean = laps.dropna(subset=['DriverNumber', 'LapNumber', 'LapTime'])
            if not laps_clean.empty:
                max_lap = int(laps_clean['LapNumber'].max())
                if max_lap >= 10:
                    chk_10 = 10
                    chk_25 = int(round(max_lap * 0.25))
                    chk_50 = int(round(max_lap * 0.50))
                    chk_75 = int(round(max_lap * 0.75))
                    
                    for d, group in laps_clean.groupby('DriverNumber'):
                        d_max = int(group['LapNumber'].max())
                        if d_max >= chk_10: eligible_10.add(d)
                        else: dropped.append((d, "Laps", "DNF before Lap 10"))
                            
                        if d_max >= chk_25: eligible_25.add(d)
                        else: dropped.append((d, "Laps", "DNF before 25%"))
                            
                        if d_max >= chk_50: eligible_50.add(d)
                        else: dropped.append((d, "Laps", "DNF before 50%"))
                            
                        if d_max >= chk_75: eligible_75.add(d)
                        else: dropped.append((d, "Laps", "DNF before 75%"))
                else:
                    for d in results_drivers: dropped.append((d, "Laps", "Race too short (<10 laps)"))
            else:
                for d in results_drivers: dropped.append((d, "Laps", "All laps dropped due to missing LapTime"))
        else:
            for d in results_drivers: dropped.append((d, "Laps", "Laps file empty or unreadable"))
            
        for d in results_drivers:
            if d not in eligible_10 and not any(x[0] == d and x[2] == "DNF before Lap 10" for x in dropped):
                dropped.append((d, "Laps", "Missing from laps entirely"))
                
        total_lap_10 += len(eligible_10)
        total_dist_25 += len(eligible_25)
        total_dist_50 += len(eligible_50)
        total_dist_75 += len(eligible_75)
        
        records.append({
            "race_id": race_id,
            "raw_official_result_records": len(results_drivers),
            "lap_10_eligible_records": len(eligible_10),
            "dist_25_eligible_records": len(eligible_25),
            "dist_50_eligible_records": len(eligible_50),
            "dist_75_eligible_records": len(eligible_75),
            "checkpoint_level_retention_pct": round(((len(eligible_10) + len(eligible_25) + len(eligible_50) + len(eligible_75)) / (len(results_drivers) * 4)) * 100, 2) if results_drivers else 0,
            "dropped_driver_ids": [x[0] for x in dropped],
            "drop_reasons": list(set([x[2] for x in dropped]))
        })
        
    df_audit = pd.DataFrame(records)
    
    os.makedirs("reports/predictive", exist_ok=True)
    df_audit.to_csv("reports/predictive/checkpoint_coverage.csv", index=False)
    
    print(f"Total raw result records: {total_results_rows}")
    print(f"Total sequences generated: {total_lap_10 + total_dist_25 + total_dist_50 + total_dist_75}")
    print("Checkpoint-level coverage generated at reports/predictive/checkpoint_coverage.csv")
    
if __name__ == "__main__":
    audit_coverage()
