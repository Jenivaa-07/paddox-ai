import os
import glob
import json
import pandas as pd
import numpy as np

def process_race(year, rnd, session_dir):
    laps_path = f"{session_dir}/laps.csv"
    results_path = f"{session_dir}/results.csv"
    if not os.path.exists(results_path):
        return [], []
        
    results = pd.read_csv(results_path)
    if results.empty:
        return [], []
        
    laps = pd.DataFrame()
    if os.path.exists(laps_path):
        try:
            laps = pd.read_csv(laps_path, low_memory=False)
        except Exception:
            pass
            
    race_sequences = []
    
    field_size = len(results)
    
    # Process sequences only if laps are valid
    if not laps.empty:
        laps = laps.dropna(subset=['DriverNumber', 'LapNumber', 'LapTime'])
        if not laps.empty:
            laps['LapTime_ms'] = pd.to_timedelta(laps['LapTime']).dt.total_seconds() * 1000
            total_laps = int(laps['LapNumber'].max())
            if total_laps >= 10:
                # Checkpoints calculation
                chk_10 = 10
                chk_25 = int(round(total_laps * 0.25))
                chk_50 = int(round(total_laps * 0.50))
                chk_75 = int(round(total_laps * 0.75))
                
                checkpoints = {}
                for label, target_lap in [("lap_10", chk_10), ("dist_25", chk_25), ("dist_50", chk_50), ("dist_75", chk_75)]:
                    if target_lap not in checkpoints:
                        checkpoints[target_lap] = []
                    checkpoints[target_lap].append(label)
                    
                for driver_id, group in laps.groupby('DriverNumber'):
                    group = group.sort_values('LapNumber')
                    race_info = results[results['DriverNumber'] == driver_id]
                    if race_info.empty:
                        continue
                        
                    final_position = race_info['Position'].iloc[0] if not pd.isna(race_info['Position'].iloc[0]) else field_size
                    grid_position = race_info['GridPosition'].iloc[0] if not pd.isna(race_info['GridPosition'].iloc[0]) else field_size
                    status = race_info['Status'].iloc[0] if not pd.isna(race_info['Status'].iloc[0]) else 'Unknown'
                    
                    for chk_lap, labels in checkpoints.items():
                        completed_laps = group[group['LapNumber'] <= chk_lap]
                        if completed_laps.empty:
                            continue
                            
                        actual_chk_lap = completed_laps['LapNumber'].max()
                        if actual_chk_lap < chk_lap:
                            continue
                            
                        window = completed_laps.iloc[-10:]
                        seq_laps = window['LapTime_ms'].tolist()
                        mask = [1] * len(seq_laps)
                        while len(seq_laps) < 10:
                            seq_laps.insert(0, 0.0)
                            mask.insert(0, 0)
                            
                        for label in labels:
                            race_sequences.append({
                                'season': year,
                                'round': rnd,
                                'race_id': f"{year}_{rnd}",
                                'driver_id': driver_id,
                                'DriverNumber': driver_id,
                                'checkpoint_type': label,
                                'checkpoint_lap': actual_chk_lap,
                                'current_lap': actual_chk_lap,
                                'grid_position': grid_position,
                                'field_size': field_size,
                                'seq_laps': seq_laps,
                                'mask': mask,
                                'final_position': final_position,
                                'top10': 1.0 if float(final_position) <= 10 else 0.0,
                                'status': status
                            })
                            
    # For fantasy, we process regardless of laps validity
    fantasy_rows = []
    for _, row in results.iterrows():
        pos = row.get('Position', field_size)
        grid = row.get('GridPosition', field_size)
        driver_id = row.get('DriverNumber')
        constructor_id = row.get('TeamId', row.get('TeamName', 'Unknown'))
        status = str(row.get('Status', 'Unknown'))
        
        fantasy_rows.append({
            'season': year,
            'round': rnd,
            'DriverNumber': driver_id,
            'constructor_id': constructor_id,
            'qualifying_position': grid,
            'final_position': pos,
            'status': status,
            'field_size': field_size
        })
        
    return race_sequences, fantasy_rows

def build_features(raw_dir, out_dir):
    print("Building sequences and fantasy features across all fetched races...")
    
    all_seqs = []
    all_fantasy = []
    
    # Traverse raw directories
    for year_dir in glob.glob(f"{raw_dir}/*"):
        if not os.path.isdir(year_dir): continue
        year = int(os.path.basename(year_dir))
        
        for rnd_dir in glob.glob(f"{year_dir}/*"):
            if not os.path.isdir(rnd_dir): continue
            rnd = int(os.path.basename(rnd_dir))
            
            race_dir = f"{rnd_dir}/R"
            manifest = f"{race_dir}/completion_manifest.json"
            if os.path.exists(manifest):
                seqs, fant = process_race(year, rnd, race_dir)
                all_seqs.extend(seqs)
                all_fantasy.extend(fant)
                
    seq_df = pd.DataFrame(all_seqs)
    fant_df = pd.DataFrame(all_fantasy)
    
    if seq_df.empty or fant_df.empty:
        print("No valid race data found to process.")
        return
        
    # Calculate rolling_avg_finish globally without leakage
    # Sort by season, round chronologically
    fant_df = fant_df.sort_values(['season', 'round'])
    fant_df['rolling_avg_finish'] = (
        fant_df.groupby('DriverNumber')['final_position']
        .apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    # Default to 10.0 or mid-pack if missing
    fant_df['rolling_avg_finish'] = fant_df['rolling_avg_finish'].fillna(10.0)
    
    # Map rolling_avg_finish into seq_df
    seq_df = pd.merge(
        seq_df, 
        fant_df[['season', 'round', 'DriverNumber', 'rolling_avg_finish']], 
        on=['season', 'round', 'DriverNumber'], 
        how='left'
    )
    
    # Split rules: Train: 2022-2024, Val: 2025, Test: 2026
    train_cond = seq_df['season'].between(2022, 2024)
    val_cond = seq_df['season'] == 2025
    test_cond = seq_df['season'] == 2026
    
    # Check if 2025/2026 exist
    has_val = seq_df[val_cond].shape[0] > 0
    has_test = seq_df[test_cond].shape[0] > 0
    
    if not has_val or not has_test:
        print("2025/2026 data unavailable. Falling back to chronological 70/15/15 race-level split.")
        races = seq_df[['season', 'round']].drop_duplicates().sort_values(['season', 'round'])
        n_races = len(races)
        train_races = races.iloc[:int(n_races*0.7)]
        val_races = races.iloc[int(n_races*0.7):int(n_races*0.85)]
        test_races = races.iloc[int(n_races*0.85):]
        
        def apply_split(df):
            tr = df.merge(train_races, on=['season', 'round'])
            v = df.merge(val_races, on=['season', 'round'])
            te = df.merge(test_races, on=['season', 'round'])
            return tr, v, te
            
        train_seq, val_seq, test_seq = apply_split(seq_df)
        train_fant, val_fant, test_fant = apply_split(fant_df)
    else:
        train_seq, val_seq, test_seq = seq_df[train_cond], seq_df[val_cond], seq_df[test_cond]
        train_fant, val_fant, test_fant = fant_df[train_cond], fant_df[val_cond], fant_df[test_cond]
        
    os.makedirs(out_dir, exist_ok=True)
    
    # Save sequences
    train_seq.to_csv(f"{out_dir}/train_race_seq.csv", index=False)
    val_seq.to_csv(f"{out_dir}/val_race_seq.csv", index=False)
    test_seq.to_csv(f"{out_dir}/test_race_seq.csv", index=False)
    
    # Save raw fantasy before targets are built
    train_fant.to_csv(f"{out_dir}/train_fantasy_raw.csv", index=False)
    val_fant.to_csv(f"{out_dir}/val_fantasy_raw.csv", index=False)
    test_fant.to_csv(f"{out_dir}/test_fantasy_raw.csv", index=False)

if __name__ == "__main__":
    build_features('data/raw/predictive', 'data/processed/predictive')
