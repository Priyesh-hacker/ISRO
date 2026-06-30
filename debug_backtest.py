import os
import sys
import pandas as pd
import numpy as np

print("Starting debug script...", flush=True)
sys.path.append(".")
from config import FORECAST_HORIZONS
from database.supabase_client import fetch_storm_events
from features.engineer import load_and_engineer
from models.baseline_xgb import load_xgb

def run():
    print("Loading engineered features...", flush=True)
    df = load_and_engineer()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ts_int"] = df["timestamp"].astype('int64') // 10**9
    df = df.sort_values("ts_int").reset_index(drop=True)
    ts_array = df["ts_int"].values

    # Print the database date range so we can inspect it
    if len(df) > 0:
        print(f"Database range: {df['timestamp'].min()} to {df['timestamp'].max()}", flush=True)

    print("Loading storm events...", flush=True)
    storms = fetch_storm_events()
    storms["event_time"] = pd.to_datetime(storms["event_time"], utc=True)
    storms["ts_int"] = storms["event_time"].astype('int64') // 10**9

    print("Loading models...", flush=True)
    models = {}
    for h in FORECAST_HORIZONS:
        try:
            models[h] = load_xgb(h)
            print(f"Loaded {h}h model successfully", flush=True)
        except Exception as e:
            print(f"Failed to load {h}h: {e}", flush=True)

    print(f"\nBacktesting on {len(storms)} storms...", flush=True)

    for h in FORECAST_HORIZONS:
        if h not in models: continue
        model, scaler, features = models[h]
        print(f"\nEvaluating horizon {h}h...", flush=True)
        
        batch_X = []
        
        for idx, storm in storms.iterrows():
            print(f"  Storm {idx}: {storm['event_time']}", flush=True)
            storm_ts = storm["ts_int"]
            lookback_ts = storm_ts - (h * 3600)
            
            search_idx = np.searchsorted(ts_array, lookback_ts, side='right')
            if search_idx < 24:
                print(f"    Skipping: Not enough history (search_idx={search_idx})", flush=True)
                continue
                
            # SAFE SLICE FIX: Bypasses the Pandas 2.2 fancy-indexing segfault
            latest_row = df.iloc[search_idx - 1 : search_idx]
            X = latest_row[features]
            
            print(f"    Transforming features for storm {idx}...", flush=True)
            try:
                X_sc = scaler.transform(X)
                batch_X.append(X_sc[0])
                print(f"    Transform successful.", flush=True)
            except Exception as e:
                print(f"    Transform FAILED: {e}", flush=True)
                
        if not batch_X:
            print(f"  No batches collected for {h}h", flush=True)
            continue
            
        print(f"  Predicting batch of size {len(batch_X)}...", flush=True)
        X_matrix = np.array(batch_X)
        try:
            preds = model.predict(X_matrix)
            print(f"  Predict successful.", flush=True)
        except Exception as e:
            print(f"  Predict FAILED: {e}", flush=True)

    print("\nFinished successfully.", flush=True)

if __name__ == "__main__":
    run()
