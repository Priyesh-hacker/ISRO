import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import sys
sys.path.append("..")
from config import FORECAST_HORIZONS
from database.supabase_client import fetch_storm_events
from features.engineer import load_and_engineer, get_feature_columns, classify_risk
from models.baseline_xgb import load_xgb
from loguru import logger

logger.add("logs/backtest.log", rotation="1 MB")

def load_storm_events() -> pd.DataFrame:
    df = fetch_storm_events()
    if not df.empty:
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df

def run_backtest():
    print("Loading engineered features...")
    df = load_and_engineer()
    if df.empty:
        print("No data. Run omniweb_fetcher.py first.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ts_int"] = df["timestamp"].astype('int64') // 10**9
    
    # Sort and reset index for binary search
    df = df.sort_values("ts_int").reset_index(drop=True)
    ts_array = df["ts_int"].values

    print("Loading storm events...")
    storms = load_storm_events()
    if storms.empty:
        print("No storm events. Run donki_fetcher.py first.")
        return
        
    storms["event_time"] = pd.to_datetime(storms["event_time"], utc=True)
    storms["ts_int"] = storms["event_time"].astype('int64') // 10**9

    print("Loading models safely...")
    models = {}
    for h in FORECAST_HORIZONS:
        try:
            models[h] = load_xgb(h)
        except Exception as e:
            print(f"Skipping {h}h: {e}")

    print(f"\nBacktesting on {len(storms)} geomagnetic storm events (Batch Mode)...\n")

    results = []

    for h in FORECAST_HORIZONS:
        if h not in models:
            continue
            
        model, scaler, features = models[h]
        
        batch_X = []
        batch_meta = []
        
        for _, storm in storms.iterrows():
            storm_ts = storm["ts_int"]
            storm_time = storm["event_time"]
            
            lookback_ts = storm_ts - (h * 3600)
            
            # Binary search
            idx = np.searchsorted(ts_array, lookback_ts, side='right')
            
            if idx < 24:
                continue

            # SAFE SLICE FIX
            latest_row = df.iloc[idx - 1 : idx]
            X = latest_row[features]
            X_sc = scaler.transform(X)
            
            batch_X.append(X_sc[0])
            batch_meta.append((storm_time, storm_ts))
            
        if not batch_X:
            continue
            
        # Batch Predict
        X_matrix = np.array(batch_X)
        preds = model.predict(X_matrix)
        
        for i in range(len(preds)):
            storm_time, storm_ts = batch_meta[i]
            predicted_flux = float(preds[i])
            predicted_risk = classify_risk(predicted_flux)

            # Actual flux at storm time (Safe slice + Python float coercion)
            actual_idx = np.searchsorted(ts_array, storm_ts, side='left')
            actual_flux = None
            if actual_idx < len(ts_array) and ts_array[actual_idx] == storm_ts:
                actual_row = df.iloc[actual_idx : actual_idx + 1]
                actual_flux = float(actual_row["proton_flux"].values[0])

            actual_risk = classify_risk(actual_flux) if actual_flux is not None else "Unknown"

            results.append({
                "storm_time": str(storm_time),
                "forecast_horizon_h": int(h),
                "predicted_flux": round(predicted_flux, 4),
                "predicted_risk": str(predicted_risk),
                "actual_flux": round(actual_flux, 4) if actual_flux is not None else None,
                "actual_risk": str(actual_risk),
                "correct_risk": bool(predicted_risk == actual_risk)
            })

    if not results:
        print("No backtest results generated.")
        return

    # PURE PYTHON REPORTING: Bypasses Pandas entirely to prevent C-API Segfaults
    print("=== Backtest Results ===\n")
    for h in FORECAST_HORIZONS:
        horizon_results = [r for r in results if r["forecast_horizon_h"] == h]
        if horizon_results:
            correct = sum(1 for r in horizon_results if r["correct_risk"])
            total = len(horizon_results)
            accuracy = (correct / total) * 100
            print(f"{h}h horizon | Storms: {total} | Risk accuracy: {accuracy:.1f}%")

    print("\n=== Sample Predictions ===")
    print(f"{'Storm Time':<25} | {'Horizon':<8} | {'Predicted Risk':<15} | {'Actual Risk':<15} | {'Correct':<8}")
    print("-" * 80)
    for r in results[:15]:
        print(f"{r['storm_time']:<25} | {r['forecast_horizon_h']:<8} | {r['predicted_risk']:<15} | {r['actual_risk']:<15} | {str(r['correct_risk']):<8}")

    # PURE PYTHON CSV WRITER: Safe from PyArrow memory errors
    import csv
    os.makedirs("data", exist_ok=True)
    with open("data/backtest_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["storm_time", "forecast_horizon_h", "predicted_flux", "predicted_risk", "actual_flux", "actual_risk", "correct_risk"])
        writer.writeheader()
        writer.writerows(results)
        
    print("\nBacktest results saved to data/backtest_results.csv")

if __name__ == "__main__":
    run_backtest()
    os._exit(0)
