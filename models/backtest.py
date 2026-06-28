import pandas as pd
import numpy as np
import sqlite3
import sys
sys.path.append("..")
from config import DB_PATH, FORECAST_HORIZONS
from features.engineer import load_and_engineer, get_feature_columns, classify_risk
from models.baseline_xgb import predict_xgb, load_xgb
from loguru import logger

logger.add("logs/backtest.log", rotation="1 MB")

def load_storm_events() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT * FROM storm_events
        WHERE event_type = 'GST'
        ORDER BY event_time
    """, conn)
    conn.close()
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df

def run_backtest():
    print("Loading engineered features...")
    df = load_and_engineer()
    if df.empty:
        print("No data. Run omniweb_fetcher.py first.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("Loading storm events...")
    storms = load_storm_events()
    if storms.empty:
        print("No storm events. Run donki_fetcher.py first.")
        return

    print(f"\nBacktesting on {len(storms)} geomagnetic storm events...\n")

    results = []

    for _, storm in storms.iterrows():
        storm_time = storm["event_time"]

        for h in FORECAST_HORIZONS:
            # Get the data available h hours BEFORE the storm
            lookback_time = storm_time - pd.Timedelta(hours=h)
            window = df[df["timestamp"] <= lookback_time]

            if len(window) < 24:
                continue

            latest_row = window.tail(1)

            try:
                predicted_flux = predict_xgb(latest_row, h)
                predicted_risk = classify_risk(predicted_flux)

                # Actual flux at storm time
                actual_window = df[df["timestamp"] == storm_time]
                actual_flux = actual_window["proton_flux"].values[0] if not actual_window.empty else None
                actual_risk = classify_risk(actual_flux) if actual_flux else "Unknown"

                results.append({
                    "storm_time": storm_time,
                    "forecast_horizon_h": h,
                    "predicted_flux": round(predicted_flux, 4),
                    "predicted_risk": predicted_risk,
                    "actual_flux": round(actual_flux, 4) if actual_flux else None,
                    "actual_risk": actual_risk,
                    "correct_risk": predicted_risk == actual_risk
                })

            except Exception as e:
                logger.warning(f"Backtest failed for storm {storm_time} at {h}h: {e}")

    if not results:
        print("No backtest results generated.")
        return

    results_df = pd.DataFrame(results)

    print("=== Backtest Results ===\n")
    for h in FORECAST_HORIZONS:
        subset = results_df[results_df["forecast_horizon_h"] == h]
        accuracy = subset["correct_risk"].mean() * 100
        print(f"{h}h horizon | Storms: {len(subset)} | Risk accuracy: {accuracy:.1f}%")

    print("\n=== Sample Predictions ===")
    print(results_df[["storm_time", "forecast_horizon_h", "predicted_risk", "actual_risk", "correct_risk"]].head(15).to_string())

    # Save results
    results_df.to_csv("data/backtest_results.csv", index=False)
    print("\nBacktest results saved to data/backtest_results.csv")
    return results_df

if __name__ == "__main__":
    run_backtest()
