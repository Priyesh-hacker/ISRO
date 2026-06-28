import pandas as pd
import numpy as np
from loguru import logger
import sys
sys.path.append("..")
from config import FORECAST_HORIZONS, THRESHOLDS
from database.db import fetch_observations

logger.add("logs/engineer.log", rotation="1 MB")

def classify_risk(flux_value: float) -> str:
    if pd.isna(flux_value):
        return "Unknown"
    if flux_value >= THRESHOLDS["Severe"]:
        return "Severe"
    elif flux_value >= THRESHOLDS["Storm"]:
        return "Storm"
    elif flux_value >= THRESHOLDS["Elevated"]:
        return "Elevated"
    return "Nominal"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Lag features ---
    for lag in [1, 3, 6, 12, 24]:
        df[f"bz_lag_{lag}h"]           = df["bz"].shift(lag)
        df[f"sw_speed_lag_{lag}h"]     = df["solar_wind_speed"].shift(lag)
        df[f"proton_flux_lag_{lag}h"]  = df["proton_flux"].shift(lag)
        df[f"kp_lag_{lag}h"]           = df["kp_index"].shift(lag)

    # --- Rolling statistics ---
    for window in [3, 6, 12, 24]:
        df[f"bz_mean_{window}h"]           = df["bz"].rolling(window).mean()
        df[f"bz_min_{window}h"]            = df["bz"].rolling(window).min()
        df[f"bz_std_{window}h"]            = df["bz"].rolling(window).std()
        df[f"proton_flux_max_{window}h"]   = df["proton_flux"].rolling(window).max()
        df[f"proton_flux_mean_{window}h"]  = df["proton_flux"].rolling(window).mean()
        df[f"proton_flux_std_{window}h"]   = df["proton_flux"].rolling(window).std()
        df[f"kp_max_{window}h"]            = df["kp_index"].rolling(window).max()
        df[f"sw_speed_mean_{window}h"]     = df["solar_wind_speed"].rolling(window).mean()

    # --- Rate of change ---
    df["bz_roc_3h"]        = df["bz"].diff(3)
    df["bz_roc_6h"]        = df["bz"].diff(6)
    df["sw_speed_roc_6h"]  = df["solar_wind_speed"].diff(6)
    df["proton_roc_3h"]    = df["proton_flux"].diff(3)

    # --- Sustained southward Bz ---
    df["bz_negative"]           = (df["bz"] < 0).astype(int)
    df["bz_sustained_6h"]       = df["bz_negative"].rolling(6).sum()
    df["bz_sustained_12h"]      = df["bz_negative"].rolling(12).sum()
    df["bz_sustained_24h"]      = df["bz_negative"].rolling(24).sum()

    # --- Cyclical time encoding ---
    df["hour_sin"]  = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["doy_sin"]   = np.sin(2 * np.pi * df["timestamp"].dt.dayofyear / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["timestamp"].dt.dayofyear / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.month / 12)

    # --- Target variables ---
    for h in FORECAST_HORIZONS:
        df[f"target_{h}h"]      = df["proton_flux"].shift(-h)
        df[f"risk_{h}h"]        = df[f"target_{h}h"].apply(classify_risk)

    # --- Current risk label ---
    df["current_risk"] = df["proton_flux"].apply(classify_risk)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df

def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = ["timestamp", "source", "current_risk"] + \
              [f"target_{h}h" for h in FORECAST_HORIZONS] + \
              [f"risk_{h}h" for h in FORECAST_HORIZONS]
    return [c for c in df.columns if c not in exclude]

def load_and_engineer() -> pd.DataFrame:
    raw = fetch_observations(limit=100000)
    if raw.empty:
        logger.error("No data in DB — run omniweb_fetcher.py first")
        return pd.DataFrame()
    df = engineer_features(raw)
    df = df.dropna()
    logger.info(f"Final dataset shape: {df.shape}")
    return df

if __name__ == "__main__":
    df = load_and_engineer()
    print(df.shape)
    print(df[["timestamp", "bz", "proton_flux", "target_6h", "risk_6h"]].tail(10))
