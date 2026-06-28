import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from loguru import logger
import sys
sys.path.append("..")
from config import FORECAST_HORIZONS
from features.engineer import load_and_engineer, get_feature_columns

logger.add("logs/xgb_model.log", rotation="1 MB")

def train_xgb(df: pd.DataFrame, horizon: int):
    feature_cols = get_feature_columns(df)
    target_col = f"target_{horizon}h"

    X = df[feature_cols]
    y = df[target_col]

    # Chronological split — never shuffle time series
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )

    model.fit(
        X_train_sc, y_train,
        eval_set=[(X_test_sc, y_test)],
        verbose=50
    )

    preds = model.predict(X_test_sc)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)

    logger.info(f"XGB {horizon}h | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    print(f"\n--- {horizon}h Forecast ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    # Save model and scaler
    with open(f"models/xgb_{horizon}h.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "features": feature_cols}, f)

    logger.info(f"Saved xgb_{horizon}h.pkl")
    return model, scaler, feature_cols, rmse, mae

def train_all():
    print("Loading and engineering features...")
    df = load_and_engineer()
    if df.empty:
        print("No data. Run omniweb_fetcher.py first.")
        return

    results = {}
    for h in FORECAST_HORIZONS:
        model, scaler, features, rmse, mae = train_xgb(df, h)
        results[h] = {"rmse": rmse, "mae": mae}

    print("\n=== XGBoost Summary ===")
    for h, r in results.items():
        print(f"{h}h → RMSE: {r['rmse']:.4f} | MAE: {r['mae']:.4f}")

def load_xgb(horizon: int):
    with open(f"models/xgb_{horizon}h.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"], bundle["features"]

def predict_xgb(row: pd.DataFrame, horizon: int) -> float:
    model, scaler, features = load_xgb(horizon)
    X = row[features]
    X_sc = scaler.transform(X)
    return float(model.predict(X_sc)[0])

if __name__ == "__main__":
    train_all()
