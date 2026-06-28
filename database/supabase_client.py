import os
from supabase import create_client, Client
from dotenv import load_dotenv
from loguru import logger
import pandas as pd

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(url, key)

def upsert_observations(df: pd.DataFrame):
    records = df.to_dict(orient="records")
    try:
        supabase.table("raw_observations").upsert(records).execute()
        logger.info(f"Upserted {len(records)} observations to Supabase")
    except Exception as e:
        logger.error(f"Supabase upsert failed: {e}")

def insert_prediction(created_at, horizon, flux, risk, confidence, model_type="XGBoost"):
    try:
        supabase.table("model_predictions").insert({
            "created_at": created_at,
            "forecast_horizon_hrs": horizon,
            "predicted_proton_flux": flux,
            "risk_level": risk,
            "confidence": confidence,
            "model_type": model_type
        }).execute()
    except Exception as e:
        logger.error(f"Prediction insert failed: {e}")

def insert_alert(triggered_at, risk_level, predicted_flux, lead_time, explanation):
    try:
        supabase.table("alert_log").insert({
            "triggered_at": triggered_at,
            "risk_level": risk_level,
            "predicted_flux": predicted_flux,
            "lead_time_hrs": lead_time,
            "ollama_explanation": explanation
        }).execute()
    except Exception as e:
        logger.error(f"Alert insert failed: {e}")

def fetch_latest_observation() -> dict:
    try:
        res = supabase.rpc("get_latest_observation").execute()
        return res.data[0] if res.data else {}
    except Exception as e:
        logger.error(f"Fetch latest failed: {e}")
        return {}

def fetch_recent_observations(hours=24) -> pd.DataFrame:
    try:
        res = supabase.table("recent_observations").select("*").execute()
        return pd.DataFrame(res.data)
    except Exception as e:
        logger.error(f"Fetch recent failed: {e}")
        return pd.DataFrame()

def fetch_latest_predictions() -> pd.DataFrame:
    try:
        res = supabase.table("latest_predictions").select("*").execute()
        return pd.DataFrame(res.data)
    except Exception as e:
        logger.error(f"Fetch predictions failed: {e}")
        return pd.DataFrame()

def subscribe_realtime(callback):
    supabase.realtime.on(
        "postgres_changes",
        event="INSERT",
        schema="public",
        table="raw_observations",
        callback=callback
    ).subscribe()