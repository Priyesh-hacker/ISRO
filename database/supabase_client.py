import os
from supabase import create_client, Client
from dotenv import load_dotenv
from loguru import logger
import pandas as pd
from datetime import datetime, timedelta

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(url, key)

def upsert_observations(df: pd.DataFrame):
    if df.empty:
        return
    records = df.to_dict(orient="records")
    try:
        supabase.table("raw_observations").upsert(records).execute()
        logger.info(f"Upserted {len(records)} observations to Supabase")
    except Exception as e:
        logger.error(f"Supabase upsert failed: {e}")

def insert_storm_events(df: pd.DataFrame):
    if df.empty:
        return
    records = df.to_dict(orient="records")
    try:
        supabase.table("storm_events").insert(records).execute()
        logger.info(f"Inserted {len(records)} storm events to Supabase")
    except Exception as e:
        logger.error(f"Storm events insert failed: {e}")

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
        res = supabase.table("raw_observations").select("*").order("timestamp", desc=True).limit(1).execute()
        return res.data[0] if res.data else {}
    except Exception as e:
        logger.error(f"Fetch latest failed: {e}")
        return {}

def fetch_alerts(limit=20) -> pd.DataFrame:
    try:
        res = supabase.table("alert_log").select("*").order("triggered_at", desc=True).limit(limit).execute()
        return pd.DataFrame(res.data)
    except Exception as e:
        logger.error(f"Fetch alerts failed: {e}")
        return pd.DataFrame()

def fetch_all_observations(limit=100000) -> pd.DataFrame:
    try:
        res = supabase.table("raw_observations").select("*").order("timestamp", desc=True).limit(limit).execute()
        df = pd.DataFrame(res.data)
        if not df.empty:
            df = df.sort_values("timestamp")
        return df
    except Exception as e:
        logger.error(f"Fetch observations failed: {e}")
        return pd.DataFrame()

def fetch_storm_events() -> pd.DataFrame:
    try:
        res = supabase.table("storm_events").select("*").eq("event_type", "GST").order("event_time").execute()
        return pd.DataFrame(res.data)
    except Exception as e:
        logger.error(f"Fetch storm events failed: {e}")
        return pd.DataFrame()

def fetch_recent_observations(hours=24) -> pd.DataFrame:
    try:
        # Fetch the most recent records equivalent to 24 hours (assuming 5 min intervals = 12 per hour)
        limit = hours * 12
        res = supabase.table("raw_observations").select("*").order("timestamp", desc=True).limit(limit).execute()
        df = pd.DataFrame(res.data)
        if not df.empty:
            df = df.sort_values("timestamp")
        return df
    except Exception as e:
        logger.error(f"Fetch recent failed: {e}")
        return pd.DataFrame()

def fetch_latest_predictions() -> pd.DataFrame:
    try:
        res = supabase.table("model_predictions").select("*").order("created_at", desc=True).limit(20).execute()
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
