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
    if df.empty:
        return
    records = df.to_dict(orient="records")
    import math
    for record in records:
        for k, v in record.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                record[k] = None
    try:
        # Batch upsert 1000 rows at a time
        for i in range(0, len(records), 1000): 
            supabase.table("raw_observations").upsert(records[i:i+1000]).execute()
        logger.info(f"Upserted {len(records)} observations to Supabase")
    except Exception as e:
        logger.error(f"Supabase upsert failed: {e}")

def insert_storm_events(df: pd.DataFrame):
    if df.empty:
        return
    records = df.to_dict(orient="records")
    import math
    for record in records:
        for k, v in record.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                record[k] = None
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
    all_data = []
    start = 0
    step = 1000
    while True:
        try:
            # Safely paginate to fetch massive amounts of data from Supabase
            res = supabase.table("raw_observations").select("*").order("timestamp", desc=True).range(start, start + step - 1).execute()
            data = res.data
            if not data:
                break
            all_data.extend(data)
            start += step
            if len(all_data) >= limit:
                break
        except Exception as e:
            logger.error(f"Fetch observations failed: {e}")
            break
    
    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.sort_values("timestamp")
    return df

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

def fetch_historical_bz_kp(limit=744) -> pd.DataFrame:
    """Fetch OMNIWEB historical observations (Bz & Kp source) ordered by timestamp.

    OMNIWEB data runs from 2018-2023 and is the only source with complete
    hourly Bz and Kp coverage. Returns the most recent ``limit`` rows from
    the OMNIWEB_sourced records so the charts show a meaningful window.
    Default 744 = 31 days of hourly data.
    """
    try:
        res = (
            supabase.table("raw_observations")
            .select("timestamp,bz,kp_index,solar_wind_speed")
            .eq("source", "OMNIWEB")
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        df = pd.DataFrame(res.data)
        if not df.empty:
            df = df.sort_values("timestamp")
        return df
    except Exception as e:
        logger.error(f"Fetch historical Bz/Kp failed: {e}")
        return pd.DataFrame()

def fetch_noaa_realtime(limit=500) -> pd.DataFrame:
    """Fetch the most recent NOAA_SWPC observations (proton flux / solar wind).

    NOAA real-time data is stored every ~5 minutes. Returns the most recent
    ``limit`` rows so the Proton Flux chart shows the last ~42 hours.
    """
    try:
        res = (
            supabase.table("raw_observations")
            .select("timestamp,proton_flux,solar_wind_speed,density,bz,kp_index")
            .eq("source", "NOAA_SWPC")
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        df = pd.DataFrame(res.data)
        if not df.empty:
            df = df.sort_values("timestamp")
        return df
    except Exception as e:
        logger.error(f"Fetch NOAA realtime failed: {e}")
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
