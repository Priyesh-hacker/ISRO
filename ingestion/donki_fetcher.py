import requests
import pandas as pd
import sqlite3
from loguru import logger
import sys
sys.path.append("..")
from config import DONKI_BASE_URL, NASA_API_KEY, DB_PATH

logger.add("logs/donki_fetcher.log", rotation="1 MB")

def fetch_geomagnetic_storms(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        r = requests.get(f"{DONKI_BASE_URL}/GST", params={
            "startDate": start_date,
            "endDate": end_date,
            "api_key": NASA_API_KEY
        }, timeout=15)
        data = r.json()
        if not data:
            return pd.DataFrame()

        rows = []
        for event in data:
            rows.append({
                "event_time": event.get("startTime"),
                "event_type": "GST",
                "kp_max": event.get("allKpIndex", [{}])[-1].get("kpIndex") if event.get("allKpIndex") else None,
                "source": "DONKI"
            })
        df = pd.DataFrame(rows)
        logger.info(f"Fetched {len(df)} geomagnetic storm events")
        return df
    except Exception as e:
        logger.error(f"GST fetch failed: {e}")
        return pd.DataFrame()

def fetch_cme_events(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        r = requests.get(f"{DONKI_BASE_URL}/CME", params={
            "startDate": start_date,
            "endDate": end_date,
            "api_key": NASA_API_KEY
        }, timeout=15)
        data = r.json()
        if not data:
            return pd.DataFrame()

        rows = []
        for event in data:
            rows.append({
                "event_time": event.get("startTime"),
                "event_type": "CME",
                "kp_max": None,
                "source": "DONKI"
            })
        df = pd.DataFrame(rows)
        logger.info(f"Fetched {len(df)} CME events")
        return df
    except Exception as e:
        logger.error(f"CME fetch failed: {e}")
        return pd.DataFrame()

def fetch_solar_flares(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        r = requests.get(f"{DONKI_BASE_URL}/FLR", params={
            "startDate": start_date,
            "endDate": end_date,
            "api_key": NASA_API_KEY
        }, timeout=15)
        data = r.json()
        if not data:
            return pd.DataFrame()

        rows = []
        for event in data:
            rows.append({
                "event_time": event.get("beginTime"),
                "event_type": f"FLR_{event.get('classType', 'UNK')}",
                "kp_max": None,
                "source": "DONKI"
            })
        df = pd.DataFrame(rows)
        logger.info(f"Fetched {len(df)} solar flare events")
        return df
    except Exception as e:
        logger.error(f"FLR fetch failed: {e}")
        return pd.DataFrame()

def store_events(df: pd.DataFrame):
    if df.empty:
        return
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("storm_events", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    logger.info(f"Stored {len(df)} events")

def fetch_all_events(start_date="2018-01-01", end_date="2024-12-31"):
    gst = fetch_geomagnetic_storms(start_date, end_date)
    cme = fetch_cme_events(start_date, end_date)
    flr = fetch_solar_flares(start_date, end_date)

    all_events = pd.concat([gst, cme, flr], ignore_index=True)
    store_events(all_events)
    print(f"Total events stored: {len(all_events)}")
    return all_events

if __name__ == "__main__":
    df = fetch_all_events()
    print(df.head(10))
