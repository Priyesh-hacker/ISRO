import requests
import pandas as pd
from datetime import datetime, timezone
from loguru import logger
import sys
sys.path.append("..")
from config import (
    NOAA_SOLAR_WIND_URL, NOAA_MAG_URL, NOAA_KP_URL,
    NOAA_PROTON_URL, NOAA_ELECTRON_URL
)
from database.supabase_client import upsert_observations

logger.add("logs/noaa_fetcher.log", rotation="1 MB")

def fetch_solar_wind() -> dict:
    """Fetch plasma parameters (speed, density) from the RTSW wind endpoint."""
    try:
        data = requests.get(NOAA_SOLAR_WIND_URL, timeout=10).json()
        latest = data[-1] if data else {}
        return {
            "solar_wind_speed": latest.get("proton_speed"),
            "density": latest.get("proton_density"),
            "timestamp": latest.get("time_tag")
        }
    except Exception as e:
        logger.error(f"Solar wind fetch failed: {e}")
        return {}

def fetch_mag() -> dict:
    """Fetch Bz GSM from the RTSW magnetic field endpoint (separate from plasma)."""
    try:
        data = requests.get(NOAA_MAG_URL, timeout=10).json()
        latest = data[-1] if data else {}
        return {
            "bz": latest.get("bz_gsm")
        }
    except Exception as e:
        logger.error(f"Magnetic field (Bz) fetch failed: {e}")
        return {}

def fetch_kp() -> dict:
    try:
        data = requests.get(NOAA_KP_URL, timeout=10).json()
        latest = data[-1] if data else {}
        return {
            "kp_index": latest.get("kp_index"),
            "timestamp": latest.get("time_tag")
        }
    except Exception as e:
        logger.error(f"Kp fetch failed: {e}")
        return {}

def fetch_proton_flux() -> dict:
    try:
        data = requests.get(NOAA_PROTON_URL, timeout=10).json()
        latest = data[-1] if data else {}
        return {
            "proton_flux": latest.get("flux"),
            "timestamp": latest.get("time_tag")
        }
    except Exception as e:
        logger.error(f"Proton flux fetch failed: {e}")
        return {}

def fetch_electron_flux() -> dict:
    try:
        data = requests.get(NOAA_ELECTRON_URL, timeout=10).json()
        latest = data[-1] if data else {}
        return {
            "electron_flux": latest.get("flux"),
            "timestamp": latest.get("time_tag")
        }
    except Exception as e:
        logger.error(f"Electron flux fetch failed: {e}")
        return {}

def fetch_and_store():
    sw = fetch_solar_wind()
    mag = fetch_mag()
    kp = fetch_kp()
    pf = fetch_proton_flux()
    ef = fetch_electron_flux()

    if not sw.get("timestamp"):
        logger.warning("No timestamp from solar wind — skipping")
        return

    row = {
        "timestamp": sw.get("timestamp"),
        "bz": mag.get("bz"),            # Bz from separate mag endpoint
        "solar_wind_speed": sw.get("solar_wind_speed"),
        "density": sw.get("density"),
        "kp_index": kp.get("kp_index"),
        "proton_flux": pf.get("proton_flux"),
        "electron_flux": ef.get("electron_flux"),
        "source": "NOAA_SWPC"
    }

    df = pd.DataFrame([row])
    upsert_observations(df)
    logger.info(f"Stored NOAA observation at {row['timestamp']} | Bz={row['bz']} nT | Kp={row['kp_index']}")

if __name__ == "__main__":
    fetch_and_store()
    print("NOAA fetch complete")
