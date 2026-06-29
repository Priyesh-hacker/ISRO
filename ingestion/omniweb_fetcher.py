import requests
import pandas as pd
from io import StringIO
from loguru import logger
import sys
sys.path.append("..")
from config import OMNIWEB_URL, TRAIN_START_YEAR, TRAIN_END_YEAR
from database.supabase_client import upsert_observations

logger.add("logs/omniweb_fetcher.log", rotation="1 MB")

# OMNIWeb variable codes
# 13 = Bz GSM (nT)
# 24 = Solar wind speed (km/s)
# 23 = Proton density (n/cc)
# 40 = Kp index
# 41 = Proton flux >10 MeV
# 43 = Electron flux >2 MeV

OMNI_VARS = ["13", "24", "23", "40", "41", "43"]
COL_NAMES = ["year", "doy", "hour", "bz", "solar_wind_speed", "density", "kp_index", "proton_flux", "electron_flux"]

# Fill values used by OMNIWeb for missing data
FILL_VALUES = {
    "bz": 9999.99,
    "solar_wind_speed": 99999.9,
    "density": 999.99,
    "kp_index": 99,
    "proton_flux": 99999.99,
    "electron_flux": 99999.99
}

def fetch_year(year: int) -> pd.DataFrame:
    params = {
        "activity": "retrieve",
        "res": "hour",
        "spacecraft": "omni2",
        "start_date": f"{year}0101",
        "end_date": f"{year}1231",
        "vars": OMNI_VARS,
        "submit": "Submit"
    }

    try:
        r = requests.post(OMNIWEB_URL, data=params, timeout=60)
        lines = [l.strip() for l in r.text.split('\n')
                 if l.strip() and not l.startswith('<') and not l.startswith('#')]

        if not lines:
            logger.warning(f"No data returned for {year}")
            return pd.DataFrame()

        df = pd.read_csv(
            StringIO('\n'.join(lines)),
            sep=r'\s+',
            names=COL_NAMES,
            on_bad_lines='skip'
        )

        # Replace fill values with NaN
        for col, fill in FILL_VALUES.items():
            if col in df.columns:
                df[col] = df[col].replace(fill, float('nan'))

        # Build proper timestamp
        df["timestamp"] = pd.to_datetime(
            df["year"].astype(str) + df["doy"].astype(str).str.zfill(3) + df["hour"].astype(str).str.zfill(2),
            format="%Y%j%H",
            errors="coerce"
        ).astype(str)

        df["source"] = "OMNIWEB"
        df = df[["timestamp", "bz", "solar_wind_speed", "density", "kp_index", "proton_flux", "electron_flux", "source"]]
        df = df.dropna(subset=["timestamp"])

        logger.info(f"Fetched {len(df)} rows for {year}")
        return df

    except Exception as e:
        logger.error(f"OMNIWeb fetch failed for {year}: {e}")
        return pd.DataFrame()

def fetch_historical(start_year=TRAIN_START_YEAR, end_year=TRAIN_END_YEAR):
    all_dfs = []
    for year in range(start_year, end_year + 1):
        print(f"Fetching {year}...")
        df = fetch_year(year)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        logger.error("No historical data fetched")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"])

    # Store in DB
    upsert_observations(combined)
    logger.info(f"Total historical rows stored: {len(combined)}")
    print(f"\nDone. Total rows: {len(combined)}")
    return combined

if __name__ == "__main__":
    df = fetch_historical()
    print(df.describe())
    print(df.head())
