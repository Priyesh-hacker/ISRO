import sqlite3
import pandas as pd
from loguru import logger
import sys
sys.path.append("..")
from config import DB_PATH

logger.add("logs/db.log", rotation="1 MB")

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS raw_observations (
            timestamp TEXT PRIMARY KEY,
            bz REAL,
            solar_wind_speed REAL,
            density REAL,
            kp_index REAL,
            proton_flux REAL,
            electron_flux REAL,
            source TEXT
        );

        CREATE TABLE IF NOT EXISTS model_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            forecast_horizon_hrs INTEGER,
            predicted_proton_flux REAL,
            risk_level TEXT,
            confidence REAL
        );

        CREATE TABLE IF NOT EXISTS alert_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            triggered_at TEXT,
            risk_level TEXT,
            predicted_flux REAL,
            lead_time_hrs REAL,
            ollama_explanation TEXT
        );

        CREATE TABLE IF NOT EXISTS storm_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_time TEXT,
            event_type TEXT,
            kp_max REAL,
            source TEXT
        );
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

def insert_observations(df: pd.DataFrame):
    conn = get_connection()
    try:
        df.to_sql("raw_observations", conn, if_exists="append", index=False)
        conn.commit()
        logger.info(f"Inserted {len(df)} observations")
    except Exception as e:
        logger.error(f"Insert failed: {e}")
    finally:
        conn.close()

def fetch_observations(limit: int = 10000) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(f"""
        SELECT * FROM raw_observations
        ORDER BY timestamp DESC
        LIMIT {limit}
    """, conn)
    conn.close()
    return df.sort_values("timestamp")

def insert_prediction(created_at, horizon, flux, risk, confidence):
    conn = get_connection()
    conn.execute("""
        INSERT INTO model_predictions
        (created_at, forecast_horizon_hrs, predicted_proton_flux, risk_level, confidence)
        VALUES (?, ?, ?, ?, ?)
    """, (created_at, horizon, flux, risk, confidence))
    conn.commit()
    conn.close()

def insert_alert(triggered_at, risk_level, predicted_flux, lead_time, explanation):
    conn = get_connection()
    conn.execute("""
        INSERT INTO alert_log
        (triggered_at, risk_level, predicted_flux, lead_time_hrs, ollama_explanation)
        VALUES (?, ?, ?, ?, ?)
    """, (triggered_at, risk_level, predicted_flux, lead_time, explanation))
    conn.commit()
    conn.close()

def fetch_latest_observation() -> dict:
    conn = get_connection()
    cursor = conn.execute("""
        SELECT * FROM raw_observations
        ORDER BY timestamp DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    if row:
        cols = ["timestamp","bz","solar_wind_speed","density","kp_index","proton_flux","electron_flux","source"]
        return dict(zip(cols, row))
    return {}

if __name__ == "__main__":
    init_db()
    print("DB initialized at", DB_PATH)
