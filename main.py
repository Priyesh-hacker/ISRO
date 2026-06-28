from apscheduler.schedulers.blocking import BlockingScheduler
from loguru import logger
from database.db import init_db
from ingestion.noaa_fetcher import fetch_and_store
from config import FETCH_INTERVAL_MINUTES

logger.add("logs/main.log", rotation="5 MB")

def main():
    print("=== ISRO Radiation Forecast System ===")
    print("Initializing database...")
    init_db()

    print("Running initial NOAA fetch...")
    fetch_and_store()

    print(f"Starting scheduler — fetching every {FETCH_INTERVAL_MINUTES} minutes")
    scheduler = BlockingScheduler()
    scheduler.add_job(fetch_and_store, "interval", minutes=FETCH_INTERVAL_MINUTES)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("\nScheduler stopped.")

if __name__ == "__main__":
    main()
