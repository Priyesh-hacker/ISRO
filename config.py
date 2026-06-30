import os
from dotenv import load_dotenv

load_dotenv()

# NASA API
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")

# Database
DB_PATH = "data/radiation.db"

# NOAA SWPC endpoints
NOAA_SOLAR_WIND_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
NOAA_KP_URL = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
NOAA_PROTON_URL = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json"
NOAA_ELECTRON_URL = "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-1-day.json"

# NASA DONKI endpoints
DONKI_BASE_URL = "https://api.nasa.gov/DONKI"

# NASA OMNIWeb
OMNIWEB_URL = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

# Ollama
OLLAMA_MODEL = "qwen2.5:3b"

# Model settings
SEQ_LEN = 24           # hours of history fed into LSTM
FORECAST_HORIZONS = [6, 12, 24]   # hours ahead to predict

# NOAA radiation thresholds (proton flux in pfu)
THRESHOLDS = {
    "Nominal":  1,
    "Elevated": 10,
    "Storm":    100,
    "Severe":   1000
}

# Fetch interval (minutes)
FETCH_INTERVAL_MINUTES = 5

# Historical data range for training
TRAIN_START_YEAR = 2018
TRAIN_END_YEAR = 2023
