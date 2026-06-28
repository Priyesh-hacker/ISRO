import ollama
from loguru import logger
import sys
sys.path.append("..")
from config import OLLAMA_MODEL, THRESHOLDS
from features.engineer import classify_risk

logger.add("logs/explainer.log", rotation="1 MB")

SYSTEM_PROMPT = """You are a space weather analyst assistant embedded in ISRO's satellite operations center.
Your job is to interpret radiation forecasts and provide clear, actionable guidance to satellite operators.
Be concise, technical but readable, and always end with a specific recommendation.
Never use bullet points. Write in 3-4 sentences maximum."""

def build_prompt(conditions: dict, predictions: dict) -> str:
    return f"""
Current space weather conditions:
- Interplanetary Magnetic Field Bz: {conditions.get('bz', 'N/A')} nT
- Solar wind speed: {conditions.get('solar_wind_speed', 'N/A')} km/s
- Solar wind density: {conditions.get('density', 'N/A')} protons/cm³
- Kp geomagnetic index: {conditions.get('kp_index', 'N/A')}
- Current proton flux: {conditions.get('proton_flux', 'N/A')} pfu

Radiation forecast for ISRO geostationary satellites:
- 6-hour forecast:  {predictions.get('6h', {}).get('flux', 'N/A')} pfu → Risk: {predictions.get('6h', {}).get('risk', 'N/A')}
- 12-hour forecast: {predictions.get('12h', {}).get('flux', 'N/A')} pfu → Risk: {predictions.get('12h', {}).get('risk', 'N/A')}
- 24-hour forecast: {predictions.get('24h', {}).get('flux', 'N/A')} pfu → Risk: {predictions.get('24h', {}).get('risk', 'N/A')}

Explain what is driving these conditions, assess the threat level for geostationary satellite electronics,
and give a specific operational recommendation (e.g. safe mode timing, payload adjustments).
"""

def explain_forecast(conditions: dict, predictions: dict) -> str:
    prompt = build_prompt(conditions, predictions)
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        explanation = response["message"]["content"]
        logger.info("Explanation generated successfully")
        return explanation
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return f"Explanation unavailable: {e}"

def format_predictions(pred_6h: float, pred_12h: float, pred_24h: float) -> dict:
    return {
        "6h":  {"flux": round(pred_6h, 4),  "risk": classify_risk(pred_6h)},
        "12h": {"flux": round(pred_12h, 4), "risk": classify_risk(pred_12h)},
        "24h": {"flux": round(pred_24h, 4), "risk": classify_risk(pred_24h)},
    }

if __name__ == "__main__":
    # Quick test
    conditions = {
        "bz": -14.2,
        "solar_wind_speed": 680,
        "density": 9.1,
        "kp_index": 6.3,
        "proton_flux": 0.08
    }
    predictions = format_predictions(0.45, 2.3, 8.7)
    result = explain_forecast(conditions, predictions)
    print("\n=== Ollama Explanation ===")
    print(result)
