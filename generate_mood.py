# generate_mood.py
from alpha_vantage.timeseries import TimeSeries
import json
import os
import pandas as pd

def get_vix_mood():
    print("--- Inizio recupero dati per Mood Trader Dial ---")
    api_key = os.getenv('RYBBLMHYLWA6HD23')
    if not api_key:
        print("ERRORE: API Key di Alpha Vantage non trovata.")
        return

    try:
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, meta_data = ts.get_daily(symbol='^VIX')

        last_vix_value = data['4. close'].iloc[0]
        last_day = data.index[0]

        print(f"Ultimo valore VIX recuperato: {last_vix_value} in data {last_day}")

        fear_level = max(0, min(1, (last_vix_value - 15) / (50 - 15)))
        mood_score = 100 - (fear_level * 100)

        mood_data = {
            "last_update": str(last_day.date()),
            "vix_value": round(last_vix_value, 2),
            "mood_score": round(mood_score)
        }

        with open('mood.json', 'w') as f:
            json.dump(mood_data, f)

        print(f"File 'mood.json' generato con successo. Punteggio: {mood_data['mood_score']}")

    except Exception as e:
        print(f"ERRORE durante il recupero dati: {e}")
        raise e # Rilancia l'errore per far fallire il workflow e notificarci

if __name__ == '__main__':
    get_vix_mood()
