# generate_mood.py - Versione Definitiva v2, basata su yfinance per massima affidabilità

import yfinance as yf
import json
import os
import pandas as pd

def get_vix_mood():
    """
    Recupera l'ultimo valore del VIX usando yfinance (più affidabile per gli indici),
    lo normalizza in una scala 0-100 (Fear/Greed) e salva il risultato.
    """
    print("--- Inizio recupero dati per Mood Trader Dial (metodo yfinance) ---")
    
    try:
        # yfinance è lo strumento migliore per scaricare dati di indici come VIX e S&P 500
        ticker = "^VIX"
        vix_data = yf.download(ticker, period="5d", progress=False) # Scarichiamo solo gli ultimi giorni
        
        if vix_data.empty:
            raise ValueError(f"Nessun dato scaricato per il ticker {ticker}")

        # Estrai l'ultimo valore di chiusura e la data
        last_vix_value = vix_data['Close'].iloc[-1]
        last_day = vix_data.index[-1]
        
        print(f"Ultimo valore VIX recuperato: {last_vix_value:.2f} in data {last_day.date()}")

        # Logica di normalizzazione (100 = Greed, 0 = Fear)
        fear_level = max(0, min(1, (last_vix_value - 15) / (50 - 15)))
        mood_score = 100 - (fear_level * 100)
        
        mood_data = {
            "last_update": str(last_day.date()),
            "vix_value": round(last_vix_value, 2),
            "mood_score": round(mood_score)
        }
        
        # Salva i dati in un file JSON
        with open('mood.json', 'w') as f:
            json.dump(mood_data, f)
            
        print(f"File 'mood.json' generato con successo. Punteggio: {mood_data['mood_score']}")

    except Exception as e:
        print(f"ERRORE CRITICO durante la generazione del mood: {e}")
        raise e # Fai fallire il workflow per ricevere una notifica

if __name__ == '__main__':
    get_vix_mood()
