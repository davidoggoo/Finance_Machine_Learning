# generate_mood.py - Versione Definitiva v2.1 AI-Powered, con fix sul tipo di dato scalare

import yfinance as yf
import pandas as pd
from xgboost import XGBRegressor
import json
from datetime import date, timedelta
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def generate_ai_mood_dial():
    """
    Recupera i dati storici del VIX, addestra un modello XGBoost per prevedere
    il valore del giorno successivo, e calcola sia il mood attuale che quello previsto.
    Questa versione assicura che i valori estratti siano scalari (float).
    """
    print("--- Inizio processo AI per Mood Trader Dial ---")

    try:
        # --- 1. Download Dati Storici del VIX ---
        print("Fase 1: Download dati storici VIX...")
        ticker = "^VIX"
        vix_df = yf.download(ticker, start="2010-01-01", end=date.today(), progress=False, auto_adjust=True)
        if vix_df.empty:
            raise ValueError(f"Nessun dato scaricato per {ticker}")

        # --- 2. Feature Engineering ---
        print("Fase 2: Creazione feature...")
        vix_df.rename(columns={'Close': 'price'}, inplace=True)
        vix_df['lag_1'] = vix_df['price'].shift(1)
        vix_df['rolling_mean_7'] = vix_df['price'].shift(1).rolling(window=7).mean()
        vix_df.dropna(inplace=True)

        # --- 3. Addestramento Modello ---
        print("Fase 3: Addestramento modello XGBoost sul VIX...")
        FEATURES = ['lag_1', 'rolling_mean_7']
        TARGET = 'price'
        X_train = vix_df[FEATURES]
        y_train = vix_df[TARGET]

        model = XGBRegressor(n_estimators=100, learning_rate=0.1, objective='reg:squarederror', random_state=42)
        model.fit(X_train, y_train)
        print("Modello addestrato.")

        # --- 4. Calcolo Mood Attuale ---
        # ===== LA CORREZIONE DEFINITIVA È QUI =====
        # Estraiamo l'ultimo valore e lo convertiamo esplicitamente in un numero float.
        latest_real_vix = float(y_train.iloc[-1])
        latest_real_date = y_train.index[-1]
        
        def calculate_mood_score(vix_value):
            # Questa funzione ora riceverà sempre un numero, mai una Series.
            fear_level = max(0, min(1, (vix_value - 15) / (50 - 15)))
            return 100 - (fear_level * 100)

        current_mood_score = calculate_mood_score(latest_real_vix)
        print(f"Mood attuale calcolato: {current_mood_score:.0f}/100")

        # --- 5. Previsione AI Mood Domani ---
        print("Fase 5: Previsione VIX per domani...")
        features_for_tomorrow = vix_df[FEATURES].iloc[-1:].copy()
        
        predicted_vix = model.predict(features_for_tomorrow)[0]
        predicted_mood_score = calculate_mood_score(predicted_vix)
        print(f"Mood previsto dall'AI: {predicted_mood_score:.0f}/100")

        # --- 6. Salvataggio Dati ---
        output_data = {
            "last_update": str(latest_real_date.date()),
            "current_vix": round(float(latest_real_vix), 2),
            "current_mood_score": round(current_mood_score),
            "predicted_vix": round(float(predicted_vix), 2),
            "predicted_mood_score": round(predicted_mood_score)
        }

        with open('mood.json', 'w') as f:
            json.dump(output_data, f)
            
        print("File 'mood.json' con dati attuali e previsti generato con successo.")

    except Exception as e:
        print(f"ERRORE CRITICO: {e}")
        raise e # Fai fallire il workflow per notificarci

if __name__ == '__main__':
    generate_ai_mood_dial()
