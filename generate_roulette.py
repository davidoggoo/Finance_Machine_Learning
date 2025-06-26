# generate_roulette.py - Applica la nostra logica di forecast a più ticker

import yfinance as yf
import pandas as pd
from xgboost import XGBRegressor
import json
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_prediction_for_ticker(ticker: str) -> dict:
    """
    Funzione riutilizzabile che prende un ticker, addestra un modello
    e restituisce un "tip" sulla previsione del giorno successivo.
    Questa è una versione semplificata e veloce della logica di generate_forecast.py.
    """
    try:
        print(f"--- Analisi per {ticker} ---")
        # 1. Download Dati
        data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if data.empty or len(data) < 50: # Richiede dati sufficienti
            return None

        # 2. Feature Engineering
        data.rename(columns={'Close': 'price'}, inplace=True)
        data['lag_1'] = data['price'].shift(1)
        data['rolling_mean_7'] = data['price'].shift(1).rolling(window=7).mean()
        data.dropna(inplace=True)

        # 3. Addestramento Modello
        FEATURES = ['lag_1', 'rolling_mean_7']
        TARGET = 'price'
        X_train = data[FEATURES]
        y_train = data[TARGET]

        model = XGBRegressor(n_estimators=50, learning_rate=0.1, objective='reg:squarederror', random_state=42)
        model.fit(X_train, y_train)

        # 4. Previsione
        features_for_tomorrow = data[FEATURES].iloc[-1:].copy()
        prediction = model.predict(features_for_tomorrow)[0]
        
        # 5. Genera il "Tip"
        current_price = y_train.iloc[-1]
        change_percent = ((prediction - current_price) / current_price) * 100
        
        direction = "FLAT"
        if change_percent > 0.5:
            direction = "UP"
        elif change_percent < -0.5:
            direction = "DOWN"
            
        return {
            "ticker": ticker,
            "prediction_direction": direction,
            "details": f"Prezzo attuale: {current_price:.2f}. Previsione AI: {prediction:.2f} ({change_percent:+.2f}%)"
        }

    except Exception as e:
        print(f"Errore durante l'analisi di {ticker}: {e}")
        return None

def generate_roulette_tips():
    """
    Esegue la previsione per una lista di ticker e salva i risultati.
    """
    print("--- Inizio generazione tips per AI Trade Roulette ---")
    
    # Lista di ticker popolari su cui girare la roulette
    ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
    all_tips = []

    for ticker in ticker_list:
        tip = get_prediction_for_ticker(ticker)
        if tip: # Aggiungi solo se l'analisi è andata a buon fine
            all_tips.append(tip)
    
    if not all_tips:
        print("Nessun tip generato. Controllare gli errori.")
        return

    # Salva i dati in un file JSON
    with open('roulette_tips.json', 'w') as f:
        json.dump(all_tips, f, indent=2)
    
    print(f"\nFile 'roulette_tips.json' generato con successo con {len(all_tips)} tips.")

if __name__ == '__main__':
    generate_roulette_tips()
