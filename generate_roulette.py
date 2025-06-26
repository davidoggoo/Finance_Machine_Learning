# generate_roulette.py - Versione Definitiva v2, con logica robusta e omogenea

import yfinance as yf
import pandas as pd
from xgboost import XGBRegressor
import json
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_prediction_for_ticker(ticker: str) -> dict:
    """
    Funzione riutilizzabile e ROBUSTA che prende un ticker, addestra un modello
    e restituisce un "tip". Include i fix per MultiIndex e dtypes.
    """
    try:
        print(f"--- Analisi per {ticker} ---")
        
        # --- FASE 1: DOWNLOAD E NORMALIZZAZIONE DATI ---
        data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if data.empty or len(data) < 50:
            print(f"Dati insufficienti per {ticker}. Salto.")
            return None

        # SOLUZIONE: Normalizzazione colonne (se yfinance restituisce un MultiIndex)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Logica robusta per la selezione della colonna
        if 'Close' in data.columns:
            price_column = 'Close'
        else:
            print(f"Colonna 'Close' non trovata per {ticker}. Salto.")
            return None
            
        data.rename(columns={price_column: 'price'}, inplace=True)

        # --- FASE 2: FEATURE ENGINEERING ---
        data['lag_1'] = data['price'].shift(1)
        data['rolling_mean_7'] = data['price'].shift(1).rolling(window=7).mean()
        data.dropna(inplace=True)

        # --- FASE 3: ADDESTRAMENTO MODELLO ---
        FEATURES = ['lag_1', 'rolling_mean_7']
        TARGET = 'price'
        X_train = data[FEATURES]
        y_train = data[TARGET]

        model = XGBRegressor(n_estimators=50, learning_rate=0.1, objective='reg:squarederror', random_state=42)
        model.fit(X_train, y_train)

        # --- FASE 4: PREVISIONE ---
        features_for_tomorrow = data[FEATURES].iloc[-1:].copy()

        # SOLUZIONE: Assicura coerenza dei tipi di dato prima della previsione
        features_for_tomorrow = features_for_tomorrow[X_train.columns].astype(X_train.dtypes)
        
        prediction = model.predict(features_for_tomorrow)[0]
        
        # --- FASE 5: GENERA IL "TIP" ---
        current_price = float(y_train.iloc[-1])
        change_percent = ((prediction - current_price) / current_price) * 100
        
        direction = "FLAT"
        if change_percent > 0.5:
            direction = "UP"
        elif change_percent < -0.5:
            direction = "DOWN"
            
        print(f"Analisi per {ticker} completata. Previsione: {direction}")
        return {
            "ticker": ticker,
            "prediction_direction": direction,
            "details": f"Prezzo attuale: {current_price:.2f}. Previsione AI: {prediction:.2f} ({change_percent:+.2f}%)"
        }

    except Exception as e:
        print(f"ERRORE CRITICO durante l'analisi di {ticker}: {e}")
        return None

def generate_roulette_tips():
    """
    Esegue la previsione per una lista di ticker e salva i risultati.
    """
    print("--- Inizio generazione tips per AI Trade Roulette ---")
    ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
    all_tips = []

    for ticker in ticker_list:
        tip = get_prediction_for_ticker(ticker)
        if tip:
            all_tips.append(tip)
    
    if not all_tips:
        # Fai fallire il workflow se nessun tip viene generato, per notificarci
        raise RuntimeError("Nessun tip generato con successo. Controllare gli errori nei log.")

    # Salva i dati in un file JSON
    with open('roulette_tips.json', 'w') as f:
        json.dump(all_tips, f, indent=2)
    
    print(f"\nFile 'roulette_tips.json' generato con successo con {len(all_tips)} tips.")

if __name__ == '__main__':
    generate_roulette_tips()
