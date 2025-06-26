# generate_roulette.py - v3, Aggiunge i dati storici per i grafici nel frontend

import yfinance as yf
import pandas as pd
from xgboost import XGBRegressor
import json
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_prediction_for_ticker(ticker: str) -> dict:
    """
    Funzione robusta che genera un "tip" di previsione e include gli ultimi 60 giorni
    di dati storici per la visualizzazione nel frontend.
    """
    try:
        print(f"--- Analisi per {ticker} ---")
        
        # Scarichiamo più dati per avere uno storico sufficiente
        data = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
        if data.empty or len(data) < 60:
            print(f"Dati insufficienti per {ticker}. Salto.")
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        data.rename(columns={'Close': 'price'}, inplace=True)

        # Prepara i dati storici per il grafico (ultimi 60 punti)
        # Convertiamo in un formato efficiente per il JSON: [[timestamp, valore], ...]
        historical_data = data['price'].tail(60)
        chart_data = [[int(ts.timestamp() * 1000), round(price, 2)] for ts, price in historical_data.items()]

        # Feature Engineering e Addestramento
        data['lag_1'] = data['price'].shift(1)
        data['rolling_mean_7'] = data['price'].shift(1).rolling(window=7).mean()
        data.dropna(inplace=True)
        
        FEATURES = ['lag_1', 'rolling_mean_7']
        TARGET = 'price'
        X_train, y_train = data[FEATURES], data[TARGET]
        model = XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        # Previsione
        features_for_tomorrow = data[FEATURES].iloc[-1:].copy()
        features_for_tomorrow = features_for_tomorrow[X_train.columns].astype(X_train.dtypes)
        prediction = model.predict(features_for_tomorrow)[0]
        
        # Genera il "Tip"
        current_price = float(y_train.iloc[-1])
        change_percent = ((prediction - current_price) / current_price) * 100
        direction = "FLAT"
        if change_percent > 0.5: direction = "UP"
        elif change_percent < -0.5: direction = "DOWN"
            
        print(f"Analisi per {ticker} completata. Previsione: {direction}")
        return {
            "ticker": ticker,
            "prediction_direction": direction,
            "details": f"Prezzo attuale: {current_price:.2f}. Previsione AI: {prediction:.2f} ({change_percent:+.2f}%)",
            "chart_data": chart_data # <-- LA NOVITÀ: I DATI PER IL GRAFICO!
        }

    except Exception as e:
        print(f"ERRORE CRITICO durante l'analisi di {ticker}: {e}")
        return None

def generate_roulette_tips():
    print("--- Inizio generazione tips per AI Trade Roulette ---")
    ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
    all_tips = [tip for ticker in ticker_list if (tip := get_prediction_for_ticker(ticker)) is not None]
    
    if not all_tips:
        raise RuntimeError("Nessun tip generato con successo.")

    with open('roulette_tips.json', 'w') as f:
        json.dump(all_tips, f, indent=2)
    
    print(f"\nFile 'roulette_tips.json' generato con {len(all_tips)} tips, completi di dati per i grafici.")

if __name__ == '__main__':
    generate_roulette_tips()
