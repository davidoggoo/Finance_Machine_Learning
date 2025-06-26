# generate_roulette.py - v5, Previsione Estesa a 7 giorni

import yfinance as yf
import pandas as pd
from xgboost import XGBRegressor
import json
import warnings
from datetime import timedelta

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_prediction_for_ticker(ticker: str, forecast_horizon: int = 7) -> dict:
    """
    Genera un "tip" con una previsione estesa a 7 giorni e include
    i dati storici per la visualizzazione.
    """
    try:
        print(f"--- Analisi per {ticker} con orizzonte a {forecast_horizon} giorni ---")
        
        data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if data.empty or len(data) < 60:
            print(f"Dati insufficienti per {ticker}. Salto.")
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        data.rename(columns={'Close': 'price'}, inplace=True)

        # Prepara dati storici per il grafico
        historical_data = data['price'].tail(60)
        chart_data = [[int(ts.timestamp() * 1000), round(price, 2)] for ts, price in historical_data.items()]

        # Feature Engineering
        data['lag_1'] = data['price'].shift(1)
        data['rolling_mean_7'] = data['price'].shift(1).rolling(window=7).mean()
        data.dropna(inplace=True)
        
        FEATURES = ['lag_1', 'rolling_mean_7']
        TARGET = 'price'
        X_train, y_train = data[FEATURES], data[TARGET]
        
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        # Previsione Ricorsiva Corretta
        predictions = []
        model_history = y_train.copy()
        future_dates = pd.bdate_range(start=data.index[-1] + timedelta(days=1), periods=forecast_horizon)

        for current_date in future_dates:
            last_price = model_history.iloc[-1]
            last_rolling_mean = model_history.rolling(window=7).mean().iloc[-1]
            
            current_features = {'lag_1': last_price, 'rolling_mean_7': last_rolling_mean}
            features_for_pred = pd.DataFrame([current_features])[FEATURES]
            
            prediction = float(model.predict(features_for_pred)[0])
            predictions.append(prediction)
            model_history.loc[current_date] = prediction

        # Prepara i dati del forecast per il JSON
        forecast_data_for_json = [[int(ts.timestamp() * 1000), round(p, 2)] for ts, p in zip(future_dates, predictions)]
        
        # Genera il "Tip" basato sulla previsione finale
        current_price = float(y_train.iloc[-1])
        final_prediction = predictions[-1]
        change_percent = ((final_prediction - current_price) / current_price) * 100
        direction = "FLAT"
        if change_percent > 1.0: direction = "UP"  # Usiamo una soglia più alta per un orizzonte più lungo
        elif change_percent < -1.0: direction = "DOWN"
            
        print(f"Analisi per {ticker} completata. Previsione finale: {direction}")
        return {
            "ticker": ticker,
            "prediction_direction": direction,
            "details": f"Prezzo attuale: {current_price:.2f}. Previsione AI a 7 giorni: {final_prediction:.2f} ({change_percent:+.2f}%)",
            "historical_data": chart_data,
            "forecast_data": forecast_data_for_json # <-- NUOVO: L'intera linea di previsione
        }

    except Exception as e:
        print(f"ERRORE CRITICO durante l'analisi di {ticker}: {e}")
        return None

def generate_roulette_tips():
    print("--- Inizio generazione tips estesi per AI Trade Roulette ---")
    ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
    all_tips = [tip for ticker in ticker_list if (tip := get_prediction_for_ticker(ticker)) is not None]
    
    if not all_tips:
        raise RuntimeError("Nessun tip generato con successo.")

    with open('roulette_tips.json', 'w') as f:
        json.dump(all_tips, f) # Usiamo un formato più compatto senza indentazione
    
    print(f"\nFile 'roulette_tips.json' generato con {len(all_tips)} tips, con previsioni a 7 giorni.")

if __name__ == '__main__':
    generate_roulette_tips()
