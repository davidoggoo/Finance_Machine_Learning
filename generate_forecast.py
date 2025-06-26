# generate_forecast.py - v6, Multi-Modello e Orizzonte Esteso

import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import date, timedelta
import json
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURAZIONE ---
FORECAST_HORIZON = 15 # Giorni di previsione nel futuro

def generate_multi_model_forecast():
    """
    Esegue l'intero processo usando 3 diversi modelli (XGBoost, RandomForest, Linear)
    su un orizzonte temporale esteso.
    """
    print("--- Inizio processo Multi-Modello ---")

    # --- FASE 1: DOWNLOAD E PREPARAZIONE DATI ---
    try:
        print("Fase 1: Download dati storici...")
        df = yf.download('^GSPC', start='2015-01-01', end=date.today(), progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.rename(columns={'Close': 'price'}, inplace=True)
        print("Dati scaricati e normalizzati.")
    except Exception as e:
        print(f"ERRORE CRITICO durante il download: {e}")
        raise e

    # --- FASE 2: FEATURE ENGINEERING ---
    print("Fase 2: Creazione feature...")
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['lag_1'] = df['price'].shift(1)
    df['rolling_mean_7'] = df['price'].shift(1).rolling(window=7).mean()
    df.dropna(inplace=True)

    # --- FASE 3: DEFINIZIONE E ADDESTRAMENTO DEI MODELLI ---
    print("Fase 3: Addestramento dei modelli...")
    FEATURES = ['day_of_week', 'month', 'year', 'lag_1', 'rolling_mean_7']
    TARGET = 'price'
    X_train, y_train = df[FEATURES], df[TARGET]

    models = {
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression()
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"  - Addestramento di {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    print("Tutti i modelli sono stati addestrati.")

    # --- FASE 4: CICLO DI PREVISIONE MULTI-MODELLO ---
    print(f"Fase 4: Calcolo previsioni a {FORECAST_HORIZON} giorni...")
    future_dates = pd.bdate_range(start=df.index[-1] + timedelta(days=1), periods=FORECAST_HORIZON)
    
    all_predictions = {name: [] for name in models.keys()}

    for current_date in future_dates:
        for name, model in trained_models.items():
            # Prepara le feature per la previsione del giorno corrente
            if not all_predictions[name]: # Se è la prima previsione
                last_price = y_train.iloc[-1]
                rolling_mean = y_train.rolling(window=7).mean().iloc[-1]
            else: # Usa la previsione del giorno precedente come input
                last_price = all_predictions[name][-1]
                # Per la media mobile, usiamo una combinazione di dati reali e previsioni già fatte
                historic_for_roll = y_train.tail(6)
                predicted_for_roll = pd.Series(all_predictions[name])
                rolling_mean = pd.concat([historic_for_roll, predicted_for_roll]).rolling(window=7).mean().iloc[-1]

            current_features = {
                'day_of_week': current_date.dayofweek, 'month': current_date.month,
                'year': current_date.year, 'lag_1': last_price, 'rolling_mean_7': rolling_mean
            }
            features_for_pred = pd.DataFrame([current_features])
            features_for_pred = features_for_pred[X_train.columns].astype(X_train.dtypes)
            
            # Esegui e salva la previsione
            prediction = model.predict(features_for_pred)[0]
            all_predictions[name].append(float(prediction))

    print("Previsioni multi-modello calcolate.")

    # --- FASE 5: OUTPUT - CREAZIONE GRAFICO AVANZATO ---
    print("Fase 5: Creazione grafico multi-modello...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plotta i dati storici
    ax.plot(df.index[-60:], df[TARGET][-60:], label='Storico S&P 500', color='black', linewidth=3, zorder=5)

    # Plotta ogni previsione
    colors = ['#FF4136', '#2ECC40', '#0074D9'] # Rosso, Verde, Blu
    for i, (name, preds) in enumerate(all_predictions.items()):
        ax.plot(future_dates, preds, label=f'Forecast ({name})', color=colors[i], linestyle='--', marker='o', markersize=4, alpha=0.8)

    ax.set_title(f'Confronto Previsioni AI per l\'S&P 500 (+{FORECAST_HORIZON} giorni)', fontsize=16, weight='bold')
    ax.set_ylabel('Valore Indice')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle=':', linewidth=0.6)
    fig.tight_layout()

    # Salva il nuovo grafico (sovrascrive quello vecchio)
    plt.savefig('forecast.png', dpi=150, bbox_inches='tight')
    print("\n--- PROCESSO COMPLETATO ---")
    print("Nuovo grafico multi-modello 'forecast.png' generato e salvato.")

if __name__ == '__main__':
    generate_multi_model_forecast()
