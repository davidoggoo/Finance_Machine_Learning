# ==============================================================================
# SCRIPT FINALE v3 PER LA GENERAZIONE DEL FORECAST S&P 500
# Soluzione definitiva con normalizzazione della struttura dei dati (MultiIndex)
# ==============================================================================

import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from datetime import date, timedelta
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def generate_sp500_forecast():
    print("--- Inizio del processo di generazione del forecast ---")

    # --- FASE 1: RACCOLTA E NORMALIZZAZIONE DATI ---
    try:
        print("Fase 1: Download dei dati storici...")
        ticker = '^GSPC'
        df = yf.download(ticker, start='2015-01-01', end=date.today(), progress=False)

        if df.empty:
            print("ERRORE: Nessun dato scaricato. Fine del processo.")
            return

        # ===== SOLUZIONE INTELLIGENTE: NORMALIZZAZIONE COLONNE =====
        # Controlla se le colonne sono un MultiIndex e le appiattisce.
        # Questo rende l'intera pipeline stabile e prevedibile.
        if isinstance(df.columns, pd.MultiIndex):
            print("Rilevato MultiIndex nelle colonne. Appiattimento in corso...")
            df.columns = df.columns.droplevel(1) # Mantiene solo il primo livello (es. 'Close')
        # ==========================================================

        # Logica robusta per selezionare la colonna del prezzo
        if 'Adj Close' in df.columns:
            price_column = 'Adj Close'
        elif 'Close' in df.columns:
            price_column = 'Close'
        else:
            print(f"ERRORE: Colonne di prezzo non trovate. Colonne disponibili: {df.columns.tolist()}")
            return
            
        print(f"Dati normalizzati. Usando la colonna '{price_column}'.")
        df = df[[price_column]].copy()
        df.rename(columns={price_column: 'price'}, inplace=True)

    except Exception as e:
        print(f"ERRORE CRITICO durante il download o la normalizzazione: {e}")
        return
        
    # --- FASE 2: FEATURE ENGINEERING ---
    # Da qui in poi, siamo sicuri che 'df' ha colonne semplici
    print("Fase 2: Creazione delle feature per il modello...")
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['lag_1'] = df['price'].shift(1)
    df['rolling_mean_7'] = df['price'].shift(1).rolling(window=7).mean()
    df.dropna(inplace=True)

    # --- FASE 3: ADDESTRAMENTO DEL MODELLO ---
    print("Fase 3: Addestramento del modello XGBoost...")
    FEATURES = ['day_of_week', 'month', 'year', 'lag_1', 'rolling_mean_7']
    TARGET = 'price'

    X_train = df[FEATURES]
    y_train = df[TARGET]

    model = XGBRegressor(n_estimators=500, learning_rate=0.05, objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train, verbose=False)
    print("Modello addestrato con successo.")

    # --- FASE 4: CICLO DI PREVISIONE FUTURA ---
    print("Fase 4: Calcolo delle previsioni per i prossimi 5 giorni...")
    predictions = []
    last_known_data = df.tail(7).copy()
    future_dates = pd.bdate_range(start=df.index[-1] + timedelta(days=1), periods=5)

    for current_date in future_dates:
        last_price = last_known_data[TARGET].iloc[-1]
        rolling_mean = last_known_data[TARGET].rolling(window=7).mean().iloc[-1]
        
        current_features = {
            'day_of_week': current_date.dayofweek,
            'month': current_date.month,
            'year': current_date.year,
            'lag_1': last_price,
            'rolling_mean_7': rolling_mean
        }
        features_for_pred = pd.DataFrame([current_features])

        # La coerenza ora funziona perché sia X_train che features_for_pred hanno colonne semplici
        features_for_pred = features_for_pred[X_train.columns].astype(X_train.dtypes)
        
        prediction = model.predict(features_for_pred)[0]
        predictions.append(prediction)
        
        new_row = pd.DataFrame({TARGET: [prediction]}, index=[current_date])
        last_known_data = pd.concat([last_known_data, new_row])

    print("Previsioni calcolate.")

    # --- FASE 5: OUTPUT - CREAZIONE GRAFICO ---
    print("Fase 5: Creazione e salvataggio del grafico...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df.index[-60:], df[TARGET][-60:], label='Storico S&P 500', color='blue', linewidth=2)
    ax.plot(future_dates, predictions, label='Forecast AI (+5 giorni)', color='red', linestyle='--', marker='o')
    ax.set_title('Previsioni AI - Indice S&P 500', fontsize=16, weight='bold')
    ax.set_ylabel('Valore Indice')
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    plt.savefig('forecast.png', dpi=150, bbox_inches='tight')
    
    print("\n--- PROCESSO COMPLETATO CON SUCCESSO ---")
    print("Grafico 'forecast.png' generato e salvato.")

if __name__ == '__main__':
    generate_sp500_forecast()
