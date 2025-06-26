# generate_forecast.py - v8, La Versione Definitiva 

import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from datetime import date, timedelta
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURAZIONE ---
FORECAST_HORIZON = 15
HISTORY_DAYS = 180 # Aumentato per più contesto visivo

def generate_final_forecast():
    """
    Esegue l'intero processo con una logica di previsione ricorsiva corretta
    per ogni modello, garantendo previsioni dinamiche.
    """
    print("--- Inizio processo Multi-Modello Definitivo ---")

    # --- FASE 1 & 2: DOWNLOAD E FEATURE ENGINEERING ---
    try:
        print("Fase 1 & 2: Download e preparazione dati...")
        df = yf.download('^GSPC', start='2010-01-01', end=date.today(), progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.rename(columns={'Close': 'price'}, inplace=True)
        
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['lag_1'] = df['price'].shift(1)
        df['rolling_mean_7'] = df['price'].shift(1).rolling(window=7).mean()
        df.dropna(inplace=True)
        print("Dati e feature pronti.")
    except Exception as e:
        print(f"ERRORE CRITICO durante la preparazione dei dati: {e}")
        raise e

    # --- FASE 3: DEFINIZIONE E ADDESTRAMENTO DEI MODELLI ---
    print("Fase 3: Addestramento dei modelli...")
    FEATURES = ['day_of_week', 'month', 'year', 'lag_1', 'rolling_mean_7']
    TARGET = 'price'
    X_train, y_train = df[FEATURES], df[TARGET]

    models = {
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    # --- FASE 4: PREVISIONE RICORSIVA CORRETTA E INDIPENDENTE PER OGNI MODELLO ---
    print(f"Fase 4: Calcolo previsioni ricorsive a {FORECAST_HORIZON} giorni...")
    all_predictions = {}
    future_dates = pd.bdate_range(start=df.index[-1] + timedelta(days=1), periods=FORECAST_HORIZON)

    for name, model in models.items():
        print(f"  - Previsione con {name}...")
        # Addestriamo il modello qui, per essere sicuri che sia "pulito" per ogni ciclo
        model.fit(X_train, y_train)
        
        # Inizializziamo la cronologia per questo specifico modello
        model_history = y_train.copy()
        model_predictions = []

        for current_date in future_dates:
            # Prepara le feature basandosi sull'ULTIMO dato disponibile nella cronologia
            last_real_price = model_history.iloc[-1]
            last_rolling_mean = model_history.rolling(window=7).mean().iloc[-1]
            
            current_features = {
                'day_of_week': current_date.dayofweek, 'month': current_date.month,
                'year': current_date.year, 'lag_1': last_real_price, 'rolling_mean_7': last_rolling_mean
            }
            features_for_pred = pd.DataFrame([current_features])[FEATURES] # Assicura l'ordine
            
            # Esegui la previsione
            prediction = float(model.predict(features_for_pred)[0])
            model_predictions.append(prediction)
            
            # AGGIORNA la cronologia di QUESTO modello con la nuova previsione
            # per usarla nel prossimo ciclo
            model_history.loc[current_date] = prediction
        
        all_predictions[name] = model_predictions

    print("Previsioni dinamiche calcolate con successo.")

    # --- FASE 5: OUTPUT - CREAZIONE GRAFICO ---
    print("Fase 5: Creazione grafico finale...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plotta la cronologia storica aumentata
    ax.plot(df.index[-HISTORY_DAYS:], df[TARGET][-HISTORY_DAYS:], label='Storico S&P 500', color='black', linewidth=2.5, zorder=10)

    # Plotta ogni previsione
    colors = ['#FF4136', '#0074D9', '#2ECC40']
    linestyles = ['--', ':', '-.']
    for i, (name, preds) in enumerate(all_predictions.items()):
        ax.plot(future_dates, preds, label=f'Forecast ({name})', color=colors[i], linestyle=linestyles[i], marker='o', markersize=3)

    ax.set_title(f'Confronto Previsioni AI per l\'S&P 500 (+{FORECAST_HORIZON} giorni)', fontsize=18, weight='bold')
    ax.set_ylabel('Valore Indice')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle=':', linewidth=0.7)
    fig.tight_layout()

    plt.savefig('forecast.png', dpi=150, bbox_inches='tight')
    print("\n--- PROCESSO COMPLETATO ---")
    print("Nuovo grafico 'figo e funzionante' generato.")

if __name__ == '__main__':
    generate_final_forecast()
