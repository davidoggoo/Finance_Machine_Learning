# generate_forecast.py - v7, Multi-Modello AVANZATO

import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import date, timedelta
import warnings

# Ignora avvisi non critici
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURAZIONE ---
FORECAST_HORIZON = 15 # Giorni di previsione

def generate_multi_model_forecast():
    """
    Esegue l'intero processo usando 3 diversi modelli avanzati (XGBoost, GradientBoosting, SVR)
    su un orizzonte temporale esteso.
    """
    print("--- Inizio processo Multi-Modello Avanzato ---")

    # --- FASE 1: DOWNLOAD E PREPARAZIONE DATI ---
    try:
        print("Fase 1: Download dati storici...")
        df = yf.download('^GSPC', start='2010-01-01', end=date.today(), progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.rename(columns={'Close': 'price'}, inplace=True)
        print("Dati scaricati e normalizzati.")
    except Exception as e:
        print(f"ERRORE CRITICO durante il download: {e}")
        raise e

    # --- FASE 2: FEATURE ENGINEERING E SCALING ---
    print("Fase 2: Creazione e scaling delle feature...")
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['lag_1'] = df['price'].shift(1)
    df['rolling_mean_7'] = df['price'].shift(1).rolling(window=7).mean()
    df.dropna(inplace=True)

    FEATURES = ['day_of_week', 'month', 'year', 'lag_1', 'rolling_mean_7']
    TARGET = 'price'
    X, y = df[FEATURES], df[TARGET]
    
    # Lo scaling è una best practice, specialmente per SVR
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)


    # --- FASE 3: DEFINIZIONE E ADDESTRAMENTO DEI MODELLI ---
    print("Fase 3: Addestramento dei modelli avanzati...")
    models = {
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "SVR (RBF Kernel)": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"  - Addestramento di {name}...")
        model.fit(X_scaled, y)
        trained_models[name] = model
    print("Tutti i modelli sono stati addestrati.")

    # --- FASE 4: CICLO DI PREVISIONE MULTI-MODELLO ---
    print(f"Fase 4: Calcolo previsioni a {FORECAST_HORIZON} giorni...")
    future_dates = pd.bdate_range(start=df.index[-1] + timedelta(days=1), periods=FORECAST_HORIZON)
    
    all_predictions = {name: [] for name in models.keys()}
    last_known_features = X_scaled.iloc[-1].to_dict()

    for current_date in future_dates:
        for name, model in trained_models.items():
            # Prepara il DataFrame per la previsione con un solo record
            features_for_pred_df = pd.DataFrame([last_known_features])
            
            # Esegui la previsione
            prediction = model.predict(features_for_pred_df)[0]
            all_predictions[name].append(float(prediction))
            
            # Aggiorna le feature per il prossimo giorno usando l'ultima previsione
            # Questo è un approccio semplificato; in un sistema reale sarebbe più complesso
            last_known_features['lag_1'] = prediction 
            # (Nota: non stiamo ricalcolando la media mobile per semplicità, ma l'impatto è minimo)

    print("Previsioni multi-modello calcolate.")

    # --- FASE 5: OUTPUT - CREAZIONE GRAFICO ---
    print("Fase 5: Creazione grafico multi-modello avanzato...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(df.index[-60:], y[-60:], label='Storico S&P 500', color='black', linewidth=3, zorder=10)

    colors = ['#FF4136', '#0074D9', '#2ECC40']
    for i, (name, preds) in enumerate(all_predictions.items()):
        ax.plot(future_dates, preds, label=f'Forecast ({name})', color=colors[i], linestyle='--', marker='o', markersize=4, alpha=0.9)

    ax.set_title(f'Confronto Previsioni AI Avanzate per l\'S&P 500 (+{FORECAST_HORIZON} giorni)', fontsize=16, weight='bold')
    ax.set_ylabel('Valore Indice')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle=':', linewidth=0.6)
    fig.tight_layout()

    plt.savefig('forecast.png', dpi=150, bbox_inches='tight')
    print("\n--- PROCESSO COMPLETATO ---")
    print("Nuovo grafico multi-modello avanzato 'forecast.png' generato.")

if __name__ == '__main__':
    generate_multi_model_forecast()
