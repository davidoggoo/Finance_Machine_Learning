# generate_forecast.py - Versione Finale Definitiva
# Gestisce il forecast, la copia per la galleria e l'indice JSON.

import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from datetime import date, timedelta
import json
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def generate_sp500_forecast():
    """
    Esegue l'intero processo: download, addestramento, previsione,
    e generazione di tutti gli output necessari (grafici e JSON).
    """
    print("--- Inizio processo di generazione forecast S&P 500 ---")

    # FASE 1 & 2 & 3: Download, Feature Engineering e Addestramento
    # (Questa logica è già stata testata e funziona)
    try:
        df = yf.download('^GSPC', start='2015-01-01', end=date.today(), progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.rename(columns={'Close': 'price'}, inplace=True)
        
        df['lag_1'] = df['price'].shift(1)
        df['rolling_mean_7'] = df['price'].shift(1).rolling(window=7).mean()
        df.dropna(inplace=True)

        FEATURES = ['lag_1', 'rolling_mean_7']
        TARGET = 'price'
        X_train, y_train = df[FEATURES], df[TARGET]

        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        print("Modello addestrato con successo.")
    except Exception as e:
        print(f"ERRORE CRITICO nelle fasi di preparazione: {e}")
        raise e

    # FASE 4: Previsione
    features_for_tomorrow = df[FEATURES].iloc[-1:].copy()
    features_for_tomorrow = features_for_tomorrow[X_train.columns].astype(X_train.dtypes)
    prediction = float(model.predict(features_for_tomorrow)[0])

    # FASE 5: OUTPUT - La nostra "Fabbrica"
    print("--- Fase 5: Generazione di tutti gli output ---")
    
    # A. Crea il grafico
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index[-60:], df['price'][-60:], label='Storico S&P 500', color='blue', linewidth=2)
    future_date = df.index[-1] + timedelta(days=1)
    ax.plot([df.index[-1], future_date], [y_train.iloc[-1], prediction], 'r--o', label=f'Forecast AI: {prediction:.0f}')
    ax.set_title(f"Forecast S&P 500 per il {future_date.strftime('%Y-%m-%d')}", fontsize=16)
    ax.set_ylabel('Valore Indice')
    ax.legend()
    fig.tight_layout()

    # B. Salva il grafico per la homepage
    plt.savefig('forecast.png', dpi=120, bbox_inches='tight')
    print("✓ Grafico 'forecast.png' per la homepage creato.")

    # C. Salva una copia datata nella cartella /charts
    today_str = date.today().strftime('%Y-%m-%d')
    gallery_filename = f"{today_str}-sp500-forecast.png"
    gallery_path = os.path.join('charts', gallery_filename)
    if not os.path.exists('charts'):
        os.makedirs('charts')
    plt.savefig(gallery_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Grafico '{gallery_path}' per la galleria creato.")

    # D. Aggiorna l'indice JSON della galleria
    chart_info = {
        "path": gallery_path,
        "title": f"Forecast S&P 500 - {today_str}",
        "description": "Previsione AI a un giorno per l'indice S&P 500."
    }
    charts_list_path = 'charts_list.json'
    try:
        charts_list = []
        if os.path.exists(charts_list_path):
            with open(charts_list_path, 'r') as f:
                charts_list = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        charts_list = []
        
    charts_list = [c for c in charts_list if c.get('path') != chart_info['path']]
    charts_list.insert(0, chart_info)
    charts_list = charts_list[:30] # Mantiene solo i 30 grafici più recenti
    
    with open(charts_list_path, 'w') as f:
        json.dump(charts_list, f, indent=2)
    print("✓ File 'charts_list.json' aggiornato.")

if __name__ == "__main__":
    generate_sp500_forecast()
    print("\n--- PROCESSO COMPLETATO CON SUCCESSO ---")
