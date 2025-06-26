# generate_forecast.py - v4, Aggiunge la creazione di grafici per la galleria

import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from datetime import date
import json
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def generate_sp500_forecast():
    # ... (tutta la logica di download, feature engineering e addestramento rimane IDENTICA a prima) ...
    # ... la incollo qui sotto per completezza, ma la parte nuova è solo alla fine ...
    
    print("--- Inizio del processo di generazione del forecast ---")
    try:
        print("Fase 1: Download dei dati storici...")
        ticker = '^GSPC'
        df = yf.download(ticker, start='2015-01-01', end=date.today(), progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        price_column = 'Close' if 'Close' in df.columns else 'Adj Close'
        df = df[[price_column]].copy()
        df.rename(columns={price_column: 'price'}, inplace=True)
    except Exception as e:
        print(f"ERRORE CRITICO durante il download: {e}")
        raise e

    print("Fase 2: Creazione delle feature...")
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['lag_1'] = df['price'].shift(1)
    df['rolling_mean_7'] = df['price'].shift(1).rolling(window=7).mean()
    df.dropna(inplace=True)

    print("Fase 3: Addestramento del modello...")
    FEATURES = ['day_of_week', 'month', 'year', 'lag_1', 'rolling_mean_7']
    TARGET = 'price'
    X_train, y_train = df[FEATURES], df[TARGET]
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    print("Fase 4: Calcolo delle previsioni...")
    features_for_tomorrow = df[FEATURES].iloc[-1:].copy()
    features_for_tomorrow = features_for_tomorrow[X_train.columns].astype(X_train.dtypes)
    prediction = float(model.predict(features_for_tomorrow)[0])

    # --- FASE 5: OUTPUT - CREAZIONE GRAFICI E INDICE JSON (Parte Nuova) ---
    print("Fase 5: Creazione e salvataggio dei grafici e dell'indice...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index[-60:], df['price'][-60:], label='Storico S&P 500', color='blue', linewidth=2)
    future_date = df.index[-1] + pd.Timedelta(days=1)
    ax.plot([df.index[-1], future_date], [y_train.iloc[-1], prediction], 'r--o', label=f'Forecast AI: {prediction:.0f}')
    ax.set_title(f"Forecast S&P 500 per il {future_date.strftime('%Y-%m-%d')}", fontsize=16)
    ax.set_ylabel('Valore Indice')
    ax.legend()
    fig.tight_layout()

    # 1. Salva il grafico per la homepage
    plt.savefig('forecast.png', dpi=120, bbox_inches='tight')
    print("Grafico 'forecast.png' per la homepage aggiornato.")

    # 2. Salva una copia datata nella cartella /charts
    today_str = date.today().strftime('%Y-%m-%d')
    gallery_filename = f"{today_str}-sp500-forecast.png"
    gallery_path = os.path.join('charts', gallery_filename)
    plt.savefig(gallery_path, dpi=120, bbox_inches='tight')
    print(f"Grafico '{gallery_path}' per la galleria salvato.")
    plt.close(fig)

    # 3. Aggiorna l'indice JSON della galleria
    chart_info = {
        "path": gallery_path,
        "title": f"Forecast S&P 500 - {today_str}",
        "description": "Previsione AI a un giorno per l'indice S&P 500."
    }
    
    charts_list_path = 'charts_list.json'
    try:
        if os.path.exists(charts_list_path):
            with open(charts_list_path, 'r') as f:
                charts_list = json.load(f)
        else:
            charts_list = []
    except json.JSONDecodeError:
        charts_list = []

    # Rimuovi eventuali duplicati per la stessa data e aggiungi il nuovo in cima
    charts_list = [c for c in charts_list if c['title'] != chart_info['title']]
    charts_list.insert(0, chart_info)
    
    # Mantieni solo i 30 grafici più recenti
    charts_list = charts_list[:30]
    
    with open(charts_list_path, 'w') as f:
        json.dump(charts_list, f, indent=2)
    print("File 'charts_list.json' aggiornato.")


if __name__ == '__main__':
    # Assicura che la cartella 'charts' esista
    if not os.path.exists('charts'):
        os.makedirs('charts')
    generate_sp500_forecast()
    print("\n--- PROCESSO COMPLETATO CON SUCCESSO ---")
