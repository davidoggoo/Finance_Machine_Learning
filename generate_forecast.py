# generate_forecast.py - v9, Stile Moderno con "Cono di Incertezza"

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
FORECAST_HORIZON = 45
HISTORY_DAYS = 180

def generate_final_forecast():
    """
    Esegue l'intero processo e genera un grafico finale con uno stile moderno,
    incluso un "cono di incertezza" basato sulle previsioni multiple.
    """
    print("--- Inizio processo Multi-Modello Definitivo ---")

    # Fasi 1, 2, 3 e 4: (Download, Feature Engineering, Addestramento, Previsione)
    # Questa logica è già stata testata e rimane IDENTICA.
    
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

    print("Fase 3: Addestramento dei modelli...")
    FEATURES = ['day_of_week', 'month', 'year', 'lag_1', 'rolling_mean_7']
    TARGET = 'price'
    X_train, y_train = df[FEATURES], df[TARGET]

    models = {
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    print("Fase 4: Calcolo previsioni ricorsive...")
    all_predictions = {}
    future_dates = pd.bdate_range(start=df.index[-1] + timedelta(days=1), periods=FORECAST_HORIZON)

    for name, model in models.items():
        model.fit(X_train, y_train)
        model_history = y_train.copy()
        model_predictions = []
        for current_date in future_dates:
            last_real_price = model_history.iloc[-1]
            last_rolling_mean = model_history.rolling(window=7).mean().iloc[-1]
            current_features = {'day_of_week': current_date.dayofweek, 'month': current_date.month, 'year': current_date.year, 'lag_1': last_real_price, 'rolling_mean_7': last_rolling_mean}
            features_for_pred = pd.DataFrame([current_features])[FEATURES]
            prediction = float(model.predict(features_for_pred)[0])
            model_predictions.append(prediction)
            model_history.loc[current_date] = prediction
        all_predictions[name] = model_predictions

    print("Previsioni dinamiche calcolate.")

    # --- FASE 5: OUTPUT - CREAZIONE GRAFICO MODERNO (Parte Modificata) ---
    print("Fase 5: Creazione grafico finale con stile avanzato...")
    
    # Usa uno stile pulito e professionale
    plt.style.use('seaborn-v0_8-ticks')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plotta la cronologia storica
    ax.plot(df.index[-HISTORY_DAYS:], df[TARGET][-HISTORY_DAYS:], label='Storico S&P 500', color='#1a1a1a', linewidth=2.5, zorder=10)

    # Prepara i dati per il "cono di incertezza"
    predictions_df = pd.DataFrame(all_predictions, index=future_dates)
    min_preds = predictions_df.min(axis=1)
    max_preds = predictions_df.max(axis=1)

    # Disegna il cono di incertezza
    ax.fill_between(future_dates, min_preds, max_preds, color='#007bff', alpha=0.1, label='Range Previsioni AI')
    
    # Plotta la previsione principale (media di tutti i modelli)
    mean_preds = predictions_df.mean(axis=1)
    ax.plot(future_dates, mean_preds, label='Forecast Medio AI', color='#0056b3', linestyle='--', marker='o', markersize=4)

    # Abbellimento finale
    ax.set_title(f'Forecast Multi-Modello per l\'S&P 500 (+{FORECAST_HORIZON} giorni)', fontsize=20, weight='bold', pad=20)
    ax.set_ylabel('Valore Indice', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, which='major', linestyle=':', linewidth=0.5, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    plt.savefig('forecast.png', dpi=150, bbox_inches='tight')
    print("\n--- PROCESSO COMPLETATO ---")
    print("Nuovo grafico con stile moderno 'forecast.png' generato.")

if __name__ == '__main__':
    generate_final_forecast()
