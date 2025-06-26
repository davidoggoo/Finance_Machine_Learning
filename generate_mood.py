# generate_mood.py - Versione Definitiva con chiamata diretta all'API per il VIX

import requests
import json
import os

def get_vix_mood():
    """
    Recupera l'ultimo valore del VIX tramite chiamata API diretta,
    lo normalizza in una scala 0-100 (Fear/Greed) e salva il risultato.
    Questo metodo è più robusto per indicatori speciali come il VIX.
    """
    print("--- Inizio recupero dati per Mood Trader Dial (metodo diretto) ---")
    
    # Recupera la API key dai segreti di GitHub
    api_key = os.getenv('ALPHA_VANTAGE_KEY')
    if not api_key:
        print("ERRORE: API Key di Alpha Vantage non trovata.")
        # Fai fallire il workflow per ricevere una notifica
        raise ValueError("La variabile d'ambiente ALPHA_VANTAGE_KEY non è impostata.")

    # Costruiamo l'URL per l'indicatore economico VIX.
    # Nota: usiamo la funzione generica 'query' perché VIX non è una Time Series standard.
    # La documentazione per gli indicatori economici suggerisce questa via.
    url = f'https://www.alphavantage.co/query?function=VIX&interval=daily&apikey={api_key}'
    
    try:
        r = requests.get(url)
        r.raise_for_status()  # Controlla se ci sono stati errori HTTP (es. 404, 500)
        data = r.json()
        
        # Alpha Vantage può restituire un errore dentro un JSON con status 200 OK
        if "Error Message" in data or "Information" in data:
            print(f"Errore nella risposta dell'API: {data}")
            raise ValueError(f"Risposta API non valida: {data}")
        
        # Per gli indicatori come il VIX, la struttura potrebbe essere diversa
        # Assumiamo che i dati siano in una chiave 'data' e che sia una lista
        if "data" not in data or not isinstance(data['data'], list) or len(data['data']) == 0:
            print(f"La struttura dati della risposta non è quella attesa: {data}")
            raise ValueError("Formato dati per VIX non riconosciuto.")

        # Prendi il dato più recente (il primo della lista)
        latest_data_point = data['data'][0]
        last_day = latest_data_point['date']
        last_vix_value = float(latest_data_point['value'])
        
        print(f"Ultimo valore VIX recuperato: {last_vix_value} in data {last_day}")
        
        # Logica di normalizzazione
        fear_level = max(0, min(1, (last_vix_value - 15) / (50 - 15)))
        mood_score = 100 - (fear_level * 100)
        
        mood_data = {
            "last_update": last_day,
            "vix_value": round(last_vix_value, 2),
            "mood_score": round(mood_score)
        }
        
        with open('mood.json', 'w') as f:
            json.dump(mood_data, f)
            
        print(f"File 'mood.json' generato con successo. Punteggio: {mood_data['mood_score']}")

    except requests.exceptions.RequestException as e:
        print(f"ERRORE di connessione all'API: {e}")
        raise e
    except (KeyError, IndexError, TypeError) as e:
        print(f"ERRORE nel processare la risposta JSON: {e}. Risposta ricevuta: {data}")
        raise e
    except Exception as e:
        print(f"Un errore imprevisto è accaduto: {e}")
        raise e

if __name__ == '__main__':
    get_vix_mood()
