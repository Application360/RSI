import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Momentum 500 - Étape 3 (Fix)", layout="wide")

# --- 2. SIDEBAR ---
st.sidebar.header("⚙️ Paramètres")
num_assets = st.sidebar.slider("Nombre d'actions à détenir", 1, 20, 10)
lookback_months = st.sidebar.slider("Période d'analyse Momentum (mois)", 1, 12, 6)
rotation_freq = st.sidebar.slider("Fréquence de rotation (mois)", 1, 12, 3)
fees_pct = st.sidebar.slider("Frais par transaction (%)", 0.0, 0.5, 0.10, step=0.01) / 100

st.sidebar.markdown("---")
start_input = st.sidebar.text_input("Date de début", "1980/01/01")
end_input = st.sidebar.text_input("Date de fin", datetime.now().strftime("%Y/%m/%d"))

# --- 3. FONCTIONS DE RÉCUPÉRATION ---

@st.cache_data
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    df = pd.read_html(response.text)[0]
    return df['Symbol'].str.replace('.', '-', regex=True).tolist()

def download_with_retry(tickers, start, end):
    """Télécharge les données par paquets pour éviter les erreurs HTTP/Timeout"""
    all_opens = []
    all_closes = []
    
    # Création de paquets de 50 tickers
    chunk_size = 50
    ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(ticker_chunks):
        status_text.text(f"Téléchargement du groupe {i+1}/{len(ticker_chunks)}...")
        try:
            # Téléchargement uniquement de Open et Close en mensuel
            chunk_data = yf.download(chunk, start=start, end=end, interval="1mo", progress=False)[['Open', 'Close']]
            
            if not chunk_data.empty:
                all_opens.append(chunk_data['Open'])
                all_closes.append(chunk_data['Close'])
            
            # Petite pause pour ne pas être banni par Yahoo
            time.sleep(0.5)
        except Exception as e:
            st.warning(f"Erreur sur le groupe {i+1}: {e}")
        
        progress_bar.progress((i + 1) / len(ticker_chunks))
    
    if not all_opens or not all_closes:
        return pd.DataFrame(), pd.DataFrame()

    # Fusion des résultats
    full_opens = pd.concat(all_opens, axis=1)
    full_closes = pd.concat(all_closes, axis=1)
    
    return full_opens, full_closes

# --- 4. EXÉCUTION ---
st.title("🚀 Momentum 500 : Étape 3 (Optimisée)")
st.write("Chargement robuste des données (Open/Close) par paquets de 50.")

tickers = get_sp500_tickers()

if st.button("Lancer le téléchargement sécurisé"):
    s_date = start_input.replace('/', '-')
    e_date = end_input.replace('/', '-')
    
    # Téléchargement de l'indice séparément
    mkt = yf.download("^GSPC", start=s_date, end=e_date, interval="1mo", progress=False)
    
    # Téléchargement des 500 actions
    opens, closes = download_with_retry(tickers, s_date, e_date)
    
    if not opens.empty and not closes.empty:
        st.success(f"✅ Téléchargement réussi ! ({len(closes.columns)} actions récupérées)")
        
        # Aperçu pour vérification
        tab1, tab2 = st.tabs(["📊 Cours de Clôture", "📈 Cours d'Ouverture"])
        with tab1:
            st.dataframe(closes.tail(), use_container_width=True)
        with tab2:
            st.dataframe(opens.tail(), use_container_width=True)
            
        # Sauvegarde en session pour l'étape 4
        st.session_state['closes'] = closes
        st.session_state['opens'] = opens
        st.session_state['mkt'] = mkt
    else:
        st.error("Échec du téléchargement. Yahoo Finance a rejeté la demande ou les dates sont invalides.")
