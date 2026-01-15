import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, date
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Momentum 500 - Étape 3 (Dates Étendues)", layout="wide")

# --- 2. SIDEBAR : PARAMÈTRES ---
st.sidebar.header("⚙️ Paramètres de la Stratégie")

num_assets = st.sidebar.slider("Nombre d'actions à détenir", 1, 20, 10)
lookback_months = st.sidebar.slider("Période d'analyse Momentum (mois)", 1, 12, 6)
rotation_freq = st.sidebar.slider("Fréquence de rotation (mois)", 1, 12, 3)
fees_pct = st.sidebar.slider("Frais par transaction (%)", 0.0, 0.5, 0.10, step=0.01) / 100

st.sidebar.markdown("---")
st.sidebar.header("📅 Période d'Analyse (1970 - 2025)")

# MODIFICATION : Curseurs de dates allant de 1970 à 2025
start_date = st.sidebar.date_input(
    "Date de début", 
    value=date(1990, 1, 1), 
    min_value=date(1970, 1, 1), 
    max_value=date(2025, 12, 31)
)

end_date = st.sidebar.date_input(
    "Date de fin", 
    value=date(2025, 12, 31), 
    min_value=date(1970, 1, 1), 
    max_value=date(2025, 12, 31)
)

# --- 3. FONCTIONS DE RÉCUPÉRATION ---

@st.cache_data
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    df = pd.read_html(response.text)[0]
    return df['Symbol'].str.replace('.', '-', regex=True).tolist()

def fetch_data_in_batches(tickers, start, end):
    """Téléchargement ultra-fractionné pour éviter les rejets de Yahoo"""
    all_closes = []
    all_opens = []
    chunk_size = 15 # On réduit la taille pour plus de sécurité
    
    ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    for i, chunk in enumerate(ticker_chunks):
        status_text.text(f"📥 Chargement des actions : {i+1}/{len(ticker_chunks)}...")
        try:
            # On télécharge Open et Close ensemble pour limiter le nombre de requêtes totales
            data = yf.download(chunk, start=start, end=end, interval="1mo", progress=False)
            
            if not data.empty:
                if 'Close' in data: all_closes.append(data['Close'])
                if 'Open' in data: all_opens.append(data['Open'])
            
            time.sleep(0.4) # Pause entre les appels
        except Exception:
            continue
        
        progress_bar.progress((i + 1) / len(ticker_chunks))
    
    if not all_closes: return pd.DataFrame(), pd.DataFrame()
    
    return pd.concat(all_closes, axis=1), pd.concat(all_opens, axis=1)

# --- 4. EXÉCUTION ---
st.title("🚀 Momentum 500 : Étape 3")
st.info(f"Analyse configurée de {start_date.year} à {end_date.year}")

tickers = get_sp500_tickers()

if st.button("Lancer le backtest (Téléchargement des données)"):
    if start_date >= end_date:
        st.error("La date de début doit être antérieure à la date de fin.")
    else:
        s_str = start_date.strftime('%Y-%m-%d')
        e_str = end_date.strftime('%Y-%m-%d')
        
        # 1. Indice de référence
        with st.spinner("Téléchargement du S&P 500..."):
            mkt = yf.download("^GSPC", start=s_str, end=e_str, interval="1mo", progress=False)
        
        # 2. Données des actions
        closes, opens = fetch_data_in_batches(tickers, s_str, e_str)
        
        if not closes.empty:
            st.success(f"✅ Données chargées pour {len(closes.columns)} actions.")
            
            # Stockage en session
            st.session_state['closes'] = closes
            st.session_state['opens'] = opens
            st.session_state['mkt'] = mkt
            
            # Aperçu
            st.dataframe(closes.tail(5), use_container_width=True)
        else:
            st.error("Échec du téléchargement. Yahoo Finance limite l'accès. Essayez une période plus courte ou réessayez dans quelques minutes.")

elif 'closes' in st.session_state:
    st.success("Données prêtes pour l'Étape 4 (Calcul du Ranking).")
