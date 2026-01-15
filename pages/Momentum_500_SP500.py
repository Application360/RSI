import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, date
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Momentum 500 - Étape 3 (Mode Individuel)", layout="wide")

# --- 2. SIDEBAR ---
st.sidebar.header("⚙️ Paramètres")
num_assets = st.sidebar.slider("Nombre d'actions à détenir", 1, 20, 10)
lookback_months = st.sidebar.slider("Look-back (mois)", 1, 12, 6)
rotation_freq = st.sidebar.slider("Rotation (mois)", 1, 12, 3)

st.sidebar.markdown("---")
start_date = st.sidebar.date_input("Date de début", date(1990, 1, 1), min_value=date(1970, 1, 1))
end_date = st.sidebar.date_input("Date de fin", date(2025, 12, 31), min_value=date(1970, 1, 1))

# --- 3. LOGIQUE DE TÉLÉCHARGEMENT INDIVIDUEL ---

@st.cache_data
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    df = pd.read_html(response.text)[0]
    return df['Symbol'].str.replace('.', '-', regex=True).tolist()

def fetch_individual_data(tickers, start, end):
    """Télécharge les tickers un par un pour éviter les blocages de masse"""
    all_closes = {}
    all_opens = {}
    
    pbar = st.progress(0.0)
    status = st.empty()
    
    # On limite à 100 tickers pour ce test si le blocage persiste, 
    # mais essayons avec la liste complète d'abord.
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        try:
            status.text(f"📥 Récupération de {ticker} ({i+1}/{total})...")
            # Utilisation de period="max" ou interval="1mo"
            df = yf.Ticker(ticker).history(start=start, end=end, interval="1mo", raise_errors=False)
            
            if not df.empty:
                all_closes[ticker] = df['Close']
                all_opens[ticker] = df['Open']
            
            # Temps d'attente très court pour ne pas être banni
            if i % 10 == 0:
                time.sleep(0.1)
                
        except Exception:
            continue # On passe au suivant sans bloquer
            
        pbar.progress((i + 1) / total)
    
    if not all_closes: return pd.DataFrame(), pd.DataFrame()
    
    return pd.DataFrame(all_closes), pd.DataFrame(all_opens)

# --- 4. EXÉCUTION ---
st.title("🚀 Momentum 500 : Étape 3 (Mode Individuel)")

tickers = get_sp500_tickers()

if st.button("Lancer le chargement haute résilience"):
    s_str = start_date.strftime('%Y-%m-%d')
    e_str = end_date.strftime('%Y-%m-%d')
    
    # Indice (généralement autorisé car un seul ticker)
    with st.spinner("Chargement de l'indice..."):
        mkt = yf.download("^GSPC", start=s_str, end=e_str, interval="1mo", progress=False)
    
    # Actions
    closes, opens = fetch_individual_data(tickers, s_str, e_str)
    
    if not closes.empty:
        st.success(f"✅ Terminé ! {len(closes.columns)} actions récupérées avec succès.")
        
        st.session_state['closes'] = closes
        st.session_state['opens'] = opens
        st.session_state['mkt'] = mkt
        
        st.dataframe(closes.tail(3), use_container_width=True)
    else:
        st.error("Yahoo Finance bloque toujours. Solution de repli : Réduisez manuellement la liste des tickers dans le code ou attendez 10 minutes.")

elif 'closes' in st.session_state:
    st.info("Données en mémoire.")
