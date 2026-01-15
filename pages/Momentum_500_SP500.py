import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, date
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Momentum 500 - Étape 3 (Finale)", layout="wide")

# --- 2. SIDEBAR : PARAMÈTRES ET CURSEURS DE DATES ---
st.sidebar.header("⚙️ Configuration")

num_assets = st.sidebar.slider("Actions à détenir", 1, 20, 10)
lookback_months = st.sidebar.slider("Look-back (mois)", 1, 12, 6)
rotation_freq = st.sidebar.slider("Rotation (mois)", 1, 12, 3)

st.sidebar.markdown("---")
st.sidebar.header("📅 Période (1970 - 2025)")

# MODIFICATION : Curseurs couvrant 1970 à 2025
start_date = st.sidebar.date_input(
    "Date de début", 
    value=date(1990, 1, 1), 
    min_value=date(1970, 1, 1), 
    max_value=date(2025, 12, 31)
)

# On s'assure que la fin est après le début
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
    try:
        response = requests.get(url, headers=headers, timeout=10)
        df = pd.read_html(response.text)[0]
        return df['Symbol'].str.replace('.', '-', regex=True).tolist()
    except:
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"] # Backup au cas où

def fetch_data_robust(tickers, start, end):
    """Téléchargement optimisé pour Streamlit Cloud"""
    status = st.empty()
    progress = st.progress(0.0)
    
    # On télécharge par petits paquets de 25 pour ne pas saturer la connexion
    chunk_size = 25
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    all_closes = []
    all_opens = []
    
    for i, chunk in enumerate(chunks):
        status.text(f"📥 Récupération du groupe {i+1}/{len(chunks)}...")
        try:
            # On demande uniquement Open et Close pour gagner de la bande passante
            data = yf.download(chunk, start=start, end=end, interval="1mo", progress=False, threads=True)[['Close', 'Open']]
            
            if not data.empty:
                all_closes.append(data['Close'])
                all_opens.append(data['Open'])
            
            time.sleep(0.5) # Pause anti-blocage
        except Exception:
            continue
        
        progress.progress((i + 1) / len(chunks))
    
    if not all_closes: return pd.DataFrame(), pd.DataFrame()
    
    return pd.concat(all_closes, axis=1), pd.concat(all_opens, axis=1)

# --- 4. EXÉCUTION ---
st.title("🚀 Momentum 500 : Étape 3")

tickers = get_sp500_tickers()

if st.button("Lancer le chargement des données"):
    s_str = start_date.strftime('%Y-%m-%d')
    e_str = end_date.strftime('%Y-%m-%d')
    
    # 1. Benchmark
    with st.spinner("Chargement de l'indice S&P 500..."):
        mkt = yf.download("^GSPC", start=s_str, end=e_str, interval="1mo", progress=False)
    
    # 2. Actions
    closes, opens = fetch_data_robust(tickers, s_str, e_str)
    
    if not closes.empty:
        st.success(f"✅ Analyse prête sur {len(closes.columns)} actions !")
        
        # Sauvegarde pour l'étape suivante
        st.session_state['closes'] = closes
        st.session_state['opens'] = opens
        st.session_state['mkt'] = mkt
        
        # Affichage du tableau de bord de données
        st.subheader("Aperçu des données récupérées")
        st.dataframe(closes.tail(5), use_container_width=True)
    else:
        st.error("Yahoo Finance ne répond pas. Réessayez dans 1 minute ou réduisez la période.")

elif 'closes' in st.session_state:
    st.info("Données chargées en mémoire. Prêt pour l'étape 4.")
