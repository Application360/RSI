import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, date
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Momentum 100 - Étape 3", layout="wide")

# --- 2. SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
num_assets = st.sidebar.slider("Actions à détenir", 1, 20, 10)
lookback_months = st.sidebar.slider("Look-back (mois)", 1, 12, 6)
rotation_freq = st.sidebar.slider("Rotation (mois)", 1, 12, 3)

st.sidebar.markdown("---")
st.sidebar.header("📅 Période (1970 - 2025)")
start_date = st.sidebar.date_input("Date de début", value=date(2000, 1, 1), min_value=date(1970, 1, 1))
end_date = st.sidebar.date_input("Date de fin", value=date(2025, 12, 31))

# --- 3. FONCTION DE RÉCUPÉRATION ALLÉGÉE (S&P 100) ---

@st.cache_data
def get_sp100_tickers():
    # On passe sur le S&P 100 pour garantir que le téléchargement aboutisse
    url = 'https://en.wikipedia.org/wiki/S%26P_100'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        df = pd.read_html(response.text)[2] # Le tableau des membres est le 3ème sur cette page
        return df['Symbol'].str.replace('.', '-', regex=True).tolist()
    except:
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "BRK-B", "META", "LLY", "AVGO", "JPM"]

def fetch_data_light(tickers, start, end):
    status = st.empty()
    # On télécharge par paquets de 10 seulement
    chunk_size = 10
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    all_closes = []
    all_opens = []
    
    progress = st.progress(0.0)
    for i, chunk in enumerate(chunks):
        status.text(f"📥 Groupe {i+1}/{len(chunks)} (Actions: {chunk[0]}...)")
        try:
            # On force threads=False pour être moins agressif envers Yahoo
            data = yf.download(chunk, start=start, end=end, interval="1mo", progress=False, threads=False)[['Close', 'Open']]
            if not data.empty:
                all_closes.append(data['Close'])
                all_opens.append(data['Open'])
            time.sleep(1) # Pause d'une seconde complète entre chaque groupe
        except:
            continue
        progress.progress((i + 1) / len(chunks))
    
    if not all_closes: return pd.DataFrame(), pd.DataFrame()
    return pd.concat(all_closes, axis=1), pd.concat(all_opens, axis=1)

# --- 4. EXÉCUTION ---
st.title("🚀 Momentum : Étape 3 (Optimisation S&P 100)")
st.info("Note : Nous utilisons le S&P 100 pour éviter les blocages serveurs de Yahoo Finance sur Streamlit Cloud.")

tickers = get_sp100_tickers()

if st.button("Lancer le chargement"):
    s_str = start_date.strftime('%Y-%m-%d')
    e_str = end_date.strftime('%Y-%m-%d')
    
    with st.spinner("Chargement de l'indice..."):
        mkt = yf.download("^GSPC", start=s_str, end=e_str, interval="1mo", progress=False)
    
    closes, opens = fetch_data_light(tickers, s_str, e_str)
    
    if not closes.empty:
        st.session_state['closes'] = closes
        st.session_state['opens'] = opens
        st.session_state['mkt'] = mkt
        st.success(f"✅ Analyse prête sur {len(closes.columns)} actions !")
        st.dataframe(closes.tail(3))
    else:
        st.error("Blocage persistant. Essayez une période plus courte (ex: 2015-2025) pour tester.")

elif 'closes' in st.session_state:
    st.info("Données prêtes. Passons à l'étape 4.")
