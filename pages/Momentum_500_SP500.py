import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Momentum 500 - Étape 3 (Finale)", layout="wide")

# --- 2. SIDEBAR ---
st.sidebar.header("⚙️ Paramètres")
num_assets = st.sidebar.slider("Nombre d'actions à détenir", 1, 20, 10)
lookback_months = st.sidebar.slider("Période d'analyse Momentum (mois)", 1, 12, 6)
rotation_freq = st.sidebar.slider("Fréquence de rotation (mois)", 1, 12, 3)
fees_pct = st.sidebar.slider("Frais par transaction (%)", 0.0, 0.5, 0.10, step=0.01) / 100

st.sidebar.markdown("---")
# Utilisation d'un sélecteur de date pour éviter les erreurs de frappe
start_date = st.sidebar.date_input("Date de début", datetime(1980, 1, 1))
end_date = st.sidebar.date_input("Date de fin", datetime.now())

# --- 3. FONCTIONS DE RÉCUPÉRATION ---

@st.cache_data
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    df = pd.read_html(response.text)[0]
    return df['Symbol'].str.replace('.', '-', regex=True).tolist()

def fetch_safe(tickers, start, end, column="Close"):
    """Téléchargement ultra-sécurisé par paquets de 10"""
    all_data = []
    chunk_size = 10 
    ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    pbar = st.progress(0.0)
    msg = st.empty()
    
    for i, chunk in enumerate(ticker_chunks):
        msg.text(f"📥 {column} : Paquet {i+1}/{len(ticker_chunks)}")
        try:
            # Téléchargement avec timeout court
            data = yf.download(chunk, start=start, end=end, interval="1mo", progress=False, group_by='column')[column]
            if not data.empty:
                # Si une seule action est retournée, c'est une Series, on la convertit en DataFrame
                if isinstance(data, pd.Series):
                    data = data.to_frame()
                all_data.append(data)
            
            # Pause de sécurité pour éviter le blocage IP
            time.sleep(0.3)
        except Exception:
            continue # On ignore les erreurs individuelles de paquets
        
        pbar.progress((i + 1) / len(ticker_chunks))
    
    if not all_data:
        return pd.DataFrame()
    
    # Fusion en éliminant les doublons potentiels
    return pd.concat(all_data, axis=1).sort_index()

# --- 4. EXÉCUTION ---
st.title("🚀 Momentum 500 : Étape 3 (Mode Survie)")
st.warning("⚠️ Attention : Remonter à 1980 avec 500 actions est très lourd. Si cela échoue encore, essayez de mettre 2000/01/01 comme date de début.")

tickers = get_sp500_tickers()

if st.button("Lancer le chargement haute sécurité"):
    # Nettoyage des dates pour Yahoo
    s_str = start_date.strftime('%Y-%m-%d')
    e_str = end_date.strftime('%Y-%m-%d')
    
    # 1. Indice
    with st.spinner("Indice..."):
        mkt = yf.download("^GSPC", start=s_str, end=e_str, interval="1mo", progress=False)
    
    # 2. Clôtures
    closes = fetch_safe(tickers, s_str, e_str, "Close")
    
    # 3. Ouvertures
    opens = fetch_safe(tickers, s_str, e_str, "Open")
    
    if not closes.empty and not opens.empty:
        # On ne garde que les tickers présents dans les deux fichiers
        common_cols = closes.columns.intersection(opens.columns)
        closes = closes[common_cols]
        opens = opens[common_cols]
        
        st.success(f"✅ Analyse possible sur {len(common_cols)} actions.")
        
        st.session_state['closes'] = closes
        st.session_state['opens'] = opens
        st.session_state['mkt'] = mkt
        
        st.write("**Dernières lignes des Clôtures :**")
        st.dataframe(closes.tail(3))
    else:
        st.error("Échec partiel ou total. Veuillez réduire la période (ex: 1990 ou 2000).")

elif 'closes' in st.session_state:
    st.success("Données prêtes en mémoire.")
