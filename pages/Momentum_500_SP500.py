import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Momentum 500 - Étape 3 (Ultra-Robuste)", layout="wide")

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

def fetch_data_robust(tickers, start, end, column="Close"):
    """Télécharge une colonne spécifique par petits paquets de 20"""
    all_data = []
    chunk_size = 20 # Paquets plus petits pour éviter le rejet
    ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    for i, chunk in enumerate(ticker_chunks):
        status_text.text(f"Téléchargement {column} : Groupe {i+1}/{len(ticker_chunks)}...")
        try:
            # On ne demande qu'une seule colonne à la fois pour alléger la requête
            data = yf.download(chunk, start=start, end=end, interval="1mo", progress=False)[column]
            if not data.empty:
                all_data.append(data)
            time.sleep(0.2) # Pause très courte
        except Exception as e:
            st.error(f"Erreur groupe {i+1}: {e}")
        
        progress_bar.progress((i + 1) / len(ticker_chunks))
    
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, axis=1)

# --- 4. EXÉCUTION ---
st.title("🚀 Momentum 500 : Étape 3 (Méthode Ultra-Robuste)")
st.info("Le téléchargement est divisé en deux phases (Clôtures puis Ouvertures) pour garantir le succès.")

tickers = get_sp500_tickers()

if st.button("Lancer le téléchargement haute fiabilité"):
    s_date = start_input.replace('/', '-')
    e_date = end_input.replace('/', '-')
    
    # 1. Téléchargement de l'indice
    with st.spinner("Téléchargement du S&P 500..."):
        mkt = yf.download("^GSPC", start=s_date, end=e_date, interval="1mo", progress=False)
    
    # 2. Téléchargement des Clôtures
    closes = fetch_data_robust(tickers, s_date, e_date, "Close")
    
    # 3. Téléchargement des Ouvertures
    opens = fetch_data_robust(tickers, s_date, e_date, "Open")
    
    if not closes.empty and not opens.empty:
        st.success(f"✅ Succès ! {len(closes.columns)} actions chargées de {s_date} à {e_date}.")
        
        tab1, tab2 = st.tabs(["📊 Clôtures (Momentum)", "📈 Ouvertures (Exécution)"])
        with tab1:
            st.dataframe(closes.tail(), use_container_width=True)
        with tab2:
            st.dataframe(opens.tail(), use_container_width=True)
            
        # Sauvegarde
        st.session_state['closes'] = closes
        st.session_state['opens'] = opens
        st.session_state['mkt'] = mkt
    else:
        st.error("Le serveur Yahoo a encore rejeté une partie des données. Réessayez ou réduisez la période.")

elif 'closes' in st.session_state:
    st.success("Données en mémoire. Prêt pour l'Étape 4.")
