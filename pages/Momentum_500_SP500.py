import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Momentum 500 - Étape 3", layout="wide")

# --- 2. SIDEBAR (Paramètres de l'étape 2) ---
st.sidebar.header("⚙️ Paramètres")
num_assets = st.sidebar.slider("Nombre d'actions à détenir", 1, 20, 10)
lookback_months = st.sidebar.slider("Période d'analyse Momentum (mois)", 1, 12, 6)
rotation_freq = st.sidebar.slider("Fréquence de rotation (mois)", 1, 12, 3)
fees_pct = st.sidebar.slider("Frais par transaction (%)", 0.0, 0.5, 0.10, step=0.01) / 100

st.sidebar.markdown("---")
start_input = st.sidebar.text_input("Date de début", "1980/01/01")
end_input = st.sidebar.text_input("Date de fin", datetime.now().strftime("%Y/%m/%d"))

# --- 3. LOGIQUE DE TÉLÉCHARGEMENT ---

@st.cache_data
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    df = pd.read_html(response.text)[0]
    return df['Symbol'].str.replace('.', '-', regex=True).tolist()

@st.cache_data(show_spinner=False)
def download_monthly_data(tickers, start, end):
    s_date = start.replace('/', '-')
    e_date = end.replace('/', '-')
    
    # On télécharge l'indice de référence (^GSPC)
    mkt_data = yf.download("^GSPC", start=s_date, end=e_date, interval="1mo", progress=False)
    
    # Téléchargement groupé des 500 actions
    # On récupère Open et Close pour répondre à votre besoin de précision
    with st.spinner("Téléchargement des données mensuelles (Open/Close) pour 500 actions..."):
        raw_data = yf.download(tickers, start=s_date, end=e_date, interval="1mo", progress=False)
    
    return raw_data, mkt_data

# --- 4. EXÉCUTION ---
st.title("🚀 Momentum 500 : Étape 3")
st.write("Chargement des données historiques (Clôture et Ouverture du mois suivant)")

tickers = get_sp500_tickers()

if st.button("Lancer le téléchargement des données"):
    data, mkt = download_monthly_data(tickers, start_input, end_input)
    
    if not data.empty:
        st.success("✅ Téléchargement terminé avec succès !")
        
        # Extraction des clôtures et ouvertures
        # Note : yfinance renvoie un MultiIndex [Price, Ticker]
        closes = data['Close']
        opens = data['Open']
        
        # Affichage des structures pour vérification
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Aperçu des cours de clôture (Closes) :**")
            st.dataframe(closes.tail(), use_container_width=True)
        with col2:
            st.write("**Aperçu des cours d'ouverture (Opens) :**")
            st.dataframe(opens.tail(), use_container_width=True)
            
        # Stockage temporaire en session pour l'étape suivante
        st.session_state['closes'] = closes
        st.session_state['opens'] = opens
        st.session_state['mkt'] = mkt
    else:
        st.error("Le téléchargement a échoué. Vérifiez votre connexion ou les dates.")

elif 'closes' in st.session_state:
    st.info("Données déjà présentes en mémoire.")
else:
    st.warning("Cliquez sur le bouton ci-dessus pour charger les données (cela peut prendre 1 à 2 minutes la première fois).")
