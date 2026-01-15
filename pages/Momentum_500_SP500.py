import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Momentum 500 - Étape 2 (Modifiée)", layout="wide")

# --- 2. BARRE LATÉRALE : PARAMÈTRES ---
st.sidebar.header("⚙️ Paramètres de la Stratégie")

# MODIFICATION : Curseur limité de 1 à 20
num_assets = st.sidebar.slider("Nombre d'actions à détenir", 1, 20, 10)
lookback_months = st.sidebar.slider("Période d'analyse Momentum (mois)", 1, 12, 6)
rotation_freq = st.sidebar.slider("Fréquence de rotation (mois)", 1, 12, 3)
fees_pct = st.sidebar.slider("Frais par transaction (%)", 0.0, 0.5, 0.10, step=0.01) / 100

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Market Timing")
enable_filter = st.sidebar.checkbox("Activer le filtre de tendance", value=True)
ma_window = st.sidebar.slider("Moyenne Mobile S&P 500 (jours)", 50, 250, 150)

st.sidebar.markdown("---")
st.sidebar.header("📅 Période d'Analyse")
# MODIFICATION : Date de début fixée au 01/01/1980 par défaut
start_input = st.sidebar.text_input("Date de début (AAAA/MM/JJ)", "1980/01/01")
end_input = st.sidebar.text_input("Date de fin (AAAA/MM/JJ)", datetime.now().strftime("%Y/%m/%d"))

# --- 3. TITRE PRINCIPAL ---
st.title("🚀 Momentum 500 : Construction Progressive")
st.write("Étape 2 : Ajustement des curseurs et de la période historique")

# --- 4. RÉCAPITULATIF DES PARAMÈTRES ---
st.subheader("Paramètres actuels")
col1, col2, col3 = st.columns(3)

with col1:
    st.write(f"**Actions sélectionnées :** {num_assets} (Max 20)")
    st.write(f"**Look-back :** {lookback_months} mois")

with col2:
    st.write(f"**Rotation :** tous les {rotation_freq} mois")
    st.write(f"**Frais :** {fees_pct*100:.2f}%")

with col3:
    st.write(f"**Filtre de tendance :** {'Activé' if enable_filter else 'Désactivé'}")
    st.write(f"**Début d'analyse :** {start_input}")

# --- 5. FONCTION DE RÉCUPÉRATION DES TICKERS ---
@st.cache_data
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        df = pd.read_html(response.text)[0]
        return df
    except Exception as e:
        st.error(f"Erreur lors de la récupération : {e}")
        return None

df_sp500 = get_sp500_tickers()

if df_sp500 is not None:
    st.success(f"✅ Prêt pour l'analyse historique depuis 1980 sur {len(df_sp500)} actions.")
