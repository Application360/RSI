import streamlit as st
import pandas as pd
import requests

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Momentum 500 - Step 1", layout="wide")

# --- 2. BARRE LATÉRALE (STRUCTURE VIDE POUR LE MOMENT) ---
st.sidebar.header("⚙️ Paramètres")
st.sidebar.info("Nous ajouterons les curseurs à l'étape suivante.")

# --- 3. TITRE PRINCIPAL ---
st.title("🚀 Momentum 500 : Construction Progressive")
st.write("Étape 1 : Vérification de la liste des actions du S&P 500")

# --- 4. FONCTION DE RÉCUPÉRATION DES TICKERS ---
@st.cache_data
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        # On récupère le premier tableau de la page
        df = pd.read_html(response.text)[0]
        return df
    except Exception as e:
        st.error(f"Erreur lors de la récupération : {e}")
        return None

# --- 5. EXÉCUTION ET AFFICHAGE ---
df_sp500 = get_sp500_tickers()

if df_sp500 is not None:
    st.success(f"✅ Connexion réussie ! {len(df_sp500)} actions trouvées.")
    
    # On affiche un aperçu pour être sûr
    st.subheader("Aperçu des composants du S&P 500")
    st.dataframe(df_sp500[['Symbol', 'Security', 'GICS Sector']].head(10), use_container_width=True)
    
    # Nettoyage des symboles pour Yahoo Finance (ex: BRK.B -> BRK-B)
    tickers_list = df_sp500['Symbol'].str.replace('.', '-', regex=True).tolist()
    st.write(f"**Liste prête pour le téléchargement :** {tickers_list[:10]} ...")
