import streamlit as st
import yfinance as download
import pandas as pd
import plotly.graph_objects as go
from datetime import date

# Configuration de la page
st.set_page_config(page_title="Stratégie RSI 10 - V2", layout="wide")

st.title("📊 Assistant de Stratégie RSI 10 (Version Étendue)")

# --- BARRE LATÉRALE (PARAMÈTRES) ---
st.sidebar.header("⚙️ Paramètres")

symbol = st.sidebar.text_input("Symbole Yahoo Finance", value="^GSPC")

# Modification demandée : Plage de dates étendue de 1960 à 2025
start_date = st.sidebar.date_input(
    "Date de début backtest", 
    value=date(1960, 1, 1),
    min_value=date(1960, 1, 1),
    max_value=date(2025, 12, 31)
)

fee = st.sidebar.slider("Frais de transaction (%)", 0.0, 0.5, 0.0, 0.05) / 100

st.sidebar.subheader("Seuils RSI")
buy_trend = st.sidebar.number_input("Seuil Achat (Tendance)", value=50)
buy_panic = st.sidebar.number_input("Seuil Achat (Panique)", value=32)

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data(ticker, start):
    data = download.download(ticker, start=start, interval="1wk")
    return data

data = load_data(symbol, start_date)

if not data.empty:
    # Calcul du RSI 10 semaines
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=10).mean()
    rs = gain / loss
    data['RSI_10'] = 100 - (100 / (1 + rs))

    # --- LOGIQUE DE LA STRATÉGIE ---
    data['Signal'] = 0
    # Achat si RSI > 50 OU RSI < 32
    data.loc[(data['RSI_10'] > buy_trend) | (data['RSI_10'] < buy_panic), 'Signal'] = 1
    
    # Calcul des rendements
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']
    
    # Prise en compte des frais (simplifié : à chaque changement de position)
    data['Trades'] = data['Signal'].diff().fillna(0).abs()
    data['Strategy_Returns'] = data['Strategy_Returns'] - (data['Trades'] * fee)

    # Performance cumulée
    data['Cum_Market'] = (1 + data['Returns']).cumprod()
    data['Cum_Strategy'] = (1 + data['Strategy_Returns']).cumprod()

    # --- AFFICHAGE DES RÉSULTATS ---
    last_rsi = data['RSI_10'].iloc[-1]
    last_price = data['Close'].iloc[-1].values[0] if isinstance(data['Close'].iloc[-1], pd.Series) else data['Close'].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Prix Actuel", f"{last_price:,.2f}")
    col2.metric("RSI 10 Weekly", f"{last_rsi:.2f}")
    
    status = "ACHAT" if (last_rsi > buy_trend or last_rsi < buy_panic) else "CASH (ATTENTE)"
    color = "green" if status == "ACHAT" else "orange"
    col3.markdown(f"**POSITION :** :{color}[{status}]")

    # Graphique Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Cum_Market'], name="S&P 500 (Buy & Hold)", line=dict(color='gray', width=1)))
    fig.add_trace(go.Scatter(x=data.index, y=data['Cum_Strategy'], name="Stratégie RSI 10", line=dict(color='green', width=2)))
    
    fig.update_layout(
        title="Performance Historique (Échelle Log)",
        yaxis_type="log",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"Données analysées du {start_date} au {date.today()}")

else:
    st.error("Erreur lors du chargement des données. Vérifiez le symbole.")
  
