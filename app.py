import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="RSI 10 Strategy Tracker", layout="wide")

st.title("📊 Assistant de Stratégie RSI 10 (V2 - 1960/2025)")
st.markdown("""
Cette application surveille le S&P 500 et d'autres indices en utilisant votre stratégie optimisée : 
**Achat si RSI ≥ 50** ou **Achat Panique si RSI < 32**.
""")

# --- BARRE LATÉRALE (PARAMÈTRES) ---
st.sidebar.header("⚙️ Paramètres")
ticker = st.sidebar.text_input("Symbole Yahoo Finance", "^GSPC")

# --- MODIFICATION ICI : PLAGE DE DATES ÉTENDUE ---
start_date = st.sidebar.date_input(
    "Date de début backtest", 
  value=date(1960, 1, 1),      # Date affichée par défaut au chargement
    min_value=date(1960, 1, 1),  # Limite minimum (Bloqué avant 1960)
    max_value=date(2025, 12, 31) # Limite maximum (Bloqué après 2025)
)

fees = st.sidebar.slider("Frais de transaction (%)", 0.0, 0.5, 0.1, step=0.01) / 100

st.sidebar.subheader("Seuils RSI")
threshold_buy = st.sidebar.number_input("Seuil Achat (Tendance)", value=50)
threshold_panic = st.sidebar.number_input("Seuil Achat (Panique)", value=32)

# --- MOTEUR DE CALCUL ---
@st.cache_data
def get_data_and_calc(ticker, start, fees, th_buy, th_panic):
    # On télécharge les données (yfinance gère le format date de streamlit)
    df = yf.download(ticker, start=start, interval="1wk")
    if df.empty: return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[['Close']].copy()
    df.columns = ['price']
    
    # RSI Normal (SMA)
    delta = df['price'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=10).mean()
    avg_loss = loss.rolling(window=10).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Signaux
    df['signal'] = 0
    df.loc[(df['rsi'] >= th_buy) | (df['rsi'] < th_panic), 'signal'] = 1
    
    # Performance
    df['mkt_ret'] = df['price'].pct_change()
    df['strat_ret_raw'] = df['signal'].shift(1) * df['mkt_ret']
    df['trade'] = df['signal'].diff().fillna(0).abs()
    df['net_ret'] = df['strat_ret_raw'] - (df['trade'] * fees)
    
    df['cum_mkt'] = (1 + df['mkt_ret'].fillna(0)).cumprod()
    df['cum_strat'] = (1 + df['net_ret'].fillna(0)).cumprod()
    
    return df

# --- EXÉCUTION ---
data = get_data_and_calc(ticker, start_date, fees, threshold_buy, threshold_panic)

if data is not None:
    # 1. ÉTAT ACTUEL
    last_rsi = data['rsi'].iloc[-1]
    last_signal = data['signal'].iloc[-1]
    last_price = data['price'].iloc[-1]
    
    st.subheader("🚨 Signal Actuel")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Prix Actuel", f"{last_price:,.2f}")
    with col2:
        st.metric("RSI 10 Weekly", f"{last_rsi:.2f}")
    with col3:
        if last_signal == 1:
            if last_rsi < threshold_panic:
                st.success("POSITION : ACHAT (PANIQUE)")
            else:
                st.success("POSITION : ACHAT (TENDANCE)")
        else:
            st.warning("POSITION : CASH (ATTENTE)")

    # 2. GRAPHIQUE INTERACTIF
    st.subheader("📈 Performance Historique (Échelle Log)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['cum_mkt'], name="Indice (Buy & Hold)", line=dict(color='gray', width=1)))
    fig.add_trace(go.Scatter(x=data.index, y=data['cum_strat'], name="Stratégie RSI 10", line=dict(color='green', width=2)))
    fig.update_layout(
        yaxis_type="log", 
        template="plotly_white", 
        height=500, 
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. STATISTIQUES
    st.subheader("📊 Métriques Clés")
    total_ret = (data['cum_strat'].iloc[-1] - 1) * 100
    peak = data['cum_strat'].cummax()
    max_dd = ((data['cum_strat'] - peak) / peak).min() * 100
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Performance Totale", f"{total_ret:,.2f} %")
    c2.metric("Max Drawdown", f"{max_dd:.2f} %")
    c3.metric("Nombre de Trades", int(data['trade'].sum()))

else:
    st.error("Impossible de récupérer les données pour ce symbole.")
  
