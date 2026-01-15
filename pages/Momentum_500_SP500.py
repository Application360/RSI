import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Momentum Pro : Analyse Long Terme", layout="wide")

# --- TITRE ET EN-TÊTE ---
st.title("🚀 Momentum Pro : Analyse Long Terme")
st.header(f"1999-{date.today().year}")

# --- BARRE LATÉRALE (PARAMÈTRES) ---
st.sidebar.header("⚙️ Paramètres")
ticker_ref = st.sidebar.text_input("Symbole Indice de référence", "^GSPC")
risk_free_rate = st.sidebar.slider("Taux sans risque (Sharpe)", 0.0, 0.10, 0.02, 0.005)

# --- FONCTIONS DE CALCUL ---
def calc_max_drawdown(cum_series):
    peak = cum_series.cummax()
    drawdown = (cum_series - peak) / peak
    return drawdown.min() * 100

@st.cache_data
def get_data(ticker):
    # On télécharge les données pour l'indice de référence
    data = yf.download(ticker, start="1999-01-01", interval="1wk", group_by='column')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    df = data[['Close']].copy()
    df.columns = ['price']
    df['mkt_ret'] = df['price'].pct_change()
    
    # --- LOGIQUE MOMENTUM ---
    # Ici, insérez votre propre calcul de stratégie. 
    # Pour l'exemple, nous simulons les rendements de la stratégie.
    df['strat_ret'] = df['mkt_ret'] * 1.12 # Simulation de surperformance
    
    df['cum_mkt'] = (1 + df['mkt_ret'].fillna(0)).cumprod()
    df['cum_strat'] = (1 + df['strat_ret'].fillna(0)).cumprod()
    return df

data = get_data(ticker_ref)

# --- CALCULS DES KPI ---
years = (data.index[-1] - data.index[0]).days / 365.25

# Métriques Stratégie
perf_strat = (data['cum_strat'].iloc[-1] - 1) * 100
cagr_strat = (data['cum_strat'].iloc[-1] ** (1/years) - 1) * 100
vol_strat = data['strat_ret'].std() * np.sqrt(52) * 100
mdd_strat = calc_max_drawdown(data['cum_strat'])
sharpe_strat = ((data['strat_ret'].mean() * 52) - risk_free_rate) / (data['strat_ret'].std() * np.sqrt(52))

# Métriques Marché
perf_mkt = (data['cum_mkt'].iloc[-1] - 1) * 100
cagr_mkt = (data['cum_mkt'].iloc[-1] ** (1/years) - 1) * 100
vol_mkt = data['mkt_ret'].std() * np.sqrt(52) * 100
mdd_mkt = calc_max_drawdown(data['cum_mkt'])
sharpe_mkt = ((data['mkt_ret'].mean() * 52) - risk_free_rate) / (data['mkt_ret'].std() * np.sqrt(52))

# --- AFFICHAGE DU TABLEAU DE BORD ---
st.subheader("📊 Comparaison des Performances")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 🔹 Ma Stratégie")
    # Première ligne de KPI
    k1, k2, k3 = st.columns(3)
    k1.metric("Perf. Totale", f"{perf_strat:.1f}%")
    k2.metric("CAGR Net", f"{cagr_strat:.2f}%")
    k3.metric("Sharpe", f"{sharpe_strat:.2f}")
    
    # Deuxième ligne de KPI
    k4, k5, k6 = st.columns(3)
    k4.metric("Max DD", f"{mdd_strat:.1f}%")
    k5.metric("Volatilité", f"{vol_strat:.1f}%")
    k6.metric("Trades", "131") # Remplacez par votre calcul réel de trades

with col_right:
    st.markdown("### 🔸 S&P 500")
    # Première ligne de KPI
    s1, s2, s3 = st.columns(3)
    s1.metric("Perf. Totale", f"{perf_mkt:.1f}%")
    s2.metric("CAGR", f"{cagr_mkt:.2f}%")
    s3.metric("Sharpe", f"{sharpe_mkt:.2f}")
    
    # Deuxième ligne de KPI
    s4, s5 = st.columns(2)
    s4.metric("Max DD", f"{mdd_mkt:.1f}%")
    s5.metric("Volatilité", f"{vol_mkt:.1f}%")

st.write("---")

# --- GRAPHIQUE ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['cum_strat'], name="Ma Stratégie", line=dict(color='#3366CC', width=2.5)))
fig.add_trace(go.Scatter(x=data.index, y=data['cum_mkt'], name="S&P 500", line=dict(color='#FF6633', width=1.5, dash='dot')))
fig.update_layout(template="plotly_white", height=500, yaxis_type="log", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)
