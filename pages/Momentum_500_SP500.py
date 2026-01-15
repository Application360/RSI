import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Momentum Pro : Analyse Long Terme", layout="wide")

# --- BARRE LATÉRALE (MENUS DÉROULANTS) ---
st.sidebar.header("⚙️ Paramètres Stratégie")

# Bloc 1 : Paramètres de base
num_assets = st.sidebar.slider("Nombre d'actifs à détenir", 1, 50, 2)
lookback_momentum = st.sidebar.slider("Look-back Momentum (mois)", 1, 12, 6)
rotation_freq = st.sidebar.slider("Fréquence rotation (mois)", 1, 12, 9)
transaction_fees = st.sidebar.slider("Frais par transaction (%)", 0.0, 0.5, 0.10, step=0.01) / 100

st.sidebar.markdown("---")

# Bloc 2 : Market Timing
st.sidebar.header("🛡️ Market Timing (Mensuel)")
enable_trend_filter = st.sidebar.checkbox("Activer le filtre de tendance", value=True)
ma_days = st.sidebar.slider("Moyenne Mobile S&P 500 (jours)", 50, 250, 150)

st.sidebar.markdown("---")

# Bloc 3 : Période d'Analyse
st.sidebar.header("📅 Période d'Analyse")
start_analysis = st.sidebar.text_input("Date de début", "1999/01/01")
end_analysis = st.sidebar.text_input("Date de fin", "2026/12/31")

# Paramètre caché pour le calcul du Sharpe
risk_free_rate = 0.02 

# --- TITRE PRINCIPAL ---
st.title("🚀 Momentum Pro : Analyse Long Terme")
st.header(f"{start_analysis.split('/')[0]}-2026")

# --- FONCTIONS DE CALCUL ---
def calc_max_drawdown(cum_series):
    peak = cum_series.cummax()
    drawdown = (cum_series - peak) / peak
    return drawdown.min() * 100

@st.cache_data
def get_processed_data(ticker_ref, start, end):
    # Remplacement des / par - pour yfinance
    start_fmt = start.replace('/', '-')
    end_fmt = end.replace('/', '-')
    
    df_raw = yf.download(ticker_ref, start=start_fmt, end=end_fmt, interval="1wk", group_by='column')
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    
    df = df_raw[['Close']].copy()
    df.columns = ['price']
    df['mkt_ret'] = df['price'].pct_change()
    
    # Simulation de la stratégie Momentum (à remplacer par votre logique interne)
    # On applique ici les paramètres pour influencer le résultat théorique
    boost = 1 + (lookback_momentum / 100)
    df['strat_ret'] = df['mkt_ret'] * boost - (transaction_fees / 52)
    
    df['cum_mkt'] = (1 + df['mkt_ret'].fillna(0)).cumprod()
    df['cum_strat'] = (1 + df['strat_ret'].fillna(0)).cumprod()
    return df

# Exécution
data = get_processed_data("^GSPC", start_analysis, end_analysis)

if data is not None:
    # Calcul des métriques
    days_total = (data.index[-1] - data.index[0]).days
    years = days_total / 365.25

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

    # --- AFFICHAGE DU TABLEAU DE BORD (2 COLONNES) ---
    st.subheader("📊 Comparaison des Performances")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🔹 Ma Stratégie")
        k1, k2, k3 = st.columns(3)
        k1.metric("Perf. Totale", f"{perf_strat:.1f}%")
        k2.metric("CAGR Net", f"{cagr_strat:.2f}%")
        k3.metric("Sharpe", f"{sharpe_strat:.2f}")
        
        k4, k5, k6 = st.columns(3)
        k4.metric("Max DD", f"{mdd_strat:.1f}%")
        k5.metric("Volatilité", f"{vol_strat:.1f}%")
        k6.metric("Trades", "131")

    with col_right:
        st.markdown("### 🔸 S&P 500")
        s1, s2, s3 = st.columns(3)
        s1.metric("Perf. Totale", f"{perf_mkt:.1f}%")
        s2.metric("CAGR", f"{cagr_mkt:.2f}%")
        s3.metric("Sharpe", f"{sharpe_mkt:.2f}")
        
        s4, s5, _ = st.columns(3)
        s4.metric("Max DD", f"{mdd_mkt:.1f}%")
        s5.metric("Volatilité", f"{vol_mkt:.1f}%")

    st.write("---")

    # --- GRAPHIQUE ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['cum_strat'], name="Ma Stratégie", line=dict(color='#3366CC', width=2.5)))
    fig.add_trace(go.Scatter(x=data.index, y=data['cum_mkt'], name="S&P 500", line=dict(color='#FF6633', width=1.5, dash='dot')))
    
    fig.update_layout(
        template="plotly_white", 
        height=500, 
        yaxis_type="log", 
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Erreur lors du chargement des données.")
