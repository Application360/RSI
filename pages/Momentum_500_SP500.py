import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Momentum Strategy S&P 500", layout="wide")

# --- CHARGEMENT DES DONNÉES AVEC GESTION D'ERREUR ---
@st.cache_data
def load_data():
    # Liste des chemins potentiels pour le fichier
    possible_paths = [
        'sp500_data_final.csv', 
        os.path.join(os.path.dirname(__file__), 'sp500_data_final.csv'),
        os.path.join(os.getcwd(), 'sp500_data_final.csv')
    ]
    
    file_to_load = None
    for path in possible_paths:
        if os.path.exists(path):
            file_to_load = path
            break
            
    if file_to_load is None:
        return None

    df = pd.read_csv(file_to_load)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)
    
    # Nettoyage des données vides
    df = df.dropna(axis=1, how='all')
    return df

df_raw = load_data()

if df_raw is None:
    st.error("❌ Fichier 'sp500_data_final.csv' introuvable.")
    st.info("Assurez-vous que le fichier est dans le même dossier que votre script ou à la racine de votre dépôt GitHub.")
    uploaded_file = st.file_uploader("Ou uploadez le fichier manuellement :", type="csv")
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    else:
        st.stop()

# --- BARRE LATÉRALE (CONTRÔLES) ---
st.sidebar.header("🕹️ Paramètres de Simulation")

# Dates
min_date = df_raw.index.min().to_pydatetime()
max_date = df_raw.index.max().to_pydatetime()
start_date = st.sidebar.date_input("Début", min_date, min_value=min_date)
end_date = st.sidebar.date_input("Fin", max_date, max_value=max_date)

st.sidebar.markdown("---")

# Paramètres Momentum
lookback = st.sidebar.slider("Look-back (mois)", 1, 12, 6)
holding = st.sidebar.slider("Holding (mois)", 1, 6, 1)
n_tickers = st.sidebar.slider("Nombre de tickers (N)", 1, 20, 10)

# --- LOGIQUE FINANCIÈRE ---
def run_backtest(data, start, end, lb, hold, n):
    # Filtrage date
    data = data.loc[start:end].copy()
    
    # Création du benchmark (si ^GSPC absent, on prend la moyenne de l'indice)
    if '^GSPC' in data.columns:
        benchmark_prices = data['^GSPC']
        asset_prices = data.drop(columns=['^GSPC'])
    else:
        benchmark_prices = data.mean(axis=1)
        asset_prices = data

    # Resample mensuel
    monthly_assets = asset_prices.resample('ME').last()
    monthly_bench = benchmark_prices.resample('ME').last()
    
    returns_assets = monthly_assets.pct_change()
    returns_bench = monthly_bench.pct_change()
    
    # Signal Momentum (Performance sur période lb)
    momentum_signal = monthly_assets.pct_change(lb)
    
    strat_returns = []
    dates = []
    
    # Loop de rebalancement
    for i in range(lb, len(monthly_assets) - hold, hold):
        current_date = monthly_assets.index[i]
        
        # Sélection des N meilleurs tickers
        top_n = momentum_signal.loc[current_date].nlargest(n).index
        
        # Performance moyenne sur la période de holding suivante
        future_perf = returns_assets.iloc[i+1 : i+1+hold][top_n].mean(axis=1)
        
        strat_returns.extend(future_perf.values)
        dates.extend(future_perf.index)

    # Séries temporelles finales
    s_strat = pd.Series(strat_returns, index=dates)
    s_bench = returns_bench.loc[dates]
    
    return (1 + s_strat).cumprod(), (1 + s_bench).cumprod(), s_strat, s_bench

# --- CALCUL DES MÉTRIQUES ---
def get_metrics(cum_series, returns):
    total_ret = (cum_series.iloc[-1] - 1) * 100
    
    # CAGR
    years = (cum_series.index[-1] - cum_series.index[0]).days / 365.25
    cagr = ((cum_series.iloc[-1])**(1/years) - 1) * 100
    
    # Sharpe (Vol mensuelle -> annuelle)
    sharpe = (cagr / 100) / (returns.std() * np.sqrt(12))
    
    # Max Drawdown
    peak = cum_series.cummax()
    dd = (cum_series - peak) / peak
    max_dd = dd.min() * 100
    
    return total_ret, cagr, sharpe, max_dd

# --- INTERFACE ET GRAPHIQUES ---
if st.button("Lancer le Backtest"):
    res_strat, res_bench, ret_strat, ret_bench = run_backtest(
        df_raw, start_date, end_date, lookback, holding, n_tickers
    )
    
    m_s = get_metrics(res_strat, ret_strat)
    m_b = get_metrics(res_bench, ret_bench)
    
    # Affichage des métriques
    st.subheader("📊 Comparaison des Performances")
    col1, col2, col3, col4 = st.columns(4)
    metrics = ["Total Return", "CAGR", "Sharpe", "Max Drawdown"]
    
    for i, name in enumerate(metrics):
        col1.metric(f"{name} (Strat)", f"{m_s[i]:.2f}{'%' if i != 2 else ''}")
        col2.metric(f"{name} (S&P500)", f"{m_b[i]:.2f}{'%' if i != 2 else ''}")

    # Graphique Principal
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res_strat.index, y=res_strat, name="Momentum Portfolio", line=dict(color='#00FFAA')))
    fig.add_trace(go.Scatter(x=res_bench.index, y=res_bench, name="S&P 500 Benchmark", line=dict(color='white', dash='dash')))
    fig.update_layout(title="Performance Cumulative (Base 1.0)", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Graphique Drawdown
    peak = res_strat.cummax()
    drawdown = (res_strat - peak) / peak * 100
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill='tozeroy', name="Drawdown %", line=dict(color='red')))
    fig_dd.update_layout(title="Risque : Drawdown Historique (%)", template="plotly_dark")
    st.plotly_chart(fig_dd, use_container_width=True)
