import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Momentum Strategy S&P 500", layout="wide")

# --- CHARGEMENT DES DONNÉES ---
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
    
    # Nettoyage
    df = df.dropna(axis=1, how='all')
    return df

df_raw = load_data()

if df_raw is None:
    st.error("❌ Fichier 'sp500_data_final.csv' introuvable.")
    st.stop()

# --- BARRE LATÉRALE (CONTRÔLES) ---
st.sidebar.header("🕹️ Paramètres de Simulation")

# (4) Dates de 1970 à 2026
min_sim_date = datetime(1970, 1, 1)
max_sim_date = datetime(2026, 12, 31)

# Ajustement selon les données réelles disponibles pour éviter les erreurs hors limites
data_min = df_raw.index.min().to_pydatetime()
data_max = df_raw.index.max().to_pydatetime()

start_date = st.sidebar.date_input("Début", data_min, min_value=min_sim_date, max_value=max_sim_date)
end_date = st.sidebar.date_input("Fin", data_max, min_value=min_sim_date, max_value=max_sim_date)

st.sidebar.markdown("---")

# Paramètres Momentum
lookback = st.sidebar.slider("Look-back (mois)", 1, 12, 6)
# (4) Holding curseur entre 1 et 12 mois
holding = st.sidebar.slider("Holding (mois)", 1, 12, 1)
n_tickers = st.sidebar.slider("Nombre de tickers (N)", 1, 20, 10)

# --- LOGIQUE DE BACKTEST ---
def run_backtest(data, start, end, lb, hold, n):
    data = data.loc[str(start):str(end)].copy()
    
    # Gestion du Benchmark
    if '^GSPC' in data.columns:
        benchmark_prices = data['^GSPC']
        asset_prices = data.drop(columns=['^GSPC'])
    else:
        benchmark_prices = data.mean(axis=1)
        asset_prices = data

    monthly_assets = asset_prices.resample('ME').last()
    monthly_bench = benchmark_prices.resample('ME').last()
    
    returns_assets = monthly_assets.pct_change()
    returns_bench = monthly_bench.pct_change()
    
    momentum_signal = monthly_assets.pct_change(lb)
    
    strat_returns = []
    dates = []
    
    for i in range(lb, len(monthly_assets) - hold, hold):
        current_date = monthly_assets.index[i]
        top_n = momentum_signal.loc[current_date].nlargest(n).index
        future_perf = returns_assets.iloc[i+1 : i+1+hold][top_n].mean(axis=1)
        strat_returns.extend(future_perf.values)
        dates.extend(future_perf.index)

    s_strat = pd.Series(strat_returns, index=dates)
    s_bench = returns_bench.loc[dates]
    
    return (1 + s_strat).cumprod(), (1 + s_bench).cumprod(), s_strat, s_bench

# --- CALCUL DES MÉTRIQUES ---
def get_metrics(cum_series, returns):
    if len(cum_series) < 2: return [0, 0, 0, 0]
    total_ret = (cum_series.iloc[-1] - 1) * 100
    years = (cum_series.index[-1] - cum_series.index[0]).days / 365.25
    cagr = ((cum_series.iloc[-1])**(1/years) - 1) * 100 if years > 0 else 0
    sharpe = (cagr / 100) / (returns.std() * np.sqrt(12)) if returns.std() != 0 else 0
    peak = cum_series.cummax()
    max_dd = ((cum_series - peak) / peak).min() * 100
    return total_ret, cagr, sharpe, max_dd

# --- AFFICHAGE ---
if st.button("Lancer l'Analyse"):
    res_strat, res_bench, ret_strat, ret_bench = run_backtest(
        df_raw, start_date, end_date, lookback, holding, n_tickers
    )
    
    if not res_strat.empty:
        m_s = get_metrics(res_strat, ret_strat)
        m_b = get_metrics(res_bench, ret_bench)
        
        st.subheader("📊 Résultats")
        c1, c2, c3, c4 = st.columns(4)
        met_list = ["Total Return", "CAGR", "Sharpe", "Max Drawdown"]
        for i, name in enumerate(met_list):
            c1.metric(f"{name} (Strat)", f"{m_s[i]:.2f}{'%' if i != 2 else ''}")
            c2.metric(f"{name} (S&P500)", f"{m_b[i]:.2f}{'%' if i != 2 else ''}")

        # (1) Graphique en base logarithmique
        # (2) Couleurs Bleu (Strat) et Vert (S&P 500)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res_strat.index, y=res_strat, name="Stratégie Momentum", 
                                 line=dict(color='#1f77b4', width=2.5))) # Bleu
        fig.add_trace(go.Scatter(x=res_bench.index, y=res_bench, name="S&P 500", 
                                 line=dict(color='#2ca02c', width=2))) # Vert
        
        fig.update_layout(
            title="Performance Cumulative (Échelle Log)",
            yaxis_type="log", # (1) Log Scale
            template="plotly_dark",
            xaxis_title="Années",
            yaxis_title="Valeur du portefeuille (Base 1)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown
        peak = res_strat.cummax()
        drawdown = (res_strat - peak) / peak * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill='tozeroy', 
                                    name="Drawdown %", line=dict(color='orange')))
        fig_dd.update_layout(title="Risque : Drawdown Stratégie (%)", template="plotly_dark")
        st.plotly_chart(fig_dd, use_container_width=True)
    else:
        st.warning("Aucune donnée disponible pour cette période ou ces paramètres.")
