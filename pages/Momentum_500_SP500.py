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
    
    df = df.dropna(axis=1, how='all')
    return df

df_raw = load_data()

if df_raw is None:
    st.error("❌ Fichier 'sp500_data_final.csv' introuvable.")
    st.info("Assurez-vous que le fichier est à la racine de votre dépôt GitHub.")
    uploaded_file = st.file_uploader("Ou uploadez le fichier manuellement :", type="csv")
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    else:
        st.stop()

# --- BARRE LATÉRALE (CONTRÔLES) ---
st.sidebar.header("🕹️ Paramètres de Simulation")

min_sim_date = datetime(1970, 1, 1)
max_sim_date = datetime(2026, 12, 31)

data_min = df_raw.index.min().to_pydatetime()
data_max = df_raw.index.max().to_pydatetime()

start_date = st.sidebar.date_input("Début", data_min, min_value=min_sim_date, max_value=max_sim_date)
end_date = st.sidebar.date_input("Fin", data_max, min_value=min_sim_date, max_value=max_sim_date)

st.sidebar.markdown("---")
lookback = st.sidebar.slider("Look-back (mois)", 1, 12, 6)
holding = st.sidebar.slider("Holding (mois)", 1, 12, 1)
n_tickers = st.sidebar.slider("Nombre de tickers (N)", 1, 20, 10)

# --- LOGIQUE FINANCIÈRE ---
def run_backtest(data, start, end, lb, hold, n):
    data = data.loc[str(start):str(end)].copy()
    
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
    if cum_series.empty: return 0, 0, 0, 0
    total_ret = (cum_series.iloc[-1] - 1) * 100
    years = (cum_series.index[-1] - cum_series.index[0]).days / 365.25
    cagr = ((cum_series.iloc[-1])**(1/years) - 1) * 100 if years > 0 else 0
    sharpe = (cagr / 100) / (returns.std() * np.sqrt(12)) if returns.std() != 0 else 0
    peak = cum_series.cummax()
    max_dd = ((cum_series - peak) / peak).min() * 100
    return total_ret, cagr, sharpe, max_dd

# --- INTERFACE ET GRAPHIQUES ---
if st.button("Lancer le Backtest"):
    res_strat, res_bench, ret_strat, ret_bench = run_backtest(
        df_raw, start_date, end_date, lookback, holding, n_tickers
    )
    
    if not res_strat.empty:
        m_s = get_metrics(res_strat, ret_strat)
        m_b = get_metrics(res_bench, ret_bench)
        
        # --- NOUVEL AFFICHAGE TABLEAU STYLISÉ ---
        st.subheader("📊 Rapport de Performance Comparatif")
        
        # Création d'un DataFrame pour le tableau
        comparison_df = pd.DataFrame({
            "Métrique": ["Performance Totale", "CAGR (Rendement Annuel)", "Ratio de Sharpe", "Max Drawdown"],
            "Stratégie Momentum": [f"{m_s[0]:,.2f}%", f"{m_s[1]:.2f}%", f"{m_s[2]:.2f}", f"{m_s[3]:.2f}%"],
            "Benchmark S&P 500": [f"{m_b[0]:,.2f}%", f"{m_b[1]:.2f}%", f"{m_b[2]:.2f}", f"{m_b[3]:.2f}%"]
        })

        # Utilisation de Markdown pour un style plus "Dashboard"
        st.table(comparison_df)

        # Ajout d'une ligne d'Alpha (Surperformance)
        alpha = m_s[1] - m_b[1]
        color = "green" if alpha > 0 else "red"
        st.markdown(f"**Surperformance annuelle (Alpha) :** :{color}[{alpha:.2f}%]")

        # --- GRAPHIQUES ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res_strat.index, y=res_strat, name="Momentum Portfolio", 
                                 line=dict(color='#1f77b4', width=3))) 
        fig.add_trace(go.Scatter(x=res_bench.index, y=res_bench, name="S&P 500 Benchmark", 
                                 line=dict(color='#2ca02c', width=2, dash='dot'))) 
        
        fig.update_layout(
            title="Performance Cumulative (Échelle Logarithmique)",
            template="plotly_dark",
            hovermode="x unified",
            yaxis_type="log",
            xaxis_title="Date",
            yaxis_title="Valeur (Log Scale)"
        )
        st.plotly_chart(fig, use_container_width=True)

        peak = res_strat.cummax()
        drawdown = (res_strat - peak) / peak * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill='tozeroy', name="Drawdown %", line=dict(color='#d62728')))
        fig_dd.update_layout(title="Risque : Drawdown Historique de la Stratégie (%)", template="plotly_dark")
        st.plotly_chart(fig_dd, use_container_width=True)
    else:
        st.error("Données insuffisantes pour la période sélectionnée. Augmentez la plage de dates.")
