import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import yfinance as yf
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Momentum Strategy S&P 500", layout="wide")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_local_data():
    path = 'sp500_data_final.csv'
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df.sort_index()

@st.cache_data
def download_sp500_benchmark(start, end):
    # Télécharge le ^GSPC pour le benchmark et la MM
    data = yf.download("^GSPC", start=start, end=end, interval="1d")
    if isinstance(data.columns, pd.MultiIndex): # Gère le nouveau format yfinance
        data = data['Close']
    else:
        data = data[['Close']]
    data.columns = ['^GSPC']
    return data

# --- INITIALISATION ---
df_assets = load_local_data()

if df_assets is None:
    st.error("❌ Fichier 'sp500_data_final.csv' introuvable.")
    st.stop()

# --- BARRE LATÉRALE ---
st.sidebar.header("🕹️ Paramètres")
data_min, data_max = df_assets.index.min().to_pydatetime(), df_assets.index.max().to_pydatetime()

start_date = st.sidebar.date_input("Début", data_min, min_value=datetime(1970, 1, 1))
end_date = st.sidebar.date_input("Fin", data_max, max_value=datetime(2026, 12, 31))

st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Momentum")
lookback = st.sidebar.slider("Look-back (mois)", 1, 12, 6)
holding = st.sidebar.slider("Holding (mois)", 1, 12, 1)
n_tickers = st.sidebar.slider("Nombre de tickers (N)", 1, 20, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Filtre de Tendance")
ma_window = st.sidebar.slider("Moyenne Mobile S&P 500 (mois)", 2, 24, 10)

# --- LOGIQUE FINANCIÈRE ---
def run_backtest(assets, start, end, lb, hold, n, ma_win):
    # 1. Récupération du benchmark externe
    df_bench = download_sp500_benchmark(start, end)
    
    # 2. Alignement des données
    combined = pd.concat([assets, df_bench], axis=1).ffill().dropna(subset=['^GSPC'])
    combined = combined.loc[pd.Timestamp(start):pd.Timestamp(end)]
    
    benchmark_prices = combined['^GSPC']
    asset_prices = combined.drop(columns=['^GSPC'])

    # 3. Resample mensuel
    m_assets = asset_prices.resample('ME').last()
    m_bench = benchmark_prices.resample('ME').last()
    
    # 4. Calcul MM et Signaux
    ma_bench = m_bench.rolling(window=ma_win).mean()
    returns_assets = m_assets.pct_change()
    returns_bench = m_bench.pct_change()
    momentum_signal = m_assets.pct_change(lb)
    
    strat_returns, dates, trend_bits = [], [], []
    start_idx = max(lb, ma_win)
    
    for i in range(start_idx, len(m_assets) - hold, hold):
        current_date = m_assets.index[i]
        
        # Filtre de tendance sur le S&P 500 téléchargé
        if m_bench.loc[current_date] > ma_bench.loc[current_date]:
            top_n = momentum_signal.loc[current_date].nlargest(n).index
            future_perf = returns_assets.iloc[i+1 : i+1+hold][top_n].mean(axis=1)
            trend_bits.extend([1] * len(future_perf))
        else:
            future_perf = pd.Series(0, index=returns_assets.index[i+1 : i+1+hold])
            trend_bits.extend([0] * len(future_perf))
        
        strat_returns.extend(future_perf.values)
        dates.extend(future_perf.index)

    if not strat_returns: return None
    
    s_strat = pd.Series(strat_returns, index=dates)
    s_bench = returns_bench.loc[dates]
    return (1 + s_strat).cumprod(), (1 + s_bench).cumprod(), s_strat, s_bench, trend_bits

# --- CALCUL MÉTRIQUES ---
def get_metrics(cum, ret):
    tr = (cum.iloc[-1] - 1) * 100
    yrs = (cum.index[-1] - cum.index[0]).days / 365.25
    cagr = ((cum.iloc[-1])**(1/yrs) - 1) * 100 if yrs > 0 else 0
    vol = ret.std() * np.sqrt(12)
    sharpe = (cagr / 100) / vol if vol > 0 else 0
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    return tr, cagr, sharpe, max_dd

# --- INTERFACE ---
if st.button("🚀 Lancer le Backtest (Data Locale + ^GSPC Live)"):
    with st.spinner("Téléchargement du S&P 500 et calcul..."):
        results = run_backtest(df_raw, start_date, end_date, lookback, holding, n_tickers, ma_window)
    
    if results:
        res_s, res_b, ret_s, ret_b, trend_bits = results
        m_s = get_metrics(res_s, ret_s)
        m_b = get_metrics(res_b, ret_b)

        # Tableau Comparatif
        st.subheader("📊 Rapport de Performance")
        comparison_df = pd.DataFrame({
            "Métrique": ["Performance Totale", "CAGR (Annuel)", "Ratio de Sharpe", "Max Drawdown"],
            "Stratégie Momentum": [f"{m_s[0]:,.2f}%", f"{m_s[1]:.2f}%", f"{m_s[2]:.2f}", f"{m_s[3]:.2f}%"],
            "Benchmark S&P 500": [f"{m_b[0]:,.2f}%", f"{m_b[1]:.2f}%", f"{m_b[2]:.2f}", f"{m_b[3]:.2f}%"]
        })
        st.table(comparison_df)

        # Graphique Cumulative
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res_s.index, y=res_s, name="Stratégie", line=dict(color='#1f77b4', width=3)))
        fig.add_trace(go.Scatter(x=res_b.index, y=res_b, name="S&P 500 Live", line=dict(color='#2ca02c', dash='dot')))
        fig.update_layout(title="Performance (Échelle Log)", template="plotly_dark", yaxis_type="log", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Graphique d'Exposition
        fig_exp = go.Figure()
        fig_exp.add_trace(go.Scatter(x=res_s.index, y=trend_bits, fill='tozeroy', name="Exposition", line=dict(color='yellow', width=0)))
        fig_exp.update_layout(title="Exposition Marché (1 = Investi, 0 = Cash)", template="plotly_dark", height=150, yaxis=dict(tickvals=[0, 1]))
        st.plotly_chart(fig_exp, use_container_width=True)
    else:
        st.warning("⚠️ Données insuffisantes. Essayez d'élargir la période de dates.")
