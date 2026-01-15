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
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df.dropna(axis=1, how='all')

df_raw = load_data()

if df_raw is None:
    st.error("❌ Fichier 'sp500_data_final.csv' introuvable.")
    st.stop()

# --- BARRE LATÉRALE (CONTRÔLES) ---
st.sidebar.header("🕹️ Paramètres de Simulation")
min_sim_date, max_sim_date = datetime(1970, 1, 1), datetime(2026, 12, 31)
data_min, data_max = df_raw.index.min().to_pydatetime(), df_raw.index.max().to_pydatetime()

start_date = st.sidebar.date_input("Début", data_min, min_value=min_sim_date, max_value=max_sim_date)
end_date = st.sidebar.date_input("Fin", data_max, min_value=min_sim_date, max_value=max_sim_date)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Stratégie Momentum")
lookback = st.sidebar.slider("Look-back (mois)", 1, 12, 6)
holding = st.sidebar.slider("Holding (mois)", 1, 12, 1)
n_tickers = st.sidebar.slider("Nombre de tickers (N)", 1, 20, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Filtre de Tendance (Market Timing)")
# Curseur pour la moyenne mobile sur le S&P 500
ma_window = st.sidebar.slider("Moyenne Mobile S&P 500 (mois)", 2, 12, 10, 
                              help="Si le S&P 500 est sous cette moyenne, on passe en Cash (0% rendement).")

# --- LOGIQUE FINANCIÈRE ---
def run_backtest(data, start, end, lb, hold, n, ma_win):
    data = data.loc[str(start):str(end)].copy()
    
    # Identification du Benchmark
    benchmark_prices = data['^GSPC'] if '^GSPC' in data.columns else data.mean(axis=1)
    asset_prices = data.drop(columns=['^GSPC']) if '^GSPC' in data.columns else data

    # Resample mensuel
    monthly_assets = asset_prices.resample('ME').last()
    monthly_bench = benchmark_prices.resample('ME').last()
    
    # Calcul de la Moyenne Mobile sur le Benchmark
    ma_bench = monthly_bench.rolling(window=ma_win).mean()
    
    returns_assets = monthly_assets.pct_change()
    returns_bench = monthly_bench.pct_change()
    
    momentum_signal = monthly_assets.pct_change(lb)
    
    strat_returns = []
    dates = []
    trend_status = [] # Pour savoir quand on était investi ou cash

    # Loop de rebalancement
    for i in range(lb, len(monthly_assets) - hold, hold):
        current_date = monthly_assets.index[i]
        
        # --- RÈGLE DU FILTRE DE TENDANCE ---
        # Si le S&P 500 à la date de décision est inférieur à sa Moyenne Mobile
        if monthly_bench.loc[current_date] < ma_bench.loc[current_date]:
            # ON PASSE CASH : Rendement de 0 pour la période de holding
            future_perf = pd.Series(0, index=returns_assets.index[i+1 : i+1+hold])
            trend_status.extend([0] * len(future_perf))
        else:
            # ON EST INVESTI : Sélection des N meilleurs
            top_n = momentum_signal.loc[current_date].nlargest(n).index
            future_perf = returns_assets.iloc[i+1 : i+1+hold][top_n].mean(axis=1)
            trend_status.extend([1] * len(future_perf))
        
        strat_returns.extend(future_perf.values)
        dates.extend(future_perf.index)

    s_strat = pd.Series(strat_returns, index=dates)
    s_bench = returns_bench.loc[dates]
    
    return (1 + s_strat).cumprod(), (1 + s_bench).cumprod(), s_strat, s_bench, trend_status

def get_metrics(cum, ret):
    if cum.empty: return [0]*4
    tr = (cum.iloc[-1] - 1) * 100
    yrs = (cum.index[-1] - cum.index[0]).days / 365.25
    cagr = ((cum.iloc[-1])**(1/yrs) - 1) * 100 if yrs > 0 else 0
    vol = ret.std() * np.sqrt(12)
    sharpe = (cagr / 100) / vol if vol != 0 else 0
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    return tr, cagr, sharpe, max_dd

# --- INTERFACE ET GRAPHIQUES ---
if st.button("🚀 Lancer le Backtest avec Filtre"):
    res_s, res_b, ret_s, ret_b, trend_bits = run_backtest(
        df_raw, start_date, end_date, lookback, holding, n_tickers, ma_window
    )
    
    if not res_s.empty:
        m_s = get_metrics(res_s, ret_s)
        m_b = get_metrics(res_b, ret_b)
        
        # --- TABLEAU COMPARATIF STYLISÉ ---
        st.subheader("📊 Rapport de Performance avec Filtre de Tendance")
        comparison_df = pd.DataFrame({
            "Métrique": ["Performance Totale", "CAGR (Annuel)", "Ratio de Sharpe", "Max Drawdown"],
            "Stratégie Momentum + Filtre": [f"{m_s[0]:,.2f}%", f"{m_s[1]:.2f}%", f"{m_s[2]:.2f}", f"{m_s[3]:.2f}%"],
            "Benchmark S&P 500": [f"{m_b[0]:,.2f}%", f"{m_b[1]:.2f}%", f"{m_b[2]:.2f}", f"{m_b[3]:.2f}%"]
        })
        st.table(comparison_df)

        # Calcul Alpha et Taux d'exposition
        exposure = (sum(trend_bits) / len(trend_bits)) * 100
        alpha = m_s[1] - m_b[1]
        c1, c2 = st.columns(2)
        c1.markdown(f"**Alpha (Surperformance annuelle) :** :{'green' if alpha > 0 else 'red'}[{alpha:.2f}%]")
        c2.markdown(f"**Taux d'exposition au marché :** `{exposure:.1f}%` du temps")

        # --- GRAPHIQUE PRINCIPAL (LOG) ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res_s.index, y=res_s, name="Stratégie (Momentum + Cash Filter)", line=dict(color='#1f77b4', width=3)))
        fig.add_trace(go.Scatter(x=res_b.index, y=res_b, name="S&P 500 Benchmark", line=dict(color='#2ca02c', width=2, dash='dot')))
        
        fig.update_layout(
            title=f"Performance Cumulative (Log Scale) - Filtre MM {ma_window} mois",
            template="plotly_dark", yaxis_type="log", hovermode="x unified",
            xaxis_title="Date", yaxis_title="Valeur (Base 1)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- GRAPHIQUE D'EXPOSITION (CASH VS INVESTI) ---
        fig_exp = go.Figure()
        fig_exp.add_trace(go.Scatter(x=res_s.index, y=trend_bits, fill='tozeroy', name="Investi (1) / Cash (0)", line=dict(color='yellow', width=0.5)))
        fig_exp.update_layout(title="Périodes d'activité (Jaune = Investi, Vide = Cash)", template="plotly_dark", yaxis=dict(tickvals=[0, 1], ticktext=["CASH", "INVESTI"]), height=200)
        st.plotly_chart(fig_exp, use_container_width=True)

        # Graphique Drawdown
        peak = res_s.cummax()
        drawdown = (res_s - peak) / peak * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill='tozeroy', name="Drawdown %", line=dict(color='#d62728')))
        fig_dd.update_layout(title="Risque : Drawdown Historique de la Stratégie (%)", template="plotly_dark")
        st.plotly_chart(fig_dd, use_container_width=True)
    else:
        st.error("Données insuffisantes.")
