import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="Momentum Optimizer S&P 500", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('sp500_data_final.csv', index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    # On identifie le benchmark (S&P 500)
    if '^GSPC' not in df.columns:
        df['^GSPC'] = df.mean(axis=1) # Proxy si manquant
    return df

data = load_data()

# --- SIDEBAR ---
st.sidebar.header("🕹️ Contrôles de Simulation")

# Filtre de dates
min_date = data.index.min().date()
max_date = data.index.max().date()
start_date = st.sidebar.date_input("Début", min_date)
end_date = st.sidebar.date_input("Fin", max_date)

# Paramètres Momentum
lookback = st.sidebar.slider("Look-back (mois)", 1, 12, 6, help="Période de mesure de la performance passée")
holding = st.sidebar.slider("Holding (mois)", 1, 6, 1, help="Fréquence de rebalancement")

# NOUVEAU : Sélection du nombre de tickers (1 à 20)
n_tickers = st.sidebar.slider("Nombre de tickers retenus (N)", 1, 20, 10, 
                             help="Plus le nombre est bas, plus le portefeuille est concentré sur les 'Top Performers'")

# --- MOTEUR DE BACKTEST ---
def run_momentum_strategy(df, start, end, lb, hold, n):
    # Filtrage temporel
    mask = (df.index.date >= start) & (df.index.date <= end)
    subset = df.loc[mask].copy()
    
    # Passage en mensuel
    monthly = subset.resample('ME').last()
    returns = monthly.pct_change()
    
    # Calcul du Momentum (Rendement cumulé)
    momentum_scores = monthly.pct_change(lb)
    
    results = []
    dates = []
    tickers_history = {} # Pour voir quels titres ont été choisis

    # On commence après la période de lookback
    for i in range(lb, len(monthly) - hold, hold):
        current_date = monthly.index[i]
        
        # Sélection des N meilleurs (hors benchmark)
        eligible_assets = momentum_scores.columns.drop('^GSPC')
        top_n = momentum_scores.loc[current_date, eligible_assets].nlargest(n).index
        
        # Calcul de la performance sur la période de détention
        next_returns = returns.loc[monthly.index[i+1 : i+1+hold], top_n].mean(axis=1)
        
        results.extend(next_returns.values)
        dates.extend(next_returns.index)
        tickers_history[current_date] = list(top_n)

    # Séries temporelles
    strat_series = pd.Series(results, index=dates)
    bench_series = returns.loc[dates, '^GSPC']
    
    cum_strat = (1 + strat_series).cumprod()
    cum_bench = (1 + bench_series).cumprod()
    
    return cum_strat, cum_bench, strat_series, bench_series, tickers_history

# --- EXECUTION ET AFFICHAGE ---
if st.button("Calculer la Performance"):
    c_strat, c_bench, r_strat, r_bench, history = run_momentum_strategy(
        data, start_date, end_date, lookback, holding, n_tickers
    )

    # Calcul Métriques
    def calc_metrics(cum, ret):
        tr = (cum.iloc[-1] - 1) * 100
        ann = ((cum.iloc[-1])**(12/len(cum)) - 1) * 100 # Approx annuelle
        vol = ret.std() * np.sqrt(12)
        sharpe = (ann/100) / vol if vol != 0 else 0
        dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
        return tr, ann, sharpe, dd

    m_s = calc_metrics(c_strat, r_strat)
    m_b = calc_metrics(c_bench, r_bench)

    # Affichage
    st.subheader(f"Résultats pour Top {n_tickers} Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Stratégie Momentum**")
        st.dataframe(pd.DataFrame({
            "Métrique": ["Total Return", "CAGR", "Sharpe", "Max Drawdown"],
            "Valeur": [f"{m_s[0]:.2f}%", f"{m_s[1]:.2f}%", f"{m_s[2]:.2f}", f"{m_s[3]:.2f}%"]
        }))
    with col2:
        st.write("**Benchmark (S&P 500)**")
        st.dataframe(pd.DataFrame({
            "Métrique": ["Total Return", "CAGR", "Sharpe", "Max Drawdown"],
            "Valeur": [f"{m_b[0]:.2f}%", f"{m_b[1]:.2f}%", f"{m_b[2]:.2f}", f"{m_b[3]:.2f}%"]
        }))

    # Graphes
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c_strat.index, y=c_strat, name=f"Momentum (Top {n_tickers})", line=dict(color='#00ffcc')))
    fig.add_trace(go.Scatter(x=c_bench.index, y=c_bench, name="S&P 500", line=dict(color='#ff4b4b')))
    fig.update_layout(title="Comparaison Performance Cumulative", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Affichage des derniers tickers sélectionnés
    with st.expander("Voir la dernière sélection de titres"):
        last_date = list(history.keys())[-1]
        st.write(f"Titres sélectionnés au {last_date.date()} :")
        st.write(", ".join(history[last_date]))
