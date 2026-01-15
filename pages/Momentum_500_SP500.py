import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Stratégie Momentum S&P 500", layout="wide")

st.title("🚀 Stratégie Momentum : Top 500 S&P")
st.markdown("""
Cette application simule une stratégie de **Momentum** : on achète les actions ayant eu la meilleure performance sur une période passée (*Look-back*) et on les conserve pendant une période définie (*Holding*).
""")

# --- FONCTION DE CHARGEMENT ROBUSTE ---
@st.cache_data
def load_data(file_path):
    # Chargement avec gestion de l'index Date
    df = pd.read_csv(file_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)
    
    # Nettoyage : suppression des colonnes totalement vides
    df = df.dropna(axis=1, how='all')
    return df

# Tentative de localisation du fichier
filename = 'sp500_data_final.csv'
paths_to_check = [filename, os.path.join('..', filename), os.path.join('pages', filename)]
data_file = next((p for p in paths_to_check if os.path.exists(p)), None)

if data_file is not None:
    data = load_data(data_file)
else:
    st.error(f"⚠️ Fichier '{filename}' non trouvé dans le répertoire.")
    uploaded_file = st.file_uploader("Veuillez uploader le fichier CSV manuellement ici :", type=["csv"])
    if uploaded_file:
        data = load_data(uploaded_file)
    else:
        st.stop()

# --- SIDEBAR - PARAMÈTRES ---
st.sidebar.header("⚙️ Paramètres du Backtest")

# Filtre de dates
min_date = data.index.min().to_pydatetime()
max_date = data.index.max().to_pydatetime()
start_date = st.sidebar.date_input("Date de début", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("Date de fin", max_date, min_value=min_date, max_value=max_date)

st.sidebar.separator()

# Paramètres de stratégie
lookback_months = st.sidebar.slider("Période de Look-back (mois)", 1, 12, 6)
holding_months = st.sidebar.slider("Période de Holding (mois)", 1, 12, 1)
n_tickers = st.sidebar.slider("Nombre de tickers retenus", 1, 20, 10)

# --- CALCUL DU BENCHMARK ---
# Si le ticker ^GSPC n'est pas dans les données, on crée un index équi-pondéré des actions disponibles
if '^GSPC' in data.columns:
    benchmark_raw = data['^GSPC']
else:
    benchmark_raw = data.mean(axis=1) # Proxy S&P 500 (Equal Weight)

# --- MOTEUR DE BACKTEST ---
def run_backtest(df, bench, start, end, lb, hold, n):
    # Tronquer les données
    df = df.loc[start:end].copy()
    bench = bench.loc[start:end].copy()
    
    # Conversion en rendement mensuel pour simplifier le calcul momentum
    monthly_df = df.resample('ME').last()
    monthly_bench = bench.resample('ME').last()
    
    monthly_returns = monthly_df.pct_change()
    bench_returns = monthly_bench.pct_change()
    
    # Momentum : Performance cumulée sur les 'lb' derniers mois
    momentum_signal = monthly_df.pct_change(lb)
    
    portfolio_returns = []
    dates = []
    
    # Boucle de rebalancement
    for i in range(lb, len(monthly_df) - hold, hold):
        # 1. Date actuelle de décision
        current_date = monthly_df.index[i]
        
        # 2. Sélection des N meilleurs tickers (on ignore les colonnes avec trop de NaNs à cette date)
        top_performers = momentum_signal.loc[current_date].nlargest(n).index
        
        # 3. Calcul de la performance sur la période de holding
        # Rendement moyen équi-pondéré des sélectionnés
        future_returns = monthly_returns.iloc[i + 1 : i + 1 + hold][top_performers].mean(axis=1)
        
        portfolio_returns.extend(future_returns.values)
        dates.extend(future_returns.index)

    # Séries finales
    strat_results = pd.Series(portfolio_returns, index=dates).dropna()
    bench_results = bench_returns.loc[strat_results.index]
    
    return (1 + strat_results).cumprod(), (1 + bench_results).cumprod(), strat_results, bench_results

# --- CALCUL DES MÉTRIQUES ---
def get_metrics(cum_series, returns):
    if len(cum_series) == 0: return [0]*4
    total_perf = (cum_series.iloc[-1] - 1) * 100
    
    # CAGR
    years = (cum_series.index[-1] - cum_series.index[0]).days / 365.25
    cagr = ((cum_series.iloc[-1])**(1/years) - 1) * 100 if years > 0 else 0
    
    # Sharpe (RF = 0 pour simplification)
    vol = returns.std() * np.sqrt(12)
    sharpe = (cagr / 100) / vol if vol > 0 else 0
    
    # Max Drawdown
    peak = cum_series.cummax()
    dd = (cum_series - peak) / peak
    max_dd = dd.min() * 100
    
    return total_perf, cagr, sharpe, max_dd

# --- EXÉCUTION ET AFFICHAGE ---
if st.button("🚀 Lancer l'Analyse"):
    c_strat, c_bench, r_strat, r_bench = run_backtest(
        data, benchmark_raw, start_date, end_date, lookback_months, holding_months, n_tickers
    )
    
    if len(c_strat) > 0:
        m_s = get_metrics(c_strat, r_strat)
        m_b = get_metrics(c_bench, r_bench)
        
        # Métriques en colonnes
        st.subheader("📊 Performance Comparée")
        cols = st.columns(4)
        labels = ["Performance Totale", "CAGR (Annuel)", "Ratio de Sharpe", "Max Drawdown"]
        
        for i in range(4):
            with cols[i]:
                st.metric(labels[i], f"{m_s[i]:.2f}{'%' if i != 2 else ''}")
                st.caption(f"S&P 500: {m_b[i]:.2f}{'%' if i != 2 else ''}")

        # Graphique de Performance
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c_strat.index, y=c_strat, name=f"Stratégie (Top {n_tickers})", line=dict(color='#00FFCC', width=3)))
        fig.add_trace(go.Scatter(x=c_bench.index, y=c_bench, name="S&P 500 (Reference)", line=dict(color='white', dash='dash')))
        fig.update_layout(title="Croissance du Capital (Base 1.0)", template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Graphique de Drawdown
        peak = c_strat.cummax()
        dd_strat = (c_strat - peak) / peak * 100
        
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd_strat.index, y=dd_strat, fill='tozeroy', name="Drawdown", line=dict(color='red')))
        fig_dd.update_layout(title="Risque : Drawdown Historique (%)", template="plotly_dark")
        st.plotly_chart(fig_dd, use_container_width=True)
        
    else:
        st.warning("La période choisie est trop courte pour les paramètres sélectionnés.")
