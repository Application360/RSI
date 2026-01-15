import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Momentum 500 : Full Stocks SP500", layout="wide")

# --- BARRE LATÉRALE ---
st.sidebar.header("⚙️ Paramètres Stratégie")
num_assets = st.sidebar.slider("Nombre d'actions à détenir", 5, 50, 20)
lookback_months = st.sidebar.slider("Look-back Momentum (mois)", 1, 12, 6)
rotation_freq = st.sidebar.slider("Fréquence rotation (mois)", 1, 12, 3)
fees_pct = st.sidebar.slider("Frais par transaction (%)", 0.0, 0.5, 0.10, step=0.01) / 100

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Market Timing")
enable_filter = st.sidebar.checkbox("Activer le filtre de tendance", value=True)
ma_window = st.sidebar.slider("Moyenne Mobile S&P 500 (jours)", 50, 250, 150)

st.sidebar.markdown("---")
st.sidebar.header("📅 Période")
start_input = st.sidebar.text_input("Date de début", "2010/01/01") # Période réduite par défaut pour la stabilité
end_input = st.sidebar.text_input("Date de fin", "2026/12/31")

# --- TITRE ---
st.title("🚀 Momentum 500 : Sélection d'Actions Individuelles")
st.info("Le calcul sur 500 actions peut prendre quelques secondes selon votre connexion.")

# --- RÉCUPÉRATION DE LA LISTE DES 500 ACTIONS (WIKIPEDIA) ---
@st.cache_data
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    df = table[0]
    tickers = df['Symbol'].str.replace('.', '-', regex=True).tolist()
    return tickers

# --- MOTEUR DE CALCUL ---
@st.cache_data
def run_stock_momentum_engine(start, end, lb, hold, f_pct, filter_on, ma_val, n_top):
    tickers = get_sp500_tickers()
    s_date = start.replace('/', '-')
    e_date = end.replace('/', '-')
    
    # Téléchargement par blocs pour éviter les erreurs
    all_data = yf.download(tickers, start=s_date, end=e_date, interval="1mo")['Close']
    mkt_price = yf.download("^GSPC", start=s_date, end=e_date, interval="1mo")['Close']
    
    if all_data.empty: return None, 0

    # Calcul Momentum (Rendement sur LB mois)
    momentum_scores = all_data.pct_change(lb)
    monthly_returns = all_data.pct_change()
    
    # Market Timing
    mkt_ma = mkt_price.rolling(window=max(1, ma_val // 20)).mean()
    trend_signal = (mkt_price > mkt_ma).shift(1) if filter_on else pd.Series(True, index=mkt_price.index)
    
    strat_returns = pd.Series(0, index=monthly_returns.index)
    current_portfolio = []
    trade_count = 0
    
    for i in range(len(monthly_returns)):
        if i % hold == 0:
            new_portfolio = []
            if trend_signal.iloc[i]:
                # On élimine les valeurs manquantes (actions pas encore cotées)
                valid_scores = momentum_scores.iloc[i].dropna()
                if not valid_scores.empty:
                    new_portfolio = valid_scores.nlargest(n_top).index.tolist()
            
            if set(new_portfolio) != set(current_portfolio):
                if i > 0: trade_count += 1
                strat_returns.iloc[i] -= f_pct
                current_portfolio = new_portfolio
        
        if current_portfolio:
            # Rendement moyen des N actions sélectionnées
            strat_returns.iloc[i] += monthly_returns[current_portfolio].iloc[i].mean()

    results = pd.DataFrame(index=monthly_returns.index)
    results['mkt_ret'] = mkt_price.pct_change().fillna(0)
    results['strat_ret'] = strat_returns.fillna(0)
    results['cum_mkt'] = (1 + results['mkt_ret']).cumprod()
    results['cum_strat'] = (1 + results['strat_ret']).cumprod()
    
    return results, trade_count

# Exécution
with st.spinner('Analyse des 500 actions en cours...'):
    df_res, total_trades = run_stock_momentum_engine(start_input, end_input, lookback_months, rotation_freq, fees_pct, enable_filter, ma_window, num_assets)

# --- KPI & AFFICHAGE ---
def get_metrics(cum_series, ret_series):
    yrs = (cum_series.index[-1] - cum_series.index[0]).days / 365.25
    total_perf = (cum_series.iloc[-1] - 1) * 100
    cagr = (cum_series.iloc[-1] ** (1/yrs) - 1) * 100
    vol = ret_series.std() * np.sqrt(12) * 100
    peak = cum_series.cummax()
    mdd = ((cum_series - peak) / peak).min() * 100
    sharpe = (ret_series.mean() * 12) / (ret_series.std() * np.sqrt(12)) if ret_series.std() != 0 else 0
    return total_perf, cagr, sharpe, mdd, vol

if df_res is not None:
    p_s, c_s, sh_s, m_s, v_s = get_metrics(df_res['cum_strat'], df_res['strat_ret'])
    p_m, c_m, sh_m, m_m, v_m = get_metrics(df_res['cum_mkt'], df_res['mkt_ret'])

    st.subheader("📊 Performance : Sélection de 500 Actions Individuelles")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(f"### 🔹 Momentum {num_assets} Actions")
        k1, k2, k3 = st.columns(3)
        k1.metric("Perf. Totale", f"{p_s:.1f}%")
        k2.metric("CAGR Net", f"{c_s:.2f}%")
        k3.metric("Sharpe (RF 0%)", f"{sh_s:.2f}")
        k4, k5, k6 = st.columns(3)
        k4.metric("Max DD", f"{m_s:.1f}%")
        k5.metric("Volatilité", f"{v_s:.1f}%")
        k6.metric("Trades", f"{total_trades}")

    with col_r:
        st.markdown("### 🔸 S&P 500 (^GSPC)")
        s1, s2, s3 = st.columns(3)
        s1.metric("Perf. Totale", f"{p_m:.1f}%")
        s2.metric("CAGR", f"{c_m:.2f}%")
        s3.metric("Sharpe (RF 0%)", f"{sh_m:.2f}")
        s4, s5, _ = st.columns(3)
        s4.metric("Max DD", f"{m_m:.1f}%")
        s5.metric("Volatilité", f"{v_m:.1f}%")

    st.write("---")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_res.index, y=df_res['cum_strat'], name="Stratégie Actions", line=dict(color='#3366CC', width=2.5)))
    fig.add_trace(go.Scatter(x=df_res.index, y=df_res['cum_mkt'], name="S&P 500", line=dict(color='#FF6633', width=1.5, dash='dot')))
    fig.update_layout(template="plotly_white", height=550, yaxis_type="log", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Échec du téléchargement. Yahoo Finance limite peut-être les requêtes massives.")
