import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Momentum 500 SP500 - Full Dynamic", layout="wide")

# --- BARRE LATÉRALE : PARAMÈTRES STRATÉGIE ---
st.sidebar.header("⚙️ Paramètres Stratégie")

# Ces curseurs pilotent maintenant directement la logique de sélection
num_assets = st.sidebar.slider("Nombre d'actifs à détenir", 1, 10, 2)
lookback_months = st.sidebar.slider("Look-back Momentum (mois)", 1, 12, 6)
rotation_freq = st.sidebar.slider("Fréquence rotation secteurs (mois)", 1, 12, 3)
fees_pct = st.sidebar.slider("Frais par transaction (%)", 0.0, 0.5, 0.10, step=0.01) / 100

st.sidebar.markdown("---")

# --- BARRE LATÉRALE : MARKET TIMING ---
st.sidebar.header("🛡️ Market Timing (Mensuel)")
enable_filter = st.sidebar.checkbox("Activer le filtre de tendance", value=True)
ma_window = st.sidebar.slider("Moyenne Mobile S&P 500 (jours)", 50, 250, 150)

st.sidebar.markdown("---")

# --- BARRE LATÉRALE : PÉRIODE ---
st.sidebar.header("📅 Période d'Analyse")
start_input = st.sidebar.text_input("Date de début", "1999/01/01")
end_input = st.sidebar.text_input("Date de fin", "2026/12/31")

# --- TITRE ---
st.title("🚀 Momentum Pro : Analyse Long Terme")
st.header(f"{start_input.split('/')[0]}-2026")

# --- LOGIQUE DE CALCUL COMPLEXE ---
@st.cache_data
def run_full_momentum_engine(start, end, lb, hold, f_pct, filter_on, ma_val, n_top):
    # 1. Chargement des données (Utilisation des secteurs S&P 500 comme univers pour la démo)
    tickers = ["XLF", "XLK", "XLE", "XLI", "XLV", "XLP", "XLU", "XLY", "XLB", "XLRE", "XLC"]
    s_date = start.replace('/', '-')
    e_date = end.replace('/', '-')
    
    # Téléchargement de l'univers + Indice de référence
    raw_data = yf.download(tickers + ["^GSPC"], start=s_date, end=e_date, interval="1mo")['Close']
    
    if raw_data.empty:
        return None

    # Séparation Indice et Univers
    mkt_price = raw_data["^GSPC"].rename("mkt_price")
    assets_prices = raw_data[tickers]
    
    # 2. Calcul du Momentum (Look-back)
    # Calcul des rendements sur la période de lookback choisie
    momentum_scores = assets_prices.pct_change(lb)
    
    # 3. Simulation mois par mois
    monthly_returns = assets_prices.pct_change()
    strat_returns = pd.Series(0, index=monthly_returns.index)
    
    # Filtre de Tendance (Market Timing)
    mkt_ma = mkt_price.rolling(window=ma_val // 20).mean() # approx mois
    trend_signal = (mkt_price > mkt_ma).shift(1) if filter_on else pd.Series(True, index=mkt_price.index)
    
    current_portfolio = []
    
    # Boucle de rebalancement
    for i in range(len(monthly_returns)):
        if i % hold == 0: # Fréquence de rotation
            if trend_signal.iloc[i]:
                # Sélection des N meilleurs actifs selon le momentum
                top_assets = momentum_scores.iloc[i].nlargest(n_top).index.tolist()
                
                # Si le portefeuille change, on applique les frais
                if top_assets != current_portfolio:
                    strat_returns.iloc[i] -= f_pct
                
                current_portfolio = top_assets
            else:
                current_portfolio = [] # Cash si tendance baissière
        
        # Calcul du rendement du mois
        if current_portfolio:
            strat_returns.iloc[i] += monthly_returns[current_portfolio].iloc[i].mean()

    # 4. Calcul des performances cumulées
    results = pd.DataFrame(index=monthly_returns.index)
    results['mkt_ret'] = mkt_price.pct_change().fillna(0)
    results['strat_ret'] = strat_returns.fillna(0)
    results['cum_mkt'] = (1 + results['mkt_ret']).cumprod()
    results['cum_strat'] = (1 + results['strat_ret']).cumprod()
    
    return results

# Exécution du moteur
df_res = run_full_momentum_engine(start_input, end_input, lookback_months, rotation_freq, fees_pct, enable_filter, ma_window, num_assets)

# --- CALCUL DES MÉTRIQUES (KPI) ---
def get_metrics(cum_series, ret_series):
    yrs = (cum_series.index[-1] - cum_series.index[0]).days / 365.25
    total_perf = (cum_series.iloc[-1] - 1) * 100
    cagr = (cum_series.iloc[-1] ** (1/yrs) - 1) * 100
    vol = ret_series.std() * np.sqrt(12) * 100 # Mensuel -> Annuel
    peak = cum_series.cummax()
    mdd = ((cum_series - peak) / peak).min() * 100
    sharpe = ((ret_series.mean() * 12) - 0.02) / (ret_series.std() * np.sqrt(12))
    return total_perf, cagr, sharpe, mdd, vol

if df_res is not None:
    p_s, c_s, sh_s, m_s, v_s = get_metrics(df_res['cum_strat'], df_res['strat_ret'])
    p_m, c_m, sh_m, m_m, v_m = get_metrics(df_res['cum_mkt'], df_res['mkt_ret'])

    # --- AFFICHAGE DASHBOARD ---
    st.subheader("📊 Comparaison des Performances")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 🔹 Ma Stratégie")
        l1, l2, l3 = st.columns(3)
        l1.metric("Perf. Totale", f"{p_s:.1f}%")
        l2.metric("CAGR Net", f"{c_s:.2f}%")
        l3.metric("Sharpe", f"{sh_s:.2f}")
        l4, l5, l6 = st.columns(3)
        l4.metric("Max DD", f"{m_s:.1f}%")
        l5.metric("Volatilité", f"{v_s:.1f}%")
        l6.metric("Trades", "Calculé")

    with col_r:
        st.markdown("### 🔸 S&P 500")
        r1, r2, r3 = st.columns(3)
        r1.metric("Perf. Totale", f"{p_m:.1f}%")
        r2.metric("CAGR", f"{c_m:.2f}%")
        r3.metric("Sharpe", f"{sh_m:.2f}")
        r4, r5, _ = st.columns(3)
        r4.metric("Max DD", f"{m_m:.1f}%")
        r5.metric("Volatilité", f"{v_m:.1f}%")

    st.write("---")
    
    # --- GRAPHIQUE ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_res.index, y=df_res['cum_strat'], name="Ma Stratégie", line=dict(color='#3366CC', width=2.5)))
    fig.add_trace(go.Scatter(x=df_res.index, y=df_res['cum_mkt'], name="S&P 500", line=dict(color='#FF6633', width=1.5, dash='dot')))
    fig.update_layout(template="plotly_white", height=500, yaxis_type="log", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
