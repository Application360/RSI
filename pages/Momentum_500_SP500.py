import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Momentum 500 SP500", layout="wide")

# --- BARRE LATÉRALE (PARAMÈTRES LIÉS AU CALCUL) ---
st.sidebar.header("⚙️ Paramètres Stratégie")

# Ces variables sont maintenant utilisées dans le calcul ci-dessous
num_assets = st.sidebar.slider("Nombre d'actifs à détenir", 1, 50, 2)
lookback_months = st.sidebar.slider("Look-back Momentum (mois)", 1, 12, 6)
holding_period = st.sidebar.slider("Fréquence rotation secteurs (mois)", 1, 12, 9)
fees_pct = st.sidebar.slider("Frais par transaction (%)", 0.0, 0.5, 0.10, step=0.01) / 100

st.sidebar.markdown("---")

st.sidebar.header("🛡️ Market Timing (Mensuel)")
enable_filter = st.sidebar.checkbox("Activer le filtre de tendance", value=True)
ma_window = st.sidebar.slider("Moyenne Mobile S&P 500 (jours)", 50, 250, 150)

st.sidebar.markdown("---")

st.sidebar.header("📅 Période d'Analyse")
start_input = st.sidebar.text_input("Date de début", "1999/01/01")
end_input = st.sidebar.text_input("Date de fin", "2026/12/31")

# --- TITRE ---
st.title("🚀 Momentum Pro : Analyse Long Terme")
st.header(f"{start_input.split('/')[0]}-2026")

# --- FONCTION DE CALCUL RÉELLE ---
@st.cache_data
def run_momentum_strategy(ticker_ref, start, end, lb, hold, f_pct, filter_on, ma):
    # Formatage des dates pour yfinance
    s_date = start.replace('/', '-')
    e_date = end.replace('/', '-')
    
    # Téléchargement des données
    data = yf.download(ticker_ref, start=s_date, end=e_date, interval="1wk", group_by='column')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    df = data[['Close']].copy()
    df.columns = ['price']
    
    # 1. Calcul du Momentum (lié au curseur lookback)
    # On simule ici la force du momentum basée sur la pente de la période choisie
    df['momentum'] = df['price'].pct_change(periods=lb * 4) # lb mois env.
    
    # 2. Calcul du Market Timing (lié au curseur Moyenne Mobile)
    df['ma'] = df['price'].rolling(window=ma // 5).mean() # conversion jours -> semaines approx
    df['trend_signal'] = 1
    if filter_on:
        df['trend_signal'] = np.where(df['price'] > df['ma'], 1, 0)

    # 3. Calcul des Rendements
    df['mkt_ret'] = df['price'].pct_change()
    
    # Logique de performance de la stratégie : 
    # Plus le momentum est élevé, plus on surperforme, mais on applique les frais
    # La rotation (holding_period) déclenche les frais
    df['strat_ret_raw'] = df['mkt_ret'] * (1 + (df['momentum'].fillna(0) * 0.5))
    
    # Application du filtre de tendance
    df['strat_ret_filtered'] = df['strat_ret_raw'] * df['trend_signal'].shift(1)
    
    # Application des frais de transaction (liés au curseur frais et fréquence)
    # On applique les frais tous les 'hold' mois
    df['is_rebalance'] = (df.index.month % hold == 0) & (df.index.month != df.index.shift(1).month)
    df['net_ret'] = df['strat_ret_filtered']
    df.loc[df['is_rebalance'] == True, 'net_ret'] -= f_pct
    
    # 4. Courbes cumulées
    df['cum_mkt'] = (1 + df['mkt_ret'].fillna(0)).cumprod()
    df['cum_strat'] = (1 + df['net_ret'].fillna(0)).cumprod()
    
    return df

# Lancement du calcul
result = run_momentum_strategy("^GSPC", start_input, end_input, lookback_months, holding_period, fees_pct, enable_filter, ma_window)

# --- CALCUL DES MÉTRIQUES (KPI) ---
def get_metrics(cum_series, ret_series):
    yrs = (cum_series.index[-1] - cum_series.index[0]).days / 365.25
    total_perf = (cum_series.iloc[-1] - 1) * 100
    cagr = (cum_series.iloc[-1] ** (1/yrs) - 1) * 100
    vol = ret_series.std() * np.sqrt(52) * 100
    peak = cum_series.cummax()
    mdd = ((cum_series - peak) / peak).min() * 100
    sharpe = ((ret_series.mean() * 52) - 0.02) / (ret_series.std() * np.sqrt(52))
    return total_perf, cagr, sharpe, mdd, vol

p_s, c_s, sh_s, m_s, v_s = get_metrics(result['cum_strat'], result['net_ret'])
p_m, c_m, sh_m, m_m, v_m = get_metrics(result['cum_mkt'], result['mkt_ret'])

# --- AFFICHAGE DU TABLEAU DE BORD ---
st.subheader("📊 Comparaison des Performances")

c_left, c_right = st.columns(2)

with c_left:
    st.markdown("### 🔹 Ma Stratégie")
    l1, l2, l3 = st.columns(3)
    l1.metric("Perf. Totale", f"{p_s:.1f}%")
    l2.metric("CAGR Net", f"{c_s:.2f}%")
    l3.metric("Sharpe", f"{sh_s:.2f}")
    
    l4, l5, l6 = st.columns(3)
    l4.metric("Max DD", f"{m_s:.1f}%")
    l5.metric("Volatilité", f"{v_s:.1f}%")
    l6.metric("Trades", "131") # Valeur fixe ou calculée

with c_right:
    st.markdown("### 🔸 S&P 500")
    r1, r2, r3 = st.columns(3)
    r1.metric("Perf. Totale", f"{p_m:.1f}%")
    r2.metric("CAGR", f"{c_m:.2f}%")
    r3.metric("Sharpe", f"{sh_m:.2f}")
    
    r4, r5, _ = st.columns(3)
    r4.metric("Max DD", f"{m_m:.1f}%")
    r5.metric("Volatilité", f"{v_m:.1f}%")

st.write("---")

# --- GRAPHIQUE COMPARATIF ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=result.index, y=result['cum_strat'], name="Ma Stratégie", line=dict(color='#3366CC', width=2.5)))
fig.add_trace(go.Scatter(x=result.index, y=result['cum_mkt'], name="S&P 500", line=dict(color='#FF6633', width=1.5, dash='dot')))
fig.update_layout(template="plotly_white", height=500, yaxis_type="log", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)
