import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="RSI CAGR & Vol Tracker", layout="wide")

# --- BARRE LATÉRALE (PARAMÈTRES) ---
st.sidebar.header("⚙️ Paramètres")
ticker = st.sidebar.text_input("Symbole Yahoo Finance", "^GSPC")

# CHOIX DE LA PÉRIODE RSI
rsi_period = st.sidebar.slider("Période du RSI (Fenêtre)", min_value=2, max_value=30, value=10)

st.title(f"📊 Stratégie RSI {rsi_period} : Performance & Risque")
st.markdown(f"""
Analyse approfondie incluant la **Volatilité** et le **Rendement Annuel Composé (CAGR)**.
""")

# --- SÉLECTEURS DE DATES ---
col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_date = st.date_input("Date de début", value=date(1960, 1, 1), min_value=date(1960, 1, 1), max_value=date(2025, 12, 31))
with col_d2:
    end_date = st.date_input("Date de fin", value=date(2025, 12, 31), min_value=date(1960, 1, 1), max_value=date(2025, 12, 31))

fees = st.sidebar.slider("Frais de transaction (%)", 0.0, 0.5, 0.1, step=0.01) / 100

st.sidebar.subheader("Seuils RSI")
threshold_buy = st.sidebar.number_input("Seuil Achat (Tendance)", value=50)
threshold_panic = st.sidebar.number_input("Seuil Achat (Panique)", value=32)

# --- FONCTIONS DE CALCUL ---
@st.cache_data
def get_data_and_calc(ticker, start, end, fees, th_buy, th_panic, period):
    df = yf.download(ticker, start=start, end=end, interval="1wk")
    if df.empty: return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[['Close']].copy()
    df.columns = ['price']
    
    # Calcul RSI
    delta = df['price'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Signaux et Rendements
    df['signal'] = 0
    df.loc[(df['rsi'] >= th_buy) | (df['rsi'] < th_panic), 'signal'] = 1
    
    df['mkt_ret'] = df['price'].pct_change()
    df['strat_ret_raw'] = df['signal'].shift(1) * df['mkt_ret']
    df['trade'] = df['signal'].diff().fillna(0).abs()
    df['net_ret'] = df['strat_ret_raw'] - (df['trade'] * fees)
    
    df['cum_mkt'] = (1 + df['mkt_ret'].fillna(0)).cumprod()
    df['cum_strat'] = (1 + df['net_ret'].fillna(0)).cumprod()
    
    return df

def calc_max_drawdown(cum_series):
    peak = cum_series.cummax()
    drawdown = (cum_series - peak) / peak
    return drawdown.min() * 100

# --- EXÉCUTION ---
if start_date >= end_date:
    st.error("Erreur : La date de début doit être antérieure à la date de fin.")
else:
    data = get_data_and_calc(ticker, start_date, end_date, fees, threshold_buy, threshold_panic, rsi_period)

    if data is not None:
        # Calcul du nombre d'années pour le CAGR
        days = (data.index[-1] - data.index[0]).days
        years = days / 365.25

        # 1. ÉTAT ACTUEL
        last_rsi = data['rsi'].iloc[-1]
        last_signal = data['signal'].iloc[-1]
        
        st.subheader(f"🚨 État au {data.index[-1].strftime('%d/%m/%Y')}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix Actuel", f"{data['price'].iloc[-1]:,.2f}")
        c2.metric(f"RSI {rsi_period}", f"{last_rsi:.2f}")
        with c3:
            if last_signal == 1: st.success("POSITION : ACHAT")
            else: st.warning("POSITION : CASH")

        # 2. GRAPHIQUE
        st.subheader("📈 Comparaison des Performances (Échelle Log)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['cum_mkt'], name=f"Indice ({ticker})", line=dict(color='gray', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=data.index, y=data['cum_strat'], name="Ma Stratégie", line=dict(color='green', width=2.5)))
        fig.update_layout(yaxis_type="log", template="plotly_white", height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # 3. CALCULS DES MÉTRIQUES
        # CAGR
        cagr_strat = (data['cum_strat'].iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0
        cagr_mkt = (data['cum_mkt'].iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Volatilité
        vol_strat = data['net_ret'].std() * np.sqrt(52) * 100
        vol_mkt = data['mkt_ret'].std() * np.sqrt(52) * 100
        
        # Drawdown
        mdd_strat = calc_max_drawdown(data['cum_strat'])
        mdd_mkt = calc_max_drawdown(data['cum_mkt'])
        
        # Sharpe
        rf = 0.02
        sharpe_strat = ((data['net_ret'].mean() * 52) - rf) / (data['net_ret'].std() * np.sqrt(52)) if data['net_ret'].std() != 0 else 0
        sharpe_mkt = ((data['mkt_ret'].mean() * 52) - rf) / (data['mkt_ret'].std() * np.sqrt(52)) if data['mkt_ret'].std() != 0 else 0

        # 4. AFFICHAGE DES RÉSULTATS
        st.subheader("📊 Tableau de Bord des Performances Annuelles & Risques")
        
        # Ligne Performance (CAGR & Totale)
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.write("**Performance Annuelle (CAGR)**")
            st.metric("CAGR Stratégie", f"{cagr_strat:.2f} % / an", delta=f"{cagr_strat - cagr_mkt:.2f} % vs Indice")
            st.metric("CAGR Indice", f"{cagr_mkt:.2f} % / an")
        with row1_col2:
            st.write("**Performance Totale**")
            st.metric("Total Stratégie", f"{(data['cum_strat'].iloc[-1]-1)*100:,.2f} %")
            st.metric("Total Indice", f"{(data['cum_mkt'].iloc[-1]-1)*100:,.2f} %")

        st.write("---")
        
        # Ligne Risque (Vol & Drawdown)
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        with row2_col1:
            st.write("**Volatilité Annualisée**")
            st.metric("Vol Stratégie", f"{vol_strat:.2f} %", delta=f"{vol_strat - vol_mkt:.2f} %", delta_color="inverse")
            st.metric("Vol Indice", f"{vol_mkt:.2f} %")
        with row2_col2:
            st.write("**Risque (Max Drawdown)**")
            st.metric("MDD Stratégie", f"{mdd_strat:.2f} %", delta=f"{mdd_strat - mdd_mkt:.2f} % pts")
            st.metric("MDD Indice", f"{mdd_mkt:.2f} %")
        with row2_col3:
            st.write("**Efficacité (Sharpe)**")
            st.metric("Sharpe Stratégie", f"{sharpe_strat:.2f}", delta=f"{sharpe_strat - sharpe_mkt:.2f}")
            st.metric("Sharpe Indice", f"{sharpe_mkt:.2f}")

        st.info(f"Période analysée : {years:.1f} ans. Nombre de trades : {int(data['trade'].sum())}")
    else:
        st.error("Aucune donnée trouvée.")
