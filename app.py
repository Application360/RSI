import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="RSI 10 Strategy Tracker", layout="wide")

st.title("📊 Comparaison : Stratégie RSI 10 vs Indice")
st.markdown("""
Cette application compare la stratégie **RSI 10** (Achat si RSI ≥ 50 ou RSI < 32) avec l'investissement passif (Buy & Hold).
""")

# --- BARRE LATÉRALE (PARAMÈTRES) ---
st.sidebar.header("⚙️ Paramètres")
ticker = st.sidebar.text_input("Symbole Yahoo Finance", "^GSPC")

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
def get_data_and_calc(ticker, start, end, fees, th_buy, th_panic):
    df = yf.download(ticker, start=start, end=end, interval="1wk")
    if df.empty: return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[['Close']].copy()
    df.columns = ['price']
    
    # Calcul RSI 10
    delta = df['price'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=10).mean()
    avg_loss = loss.rolling(window=10).mean()
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
    data = get_data_and_calc(ticker, start_date, end_date, fees, threshold_buy, threshold_panic)

    if data is not None:
        # 1. ÉTAT ACTUEL
        last_rsi = data['rsi'].iloc[-1]
        last_signal = data['signal'].iloc[-1]
        last_price = data['price'].iloc[-1]
        
        st.subheader(f"🚨 État au {data.index[-1].strftime('%d/%m/%Y')}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix Actuel", f"{last_price:,.2f}")
        c2.metric("RSI 10", f"{last_rsi:.2f}")
        with c3:
            if last_signal == 1: st.success("POSITION : ACHAT")
            else: st.warning("POSITION : CASH")

        # 2. GRAPHIQUE DES PERFORMANCES
        st.subheader("📈 Comparaison des Performances")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['cum_mkt'], name=f"Indice ({ticker})", line=dict(color='gray', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=data.index, y=data['cum_strat'], name="Ma Stratégie RSI", line=dict(color='green', width=2.5)))
        fig.update_layout(yaxis_type="log", template="plotly_white", height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # 3. CALCULS DES MÉTRIQUES FINALES
        ret_strat = (data['cum_strat'].iloc[-1] - 1) * 100
        ret_mkt = (data['cum_mkt'].iloc[-1] - 1) * 100
        mdd_strat = calc_max_drawdown(data['cum_strat'])
        mdd_mkt = calc_max_drawdown(data['cum_mkt'])
        
        rf = 0.02
        sharpe_strat = ((data['net_ret'].mean() * 52) - rf) / (data['net_ret'].std() * np.sqrt(52)) if data['net_ret'].std() != 0 else 0
        sharpe_mkt = ((data['mkt_ret'].mean() * 52) - rf) / (data['mkt_ret'].std() * np.sqrt(52)) if data['mkt_ret'].std() != 0 else 0

        # 4. AFFICHAGE DES RÉSULTATS COMPARATIFS
        st.subheader("📊 Résultats détaillés : Stratégie vs Indice")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.write("**Performance Totale**")
            st.metric("Stratégie", f"{ret_strat:,.2f} %", delta=f"{ret_strat - ret_mkt:.2f} % vs Indice")
            st.metric("Indice", f"{ret_mkt:,.2f} %")
            
        with col_b:
            st.write("**Risque (Max Drawdown)**")
            st.metric("DD Stratégie", f"{mdd_strat:.2f} %", delta=f"{mdd_strat - mdd_mkt:.2f} % pts")
            st.metric("DD Indice", f"{mdd_mkt:.2f} %")

        with col_c:
            st.write("**Efficacité (Sharpe)**")
            st.metric("Sharpe Stratégie", f"{sharpe_strat:.2f}", delta=f"{sharpe_strat - sharpe_mkt:.2f}")
            st.metric("Sharpe Indice", f"{sharpe_mkt:.2f}")

        st.write("---")
        st.write(f"**Nombre de Trades :** {int(data['trade'].sum())}")
    else:
        st.error("Aucune donnée trouvée.")
