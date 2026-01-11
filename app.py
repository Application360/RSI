import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="RSI Strategy Ultimate", layout="wide")

# --- BARRE LATÉRALE (PARAMÈTRES) ---
st.sidebar.header("⚙️ Paramètres")
ticker = st.sidebar.text_input("Symbole Yahoo Finance", "^GSPC")

# CHOIX DE LA PÉRIODE RSI
rsi_period = st.sidebar.slider("Période du RSI (Fenêtre)", min_value=2, max_value=30, value=10)

# --- AJOUT : SÉLECTEUR TAUX SANS RISQUE ---
risk_free_rate = st.sidebar.slider("Taux sans risque (pour Sharpe)", min_value=0.0, max_value=0.10, value=0.02, step=0.005)

st.title(f"📊 Analyse Comparative : Stratégie RSI {rsi_period} vs Indice")
st.markdown(f"""
Tableau de bord complet : Performance Totale, CAGR, Volatilité, Max Drawdown et Ratio de Sharpe (RF: {risk_free_rate*100:.2f}%).
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
        years = (data.index[-1] - data.index[0]).days / 365.25

        # 1. GRAPHIQUE
        st.subheader("📈 Évolution Comparative (Échelle Log)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['cum_mkt'], name=f"Indice ({ticker})", line=dict(color='gray', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=data.index, y=data['cum_strat'], name="Ma Stratégie", line=dict(color='green', width=2.5)))
        fig.update_layout(yaxis_type="log", template="plotly_white", height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # 2. CALCULS DES MÉTRIQUES
        total_strat = (data['cum_strat'].iloc[-1] - 1) * 100
        total_mkt = (data['cum_mkt'].iloc[-1] - 1) * 100
        
        cagr_strat = (data['cum_strat'].iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0
        cagr_mkt = (data['cum_mkt'].iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0
        
        vol_strat = data['net_ret'].std() * np.sqrt(52) * 100
        vol_mkt = data['mkt_ret'].std() * np.sqrt(52) * 100
        
        mdd_strat = calc_max_drawdown(data['cum_strat'])
        mdd_mkt = calc_max_drawdown(data['cum_mkt'])

        # --- MISE À JOUR : CALCUL SHARPE AVEC VARIABLE risk_free_rate ---
        sharpe_strat = ((data['net_ret'].mean() * 52) - risk_free_rate) / (data['net_ret'].std() * np.sqrt(52)) if data['net_ret'].std() != 0 else 0
        sharpe_mkt = ((data['mkt_ret'].mean() * 52) - risk_free_rate) / (data['mkt_ret'].std() * np.sqrt(52)) if data['mkt_ret'].std() != 0 else 0

        # 3. AFFICHAGE DES MÉTRIQUES (LIGNE 1 : PERF / LIGNE 2 : RISQUE)
        st.subheader("📊 Métriques Stratégie vs Indice")
        
        col1, col2, col3, col4, col5 = st.columns(5) # Ajout d'une 5ème colonne
        with col1:
            st.write("**Performance Totale**")
            st.metric("Stratégie", f"{total_strat:,.2f} %", delta=f"{total_strat - total_mkt:,.2f} %")
            st.metric("Indice", f"{total_mkt:,.2f} %")
        with col2:
            st.write("**CAGR (Annuel)**")
            st.metric("Stratégie", f"{cagr_strat:.2f} %", delta=f"{cagr_strat - cagr_mkt:.2f} %")
            st.metric("Indice", f"{cagr_mkt:.2f} %")
        with col3:
            st.write("**Volatilité Annuelle**")
            st.metric("Stratégie", f"{vol_strat:.2f} %", delta=f"{vol_strat - vol_mkt:.2f} %", delta_color="inverse")
            st.metric("Indice", f"{vol_mkt:.2f} %")
        with col4:
            st.write("**Max Drawdown**")
            st.metric("Stratégie", f"{mdd_strat:.2f} %", delta=f"{mdd_strat - mdd_mkt:.2f} % pts")
            st.metric("Indice", f"{mdd_mkt:.2f} %")
        with col5:
            st.write("**Ratio de Sharpe**")
            st.metric("Stratégie", f"{sharpe_strat:.2f}", delta=f"{sharpe_strat - sharpe_mkt:.2f}")
            st.metric("Indice", f"{sharpe_mkt:.2f}")

        st.write("---")

        # 4. ANALYSE ANNUELLE & EXPORT
        st.subheader("📅 Analyse par Année Civile")
        
        annual_strat = data['net_ret'].groupby(data.index.year).apply(lambda x: (np.prod(1 + x.fillna(0)) - 1) * 100)
        annual_mkt = data['mkt_ret'].groupby(data.index.year).apply(lambda x: (np.prod(1 + x.fillna(0)) - 1) * 100)
        
        df_annual = pd.DataFrame({
            'Stratégie (%)': annual_strat,
            'Indice (%)': annual_mkt,
            'Différence (%)': annual_strat - annual_mkt
        }).sort_index(ascending=False)

        c_table, c_export = st.columns([3, 1])
        with c_table:
            st.dataframe(df_annual.style.format("{:.2f} %").applymap(
                lambda val: f'color: {"green" if val > 0 else "red"}', subset=['Différence (%)']
            ), use_container_width=True)
        with c_export:
            st.write("📥 **Exportation**")
            csv = df_annual.to_csv(index=True).encode('utf-8')
            st.download_button("Télécharger CSV", data=csv, file_name=f"RSI_Analysis_{ticker}.csv", mime='text/csv')
            st.info(f"Trades : {int(data['trade'].sum())}")

    else:
        st.error("Données indisponibles.")
