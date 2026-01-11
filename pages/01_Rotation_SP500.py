import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date

# 1. Configuration de la page
st.set_page_config(page_title="Momentum Pure Top 3", layout="wide")

def calculate_metrics(returns):
    """Calcule les métriques de performance de manière sécurisée"""
    if returns.empty:
        return 0, 0, 0, 0, 0
    total_return = (returns + 1).prod() - 1
    # Calcul du nombre d'années réelles entre le début et la fin
    days = (returns.index[-1] - returns.index[0]).days
    n_years = max(days / 365.25, 0.1)
    
    cagr = (total_return + 1) ** (1 / n_years) - 1
    vol = returns.std() * np.sqrt(12)
    sharpe = (cagr - 0.02) / vol if vol > 0 else 0
    cum_rets = (returns + 1).cumprod()
    drawdown = (cum_rets / cum_rets.cummax() - 1).min()
    return cagr, vol, sharpe, drawdown, total_return

def run_momentum_pure():
    st.title("🚀 Momentum Pure : Top 3 Secteurs")
    
    sectors = ['XLK', 'XLF', 'XLV', 'XLY', 'XLI', 'XLP', 'XLE', 'XLC', 'XLB', 'XLU', 'XLRE']
    
    # --- BARRE LATÉRALE POUR LES PARAMÈTRES ---
    with st.sidebar:
        st.header("⚙️ Paramètres")
        # Sélection des dates
        start_date = st.date_input("Date de début", date(1999, 1, 1))
        end_date = st.date_input("Date de fin", date.today())
        
        st.divider()
        st.write("Stratégie : Top 3 secteurs sur 3 mois.")
        
    if start_date >= end_date:
        st.error("Erreur : La date de début doit être antérieure à la date de fin.")
        return

    @st.cache_data
    def load_data(s_date, e_date):
        # On télécharge avec une marge de 4 mois avant la date de début pour le momentum
        margin_start = pd.to_datetime(s_date) - pd.DateOffset(months=4)
        data = yf.download(sectors + ['SPY'], start=margin_start, end=e_date, progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data['Close'].ffill()
            open_prices = data['Open'].ffill()
        else:
            close_prices = data['Close'].ffill()
            open_prices = data['Open'].ffill()
        return close_prices, open_prices

    try:
        with st.spinner('Analyse des données en cours...'):
            close_data, open_data = load_data(start_date, end_date)
            
            # Resample mensuel : fin de mois pour les signaux
            monthly_close = close_data.resample('ME').last()
            # Calcul du Momentum (Look-back 3 mois)
            momentum = monthly_close[sectors].pct_change(3)
            
            history = []
            
            # Trouver l'index de départ réel basé sur la date choisie par l'utilisateur
            start_dt = pd.to_datetime(start_date)
            # On commence là où le momentum est disponible ET après la date de début demandée
            valid_start_idx = 0
            for j in range(len(monthly_close)):
                if monthly_close.index[j] >= start_dt and j >= 3:
                    valid_start_idx = j
                    break

            # Boucle de backtest
            for i in range(valid_start_idx, len(monthly_close) - 1):
                scores = momentum.iloc[i].dropna().sort_values(ascending=False)
                if len(scores) < 3: continue
                
                top_3 = scores.index[:3].tolist()
                
                date_debut_mois = monthly_close.index[i] + pd.Timedelta(days=1)
                date_fin_mois = monthly_close.index
