import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Momentum Pure SaaS", layout="wide")

def calculate_metrics(returns):
    """Calcule les métriques clés de performance"""
    total_return = (returns + 1).prod() - 1
    n_years = len(returns) / 12
    cagr = (total_return + 1) ** (1 / n_years) - 1
    vol = returns.std() * np.sqrt(12)
    sharpe = (cagr - 0.02) / vol if vol > 0 else 0
    cum_rets = (returns + 1).cumprod()
    rolling_max = cum_rets.cummax()
    drawdown = (cum_rets / rolling_max - 1).min()
    return cagr, vol, sharpe, drawdown, total_return

def run_momentum_pure():
    st.title("🚀 Momentum Pure : Top 3 Secteurs")
    st.markdown("""
    **Stratégie Simplifiée :**
    - **Look-back :** 3 mois.
    - **Sélection :** Les 3 meilleurs secteurs.
    - **Rebalancement :** Mensuel complet (équipondéré : 33.3% par secteur).
    - **Exécution :** Signal en clôture de mois, achat en ouverture le lendemain.
    """)

    sectors = ['XLK', 'XLF', 'XLV', 'XLY', 'XLI', 'XLP', 'XLE', 'XLC', 'XLB', 'XLU', 'XLRE']
    
    @st.cache_data
    def load_data():
        data = yf.download(sectors + ['SPY'], start="1998-10-01")
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data['Close']
            open_prices = data['Open']
        else:
            close_prices = data['Close']
            open_prices = data['Open']
        return close_prices.ffill(), open_prices.ffill()

    try:
        close_data, open_data = load_data()
        monthly_close = close_data.resample('ME').last()
        momentum = monthly_close[sectors].pct_change(3)
        
        history = []
        
        # Début du backtest (après les 3 mois de look-back)
        for i in range(3, len(monthly_close) - 1):
            # 1. Sélection des 3 meilleurs basés sur les 3 derniers mois
            scores = momentum.iloc[i].sort_values(ascending=False)
            top_3 = scores.index[:3].tolist()
            
            # 2. Dates d'exécution
            next_month_start = monthly_close.index[i] + pd.Timedelta(days=1)
            next_month_end = monthly_close.index[i+1]
            
            # Sécurité pour les jours ouvrés
            idx_start = open_data.index.get_indexer([next_month_start], method='bfill')[0]
            idx_end = close_data.index.get_indexer([next_month_end], method='ffill')[0]
            
            # 3. Calcul du rendement (33.3% par secteur)
            month_return = 0
            for ticker in top_3:
                p_open = open_data[ticker].iloc[idx_start]
                p_close = close_data[ticker].iloc[idx_end]
                month_return += ((p_close / p_open) - 1) / 3
            
            # Benchmark SPY
            spy_open = open_data['SPY'].iloc[idx_start]
            spy_close = close_data['SPY'].iloc[idx_end]
            
            history.append({
                'Date': monthly_close.index[i+1],

    except Exception as e:
        st.error(f"Détails de l'erreur : {e}")

if __name__ == "__main__":
    run_momentum_pro()
