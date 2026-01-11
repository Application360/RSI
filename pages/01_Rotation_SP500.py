import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# 1. Configuration de la page
st.set_page_config(page_title="Momentum Pure Top 3", layout="wide")

def calculate_metrics(returns):
    """Calcule les métriques de performance de manière sécurisée"""
    if returns.empty:
        return 0, 0, 0, 0, 0
    total_return = (returns + 1).prod() - 1
    n_years = max(len(returns) / 12, 0.1)
    cagr = (total_return + 1) ** (1 / n_years) - 1
    vol = returns.std() * np.sqrt(12)
    sharpe = (cagr - 0.02) / vol if vol > 0 else 0
    cum_rets = (returns + 1).cumprod()
    drawdown = (cum_rets / cum_rets.cummax() - 1).min()
    return cagr, vol, sharpe, drawdown, total_return

def run_momentum_pure():
    st.title("🚀 Momentum Pure : Top 3 Secteurs")
    st.info("Stratégie : Sélection des 3 meilleurs secteurs sur 3 mois. Rebalancement mensuel.")

    sectors = ['XLK', 'XLF', 'XLV', 'XLY', 'XLI', 'XLP', 'XLE', 'XLC', 'XLB', 'XLU', 'XLRE']
    
    @st.cache_data
    def load_data():
        # Téléchargement avec une marge pour le calcul du momentum initial
        data = yf.download(sectors + ['SPY'], start="1998-10-01", progress=False)
        # Extraction propre des colonnes
        close_prices = data['Close'].ffill()
        open_prices = data['Open'].ffill()
        return close_prices, open_prices

    try:
        close_data, open_data = load_data()
        
        # Resample mensuel : fin de mois pour les signaux
        monthly_close = close_data.resample('ME').last()
        # Calcul du Momentum (Look-back 3 mois)
        momentum = monthly_close[sectors].pct_change(3)
        
        history = []
        
        # On commence le backtest après les 3 mois de look-back (index 3)
        for i in range(3, len(monthly_close) - 1):
            # Signaux à la fin du mois i
            scores = momentum.iloc[i].dropna().sort_values(ascending=False)
            if len(scores) < 3: continue
            
            top_3 = scores.index[:3].tolist()
            
            # Dates pour le mois i+1 (investissement)
            date_debut = monthly_close.index[i] + pd.Timedelta(days=1)
            date_fin = monthly_close.index[i+1]
            
            # Sécurité pour trouver les indices réels dans le calendrier boursier
            try:
                idx_start = open_data.index.get_indexer([date_debut], method='bfill')[0]
                idx_end = close_data.index.get_indexer([date_fin], method='ffill')[0]
                
                # Calcul rendement Stratégie (33.3% par ticker)
                m_ret = sum((close_data[t].iloc[idx_end] / open_data[t].iloc[idx_start]) - 1 for t in top_3) / 3
                
                # Calcul rendement Benchmark (SPY)
                s_ret = (close_data['SPY'].iloc[idx_end] / open_data['SPY'].iloc[idx_start]) - 1
                
                history.append({'Date': monthly_close.index[i+1], 'Strat': m_ret, 'SPY': s_ret})
            except:
                continue

        if not history:
            st.error("Pas assez de données pour générer le backtest.")
            return

        df = pd.DataFrame(history).set_index('Date')
        
        # --- Affichage des Métriques ---
        m_s = calculate_metrics(df['Strat'])
        m_b = calculate_metrics(df['SPY'])

        cols = st.columns(4)
        cols[0].metric("CAGR Stratégie", f"{m_s[0]*100:.2f}%")
        cols[1].metric("Ratio Sharpe", f"{m_s[2]:.2f}")
        cols[2].metric("Max Drawdown", f"{m_s[3]*100:.2f}%")
        cols[3].metric("Total Return", f"{m_s[4]*100:.1f}%")

        # --- Graphique ---
        st.subheader("Performance Cumulée (Base 100)")
        cum_df = (1 + df).cumprod() * 100
        st.line_chart(cum_df)

        # --- Tableau Récapitulatif ---
        st.table(pd.DataFrame({
            'Métrique': ['CAGR', 'Sharpe', 'Max Drawdown', 'Total Return'],
            'Stratégie Top 3': [f"{m_s[0]*100:.2f}%", f"{m_s[2]:.2f}", f"{m_s[3]*100:.2f}%", f"{m_s[4]*100:.1f}%"],
            'S&P 500 (SPY)': [f"{m_b[0]*100:.2f}%", f"{m_b[2]:.2f}", f"{m_b[3]*100:.2f}%", f"{m_b[4]*100:.1f}%"]
        }))

        # --- Signaux Actuels ---
        st.divider()
        st.subheader("🎯 Signaux pour le mois suivant")
        current_scores = momentum.iloc[-1].sort_values(ascending=False)
        top_now = current_scores.index[:3].tolist()
        
        sig_cols = st.columns(3)
        for i, t in enumerate(top_now):
            sig_cols[i].success(f"Top {i+1} : **{t}**")

    except Exception as e:
        st.error(f"Erreur technique : {e}")

if __name__ == "__main__":
    run_momentum_pure()
