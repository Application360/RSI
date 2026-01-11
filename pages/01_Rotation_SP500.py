import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date

# 1. Configuration de la page
st.set_page_config(page_title="Momentum Analytics Pro", layout="wide")

def calculate_metrics(returns):
    """Calcule les métriques de performance de manière sécurisée"""
    if returns.empty:
        return 0, 0, 0, 0, 0, 0
    total_return = (returns + 1).prod() - 1
    days = (returns.index[-1] - returns.index[0]).days
    n_years = max(days / 365.25, 0.1)
    
    cagr = (total_return + 1) ** (1 / n_years) - 1
    # Calcul de la volatilité annualisée
    vol = returns.std() * np.sqrt(12)
    sharpe = (cagr) / vol if vol > 0 else 0
    cum_rets = (returns + 1).cumprod()
    drawdown = (cum_rets / cum_rets.cummax() - 1).min()
    return cagr, vol, sharpe, drawdown, total_return

def run_momentum_pure():
    st.title("🚀 Momentum Sectoriel : Stratégie & Volatilité")
    
    sectors = ['XLK', 'XLF', 'XLV', 'XLY', 'XLI', 'XLP', 'XLE', 'XLC', 'XLB', 'XLU', 'XLRE']
    
    # --- BARRE LATÉRALE ---
    with st.sidebar:
        st.header("⚙️ Paramètres")
        n_top = st.slider("Nombre de secteurs à détenir", 1, 5, 3)
        lookback = st.slider("Période d'analyse (Look-back en mois)", 1, 12, 3)
        holding_period = st.slider("Durée de détention (Holding en mois)", 1, 12, 1)
        
        start_date = st.date_input("Date de début", value=date(1999, 1, 1), min_value=date(1999, 1, 1), max_value=date(2026, 12, 31))
        end_date = st.date_input("Date de fin", value=date(2026, 12, 31), min_value=date(1999, 1, 1), max_value=date(2026, 12, 31))
        st.divider()
        st.info(f"Configuration : Top {n_top} / Look-back {lookback}m / Holding {holding_period}m")

    if start_date >= end_date:
        st.error("La date de début doit être antérieure à la date de fin.")
        return

    @st.cache_data
    def load_data(s_date, e_date, lb_period):
        margin_start = pd.to_datetime(s_date) - pd.DateOffset(months=lb_period + 1)
        data = yf.download(sectors + ['SPY'], start=margin_start, end=e_date, progress=False)
        if data.empty: return pd.DataFrame(), pd.DataFrame()
        return data['Close'].ffill(), data['Open'].ffill()

    try:
        with st.spinner('Analyse en cours...'):
            close_data, open_data = load_data(start_date, end_date, lookback)
            if close_data.empty: return

            monthly_close = close_data.resample('ME').last()
            momentum = monthly_close[sectors].pct_change(lookback)
            
            history = []
            portfolio_changes = 0
            current_top = []
            
            start_dt = pd.to_datetime(start_date)
            valid_start_idx = lookback
            for j in range(len(monthly_close)):
                if monthly_close.index[j] >= start_dt and j >= lookback:
                    valid_start_idx = j
                    break

            for i in range(valid_start_idx, len(monthly_close) - 1):
                if (i - valid_start_idx) % holding_period == 0:
                    scores = momentum.iloc[i].dropna().sort_values(ascending=False)
                    if len(scores) < n_top: continue
                    new_top = scores.index[:n_top].tolist()
                    if current_top:
                        portfolio_changes += len([s for s in new_top if s not in current_top])
                    else:
                        portfolio_changes += n_top
                    current_top = new_top
                
                d_start, d_end = monthly_close.index[i] + pd.Timedelta(days=1), monthly_close.index[i+1]
                try:
                    idx_s = open_data.index.get_indexer([d_start], method='bfill')[0]
                    idx_e = close_data.index.get_indexer([d_end], method='ffill')[0]
                    strat_ret = sum((close_data[t].iloc[idx_e] / open_data[t].iloc[idx_s]) - 1 for t in current_top) / n_top
                    spy_ret = (close_data['SPY'].iloc[idx_e] / open_data['SPY'].iloc[idx_s]) - 1
                    history.append({'Date': monthly_close.index[i+1], 'Strat': strat_ret, 'SPY': spy_ret})
                except: continue

        if not history:
            st.warning("Données insuffisantes.")
            return

        df = pd.DataFrame(history).set_index('Date')
        # m_s[1] et m_b[1] contiennent désormais la volatilité
        m_s = calculate_metrics(df['Strat'])
        m_b = calculate_metrics(df['SPY'])

        # --- DASHBOARD ---
        st.subheader(f"📊 Performance & Risque")
        
        # Ajout d'une ligne de metrics incluant la volatilité
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("CAGR Stratégie", f"{m_s[0]*100:.2f}%")
        c2.metric("Volatilité Ann.", f"{m_s[1]*100:.2f}%")
        c3.metric("Ratio Sharpe", f"{m_s[2]:.2f}")
        c4.metric("Max Drawdown", f"{m_s[3]*100:.2f}%")
        c5.metric("Transactions", f"{portfolio_changes}")

        st.line_chart((1 + df).cumprod() * 100)

        # Tableau de comparaison détaillé
        st.table(pd.DataFrame({
            'Métrique': ['Rendement Annuel (CAGR)', 'Volatilité Annuelle', 'Ratio Sharpe', 'Max Drawdown', 'Performance Totale', 'Transactions'],
            'Stratégie': [f"{m_s[0]*100:.2f}%", f"{m_s[1]*100:.2f}%", f"{m_s[2]:.2f}", f"{m_s[3]*100:.2f}%", f"{m_s[4]*100:.1f}%", portfolio_changes],
            'S&P 500': [f"{m_b[0]*100:.2f}%", f"{m_b[1]*100:.2f}%", f"{m_b[2]:.2f}", f"{m_b[3]*100:.2f}%", f"{m_b[4]*100:.1f}%", "-"]
        }))

        st.divider()
        st.subheader("🎯 Positions Actuelles")
        scols = st.columns(n_top)
        for idx, t in enumerate(current_top):
            scols[idx].success(f"Position {idx+1} : **{t}**")

    except Exception as e:
        st.error(f"Erreur : {e}")

if __name__ == "__main__":
    run_momentum_pure()
