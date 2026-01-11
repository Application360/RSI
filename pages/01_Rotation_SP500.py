import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date

# 1. Configuration de la page
st.set_page_config(page_title="Momentum Pure 1999-2026", layout="wide")

def calculate_metrics(returns):
    """Calcule les métriques de performance de manière sécurisée"""
    if returns.empty:
        return 0, 0, 0, 0, 0
    total_return = (returns + 1).prod() - 1
    # Calcul des années réelles sur la période
    days = (returns.index[-1] - returns.index[0]).days
    n_years = max(days / 365.25, 0.1)
    
    cagr = (total_return + 1) ** (1 / n_years) - 1
    vol = returns.std() * np.sqrt(12)
    sharpe = (cagr - 0.02) / vol if vol > 0 else 0
    cum_rets = (returns + 1).cumprod()
    drawdown = (cum_rets / cum_rets.cummax() - 1).min()
    return cagr, vol, sharpe, drawdown, total_return

def run_momentum_pure():
    st.title("🚀 Momentum Pure : Top 3 Secteurs (1999-2026)")
    
    sectors = ['XLK', 'XLF', 'XLV', 'XLY', 'XLI', 'XLP', 'XLE', 'XLC', 'XLB', 'XLU', 'XLRE']
    
    # --- BARRE LATÉRALE ---
    with st.sidebar:
        st.header("⚙️ Paramètres")
        start_date = st.date_input(
            "Date de début", 
            value=date(1999, 1, 1),
            min_value=date(1999, 1, 1),
            max_value=date(2026, 12, 31)
        )
        end_date = st.date_input(
            "Date de fin", 
            value=date(2026, 12, 31),
            min_value=date(1999, 1, 1),
            max_value=date(2026, 12, 31)
        )
        st.divider()
        st.write("Stratégie : Sélection des 3 meilleurs secteurs (look-back 3 mois).")
        
    if start_date >= end_date:
        st.error("La date de début doit être antérieure à la date de fin.")
        return

    @st.cache_data
    def load_data(s_date, e_date):
        # Marge de 4 mois pour le calcul du momentum initial
        margin_start = pd.to_datetime(s_date) - pd.DateOffset(months=4)
        data = yf.download(sectors + ['SPY'], start=margin_start, end=e_date, progress=False)
        
        if data.empty:
            return pd.DataFrame(), pd.DataFrame()
            
        close_prices = data['Close'].ffill()
        open_prices = data['Open'].ffill()
        return close_prices, open_prices

    try:
        with st.spinner('Chargement des données...'):
            close_data, open_data = load_data(start_date, end_date)
            
            if close_data.empty:
                st.warning("Aucune donnée disponible.")
                return

            # Analyse mensuelle
            monthly_close = close_data.resample('ME').last()
            momentum = monthly_close[sectors].pct_change(3)
            
            history = []
            start_dt = pd.to_datetime(start_date)
            
            # Index de départ
            valid_start_idx = 3
            for j in range(len(monthly_close)):
                if monthly_close.index[j] >= start_dt and j >= 3:
                    valid_start_idx = j
                    break

            # Backtest
            for i in range(valid_start_idx, len(monthly_close) - 1):
                scores = momentum.iloc[i].dropna().sort_values(ascending=False)
                if len(scores) < 3: continue
                top_3 = scores.index[:3].tolist()
                
                d_start = monthly_close.index[i] + pd.Timedelta(days=1)
                d_end = monthly_close.index[i+1]
                
                try:
                    idx_s = open_data.index.get_indexer([d_start], method='bfill')[0]
                    idx_e = close_data.index.get_indexer([d_end], method='ffill')[0]
                    
                    strat_ret = sum((close_data[t].iloc[idx_e] / open_data[t].iloc[idx_s]) - 1 for t in top_3) / 3
                    spy_ret = (close_data['SPY'].iloc[idx_e] / open_data['SPY'].iloc[idx_s]) - 1
                    
                    history.append({'Date': monthly_close.index[i+1], 'Strat': strat_ret, 'SPY': spy_ret})
                except:
                    continue

        if not history:
            st.warning("Période trop courte.")
            return

        df = pd.DataFrame(history).set_index('Date')
        m_s = calculate_metrics(df['Strat'])
        m_b = calculate_metrics(df['SPY'])

        # --- Dashboard ---
        st.subheader(f"📊 Résultats du {start_date} au {df.index[-1].date()}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAGR Stratégie", f"{m_s[0]*100:.2f}%")
        c2.metric("Sharpe", f"{m_s[2]:.2f}")
        c3.metric("Max Drawdown", f"{m_s[3]*100:.2f}%")
        c4.metric("Total Return", f"{m_s[4]*100:.1f}%")

        st.line_chart((1 + df).cumprod() * 100)

        st.table(pd.DataFrame({
            'Métrique': ['CAGR', 'Sharpe', 'Drawdown', 'Total Return'],
            'Stratégie': [f"{m_s[0]*100:.2f}%", f"{m_s[2]:.2f}", f"{m_s[3]*100:.2f}%", f"{m_s[4]*100:.1f}%"],
            'S&P 500': [f"{m_b[0]*100:.2f}%", f"{m_b[2]:.2f}", f"{m_b[3]*100:.2f}%", f"{m_b[4]*100:.1f}%"]
        }))

        # Signaux
        st.divider()
        st.subheader("🎯 Signaux actuels")
        last_m = momentum.iloc[-1].sort_values(ascending=False)
        top_now = last_m.index[:3].tolist()
        
        scols = st.columns(3)
        for idx, t in enumerate(top_now):
            scols[idx].success(f"Top {idx+1} : **{t}**")

    except Exception as e:
        st.error(f"Erreur technique : {e}")

if __name__ == "__main__":
    run_momentum_pure()
