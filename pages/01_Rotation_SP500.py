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
        return 0, 0, 0, 0, 0
    total_return = (returns + 1).prod() - 1
    days = (returns.index[-1] - returns.index[0]).days
    n_years = max(days / 365.25, 0.1)
    
    cagr = (total_return + 1) ** (1 / n_years) - 1
    vol = returns.std() * np.sqrt(12)
    sharpe = (cagr) / vol if vol > 0 else 0
    cum_rets = (returns + 1).cumprod()
    drawdown = (cum_rets / cum_rets.cummax() - 1).min()
    return cagr, vol, sharpe, drawdown, total_return

def run_momentum_pure():
    st.title("🚀 Momentum Sectoriel avec Filtre de Tendance MM")
    
    sectors = ['XLK', 'XLF', 'XLV', 'XLY', 'XLI', 'XLP', 'XLE', 'XLC', 'XLB', 'XLU', 'XLRE']
    
    # --- BARRE LATÉRALE ---
    with st.sidebar:
        st.header("⚙️ Paramètres")
        n_top = st.slider("Nombre de secteurs à détenir", 1, 5, 2)
        lookback = st.slider("Période Momentum (Look-back mois)", 1, 12, 6)
        holding_period = st.slider("Durée de détention (Holding mois)", 1, 12, 9)
        fees_pct = st.slider("Frais par transaction (%)", 0.0, 1.0, 0.1, step=0.05) / 100
        
        st.divider()
        st.header("🛡️ Protection (Market Timing)")
        sma_period = st.slider("Moyenne Mobile S&P 500 (jours)", 50, 250, 150)
        
        st.divider()
        start_date = st.date_input("Date de début", value=date(1999, 1, 1))
        end_date = st.date_input("Date de fin", value=date(2026, 12, 31))

    @st.cache_data
    def load_data(s_date, e_date, lb_period, sma_p):
        margin_start = pd.to_datetime(s_date) - pd.DateOffset(days=max(lb_period * 31, sma_p) + 60)
        data = yf.download(sectors + ['SPY'], start=margin_start, end=e_date, progress=False)
        if data.empty: return pd.DataFrame(), pd.DataFrame(), pd.Series()
        
        closes = data['Close'].ffill()
        opens = data['Open'].ffill()
        spy_sma = closes['SPY'].rolling(window=sma_p).mean()
        return closes, opens, spy_sma

    try:
        with st.spinner('Analyse en cours...'):
            close_data, open_data, spy_sma = load_data(start_date, end_date, lookback, sma_period)
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
                monthly_fees = 0
                dt_now = monthly_close.index[i]
                
                # --- FIX BUG TIMESTAMP ---
                idx_ref = spy_sma.index.get_indexer([dt_now], method='ffill')[0]
                price_spy = close_data['SPY'].iloc[idx_ref]
                val_sma = spy_sma.iloc[idx_ref]

                if (i - valid_start_idx) % holding_period == 0:
                    is_market_bull = price_spy > val_sma
                    
                    if is_market_bull:
                        scores = momentum.iloc[i].dropna().sort_values(ascending=False)
                        new_top = scores.index[:n_top].tolist()
                        if current_top:
                            num_changes = len([s for s in new_top if s not in current_top])
                            portfolio_changes += num_changes
                            monthly_fees = (num_changes / n_top) * fees_pct
                        else:
                            portfolio_changes += n_top
                            monthly_fees = fees_pct
                        current_top = new_top
                    else:
                        if current_top:
                            portfolio_changes += len(current_top)
                            monthly_fees = fees_pct
                        current_top = []

                d_start, d_end = monthly_close.index[i] + pd.Timedelta(days=1), monthly_close.index[i+1]
                try:
                    idx_s = open_data.index.get_indexer([d_start], method='bfill')[0]
                    idx_e = close_data.index.get_indexer([d_end], method='ffill')[0]
                    gross_ret = sum((close_data[t].iloc[idx_e] / open_data[t].iloc[idx_s]) - 1 for t in current_top) / n_top if current_top else 0.0
                    spy_ret = (close_data['SPY'].iloc[idx_e] / open_data['SPY'].iloc[idx_s]) - 1
                    history.append({'Date': monthly_close.index[i+1], 'Strat': gross_ret - monthly_fees, 'SPY': spy_ret})
                except: continue

        if not history:
            st.warning("Données insuffisantes.")
            return

        df = pd.DataFrame(history).set_index('Date')
        m_s = calculate_metrics(df['Strat'])
        m_b = calculate_metrics(df['SPY'])

        # --- DESIGN DES MÉTRIQUES SUPERPOSÉES ---
        st.subheader(f"📊 Métriques Net de Frais (MM {sma_period}j)")

        # Bloc Ma Stratégie
        st.markdown("### 🔹 Ma Stratégie")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("CAGR Net", f"{m_s[0]*100:.2f}%")
        c2.metric("Ratio Sharpe", f"{m_s[2]:.2f}")
        c3.metric("Max Drawdown", f"{m_s[3]*100:.2f}%")
        c4.metric("Volatilité", f"{m_s[1]*100:.2f}%")
        c5.metric("Transactions", portfolio_changes)
        
        st.markdown("---")

        # Bloc S&P 500
        st.markdown("### 🔸 S&P 500 (Benchmark)")
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("CAGR", f"{m_b[0]*100:.2f}%")
        b2.metric("Ratio Sharpe", f"{m_b[2]:.2f}")
        b3.metric("Max Drawdown", f"{m_b[3]*100:.2f}%")
        b4.metric("Volatilité", f"{m_b[1]*100:.2f}%")
        b5.write("") # Espace vide pour l'alignement

        st.divider()
        
        # --- GRAPHIQUES ---
        st.subheader("📈 Performance Cumulée")
        st.line_chart((1 + df[['Strat', 'SPY']]).cumprod() * 100)

        # --- TABLEAU ANNUEL ---
        st.subheader("📅 Détail Annuel")
        annual = df.groupby(df.index.year).apply(lambda x: (1 + x).prod() - 1)
        annual['Alpha'] = annual['Strat'] - annual['SPY']
        st.table(annual.sort_index(ascending=False).style.format("{:.2%}"))

        # --- ÉTAT ACTUEL & SECTEURS ---
        st.divider()
        st.subheader("🎯 État actuel du portefeuille")
        
        # On recalcule les secteurs les plus récents pour l'affichage final
        latest_momentum = momentum.iloc[-1].dropna().sort_values(ascending=False)
        top_now = latest_momentum.index[:n_top].tolist()
        
        is_bull = close_data['SPY'].iloc[-1] > spy_sma.iloc[-1]
        
        if is_bull:
            st.success(f"✅ MARCHÉ HAUSSIER (SPY > MM{sma_period})")
            st.info(f"🚀 **Secteurs à détenir actuellement :** {', '.join(top_now)}")
        else:
            st.error(f"⚠️ MARCHÉ BAISSIER (SPY < MM{sma_period})")
            st.warning("🛡️ **POSITION ACTUELLE : CASH (0% exposition)**")

    except Exception as e:
        st.error(f"Erreur technique : {e}")

if __name__ == "__main__":
    run_momentum_pure()
