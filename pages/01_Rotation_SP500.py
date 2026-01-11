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
        # Slider pour la Moyenne Mobile (Valeur par défaut 150)
        sma_period = st.slider("Période Moyenne Mobile S&P 500", 50, 250, 150)
        
        st.divider()
        start_date = st.date_input("Date de début", value=date(1999, 1, 1))
        end_date = st.date_input("Date de fin", value=date(2026, 12, 31))

    @st.cache_data
    def load_data(s_date, e_date, lb_period, sma_p):
        # On télécharge un peu plus de données au début pour calculer la MM et le Momentum dès le 1er jour
        margin_start = pd.to_datetime(s_date) - pd.DateOffset(days=max(lb_period * 31, sma_p) + 40)
        data = yf.download(sectors + ['SPY'], start=margin_start, end=e_date, progress=False)
        if data.empty: return pd.DataFrame(), pd.DataFrame(), pd.Series()
        
        closes = data['Close'].ffill()
        opens = data['Open'].ffill()
        # Calcul de la Moyenne Mobile sur le S&P 500 (SPY)
        spy_sma = closes['SPY'].rolling(window=sma_p).mean()
        
        return closes, opens, spy_sma

    try:
        with st.spinner('Analyse en cours...'):
            close_data, open_data, spy_sma = load_data(start_date, end_date, lookback, sma_period)
            if close_data.empty: return

            # Données mensuelles pour le momentum
            monthly_close = close_data.resample('ME').last()
            momentum = monthly_close[sectors].pct_change(lookback)
            
            history = []
            portfolio_changes = 0
            current_top = [] # Si vide = Cash
            
            start_dt = pd.to_datetime(start_date)
            valid_start_idx = lookback
            for j in range(len(monthly_close)):
                if monthly_close.index[j] >= start_dt and j >= lookback:
                    valid_start_idx = j
                    break

            for i in range(valid_start_idx, len(monthly_close) - 1):
                monthly_fees = 0
                dt_now = monthly_close.index[i]
                
                # --- RÉSOLUTION DU BUG DE TIMESTAMP ---
                # On utilise get_indexer pour trouver l'indice le plus proche existant (évite le KeyError)
                idx_ref = close_data.index.get_indexer([dt_now], method='ffill')[0]
                price_spy_now = close_data['SPY'].iloc[idx_ref]
                sma_spy_now = spy_sma.iloc[idx_ref]

                # --- LOGIQUE DE REBALANCEMENT & FILTRE ---
                if (i - valid_start_idx) % holding_period == 0:
                    # Filtre : Si S&P 500 > sa Moyenne Mobile
                    is_market_bull = price_spy_now > sma_spy_now
                    
                    if is_market_bull:
                        scores = momentum.iloc[i].dropna().sort_values(ascending=False)
                        new_top = scores.index[:n_top].tolist()
                        
                        if current_top:
                            num_changes = len([s for s in new_top if s not in current_top])
                            if num_changes > 0:
                                portfolio_changes += num_changes
                                monthly_fees = (num_changes / n_top) * fees_pct
                        else:
                            portfolio_changes += n_top
                            monthly_fees = fees_pct
                        current_top = new_top
                    else:
                        # MARCHÉ BAISSIER -> PASSAGE EN CASH
                        if current_top:
                            portfolio_changes += len(current_top)
                            monthly_fees = fees_pct
                        current_top = []

                # --- CALCUL DU RENDEMENT MENSUEL ---
                d_start, d_end = monthly_close.index[i] + pd.Timedelta(days=1), monthly_close.index[i+1]
                try:
                    idx_s = open_data.index.get_indexer([d_start], method='bfill')[0]
                    idx_e = close_data.index.get_indexer([d_end], method='ffill')[0]
                    
                    if current_top:
                        gross_ret = sum((close_data[t].iloc[idx_e] / open_data[t].iloc[idx_s]) - 1 for t in current_top) / n_top
                    else:
                        gross_ret = 0.0 # Performance nulle en Cash
                    
                    strat_ret_net = gross_ret - monthly_fees
                    spy_ret = (close_data['SPY'].iloc[idx_e] / open_data['SPY'].iloc[idx_s]) - 1
                    history.append({'Date': monthly_close.index[i+1], 'Strat': strat_ret_net, 'SPY': spy_ret})
                except: continue

        if not history:
            st.warning("Données insuffisantes.")
            return

        df = pd.DataFrame(history).set_index('Date')
        m_s = calculate_metrics(df['Strat'])
        m_b = calculate_metrics(df['SPY'])

        # --- DASHBOARD ---
        st.subheader(f"📊 Métriques Net de Frais (Filtre MM {sma_period}j)")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🔹 Ma Stratégie**")
            cols = st.columns(4)
            cols[0].metric("CAGR Net", f"{m_s[0]*100:.2f}%")
            cols[1].metric("Ratio Sharpe", f"{m_s[2]:.2f}")
            cols[2].metric("Max Drawdown", f"{m_s[3]*100:.2f}%")
            cols[3].metric("Transactions", portfolio_changes)
        
        with c2:
            st.markdown("**🔸 S&P 500 (Benchmark)**")
            cols = st.columns(4)
            cols[0].metric("CAGR", f"{m_b[0]*100:.2f}%")
            cols[1].metric("Ratio Sharpe", f"{m_b[2]:.2f}")
            cols[2].metric("Max Drawdown", f"{m_b[3]*100:.2f}%")

        st.divider()
        
        # --- GRAPHIQUES ---
        st.subheader("📈 Performance Cumulée")
        st.line_chart((1 + df[['Strat', 'SPY']]).cumprod() * 100)

        # --- TABLEAU ANNUEL ---
        st.subheader("📅 Détail Annuel")
        annual = df.groupby(df.index.year).apply(lambda x: (1 + x).prod() - 1)
        annual['Alpha'] = annual['Strat'] - annual['SPY']
        st.table(annual.sort_index(ascending=False).style.format("{:.2%}"))

        # --- SIGNAL ACTUEL ---
        st.divider()
        is_bull = close_data['SPY'].iloc[-1] > spy_sma.iloc[-1]
        if is_bull:
            st.success(f"✅ MARCHÉ HAUSSIER : SPY > MM{sma_period}. Secteurs actifs : {', '.join(current_top) if current_top else 'Attente rebalancement'}")
        else:
            st.error(f"⚠️ MARCHÉ BAISSIER : SPY < MM{sma_period}. POSITION : CASH 🛡️")

    except Exception as e:
        st.error(f"Erreur technique : {e}")

if __name__ == "__main__":
    run_momentum_pure()
