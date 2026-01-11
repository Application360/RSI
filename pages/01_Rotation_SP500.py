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
    st.title("🚀 Momentum Pro : Rotation Lente & Sécurité Réactive")
    
    sectors = ['XLK', 'XLF', 'XLV', 'XLY', 'XLI', 'XLP', 'XLE', 'XLC', 'XLB', 'XLU', 'XLRE']
    
    # --- BARRE LATÉRALE ---
    with st.sidebar:
        st.header("⚙️ Paramètres Stratégie")
        n_top = st.slider("Nombre de secteurs à détenir", 1, 5, 2)
        lookback = st.slider("Look-back Momentum (mois)", 1, 12, 6)
        holding_period = st.slider("Fréquence rotation secteurs (mois)", 1, 12, 9)
        fees_pct = st.slider("Frais par transaction (%)", 0.0, 1.0, 0.1, step=0.05) / 100
        
        st.divider()
        st.header("🛡️ Market Timing (Mensuel)")
        use_market_timing = st.checkbox("Activer le filtre de tendance", value=True)
        sma_period = st.slider("Moyenne Mobile S&P 500 (jours)", 50, 300, 150, disabled=not use_market_timing)
        
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
        with st.spinner('Analyse des cycles en cours...'):
            close_data, open_data, spy_sma = load_data(start_date, end_date, lookback, sma_period)
            if close_data.empty: return

            monthly_close = close_data.resample('ME').last()
            momentum = monthly_close[sectors].pct_change(lookback)
            
            history = []
            portfolio_changes = 0
            current_top = []
            is_invested = False 
            
            start_dt = pd.to_datetime(start_date)
            valid_start_idx = lookback
            for j in range(len(monthly_close)):
                if monthly_close.index[j] >= start_dt and j >= lookback:
                    valid_start_idx = j
                    break

            for i in range(valid_start_idx, len(monthly_close) - 1):
                monthly_fees = 0
                dt_now = monthly_close.index[i]
                
                # A. VERIFICATION MENSUELLE DE LA SÉCURITÉ
                idx_ref = spy_sma.index.get_indexer([dt_now], method='ffill')[0]
                price_spy = close_data['SPY'].iloc[idx_ref]
                val_sma = spy_sma.iloc[idx_ref]
                market_is_bull = (price_spy > val_sma) if use_market_timing else True

                # B. ROTATION DES SECTEURS (Lente)
                if (i - valid_start_idx) % holding_period == 0:
                    scores = momentum.iloc[i].dropna().sort_values(ascending=False)
                    new_top = scores.index[:n_top].tolist()
                    
                    if is_invested and current_top:
                        num_changes = len([s for s in new_top if s not in current_top])
                        portfolio_changes += num_changes
                        monthly_fees += (num_changes / n_top) * fees_pct
                    
                    current_top = new_top

                # C. DÉCISION D'ENTRÉE/SORTIE (Réactive chaque mois)
                if market_is_bull and not is_invested:
                    is_invested = True
                    portfolio_changes += len(current_top)
                    monthly_fees += fees_pct
                elif not market_is_bull and is_invested:
                    is_invested = False
                    portfolio_changes += len(current_top)
                    monthly_fees += fees_pct

                # D. CALCUL RENDEMENT
                d_start, d_end = monthly_close.index[i] + pd.Timedelta(days=1), monthly_close.index[i+1]
                try:
                    idx_s = open_data.index.get_indexer([d_start], method='bfill')[0]
                    idx_e = close_data.index.get_indexer([d_end], method='ffill')[0]
                    
                    if is_invested:
                        gross_ret = sum((close_data[t].iloc[idx_e] / open_data[t].iloc[idx_s]) - 1 for t in current_top) / n_top
                    else:
                        gross_ret = 0.0 
                    
                    history.append({
                        'Date': monthly_close.index[i+1], 
                        'Strat': gross_ret - monthly_fees, 
                        'SPY': (close_data['SPY'].iloc[idx_e] / open_data['SPY'].iloc[idx_s]) - 1
                    })
                except: continue

        if not history:
            st.warning("Données insuffisantes.")
            return

        df = pd.DataFrame(history).set_index('Date')
        m_s = calculate_metrics(df['Strat'])
        m_b = calculate_metrics(df['SPY'])

        # --- DASHBOARD ---
        st.subheader(f"📊 Métriques Stratégie vs S&P 500")
        st.markdown(f"### 🔹 Ma Stratégie (Rotation {holding_period}m / Sécurité Mensuelle)")
        c = st.columns(5)
        c[0].metric("CAGR Net", f"{m_s[0]*100:.2f}%")
        c[1].metric("Ratio Sharpe", f"{m_s[2]:.2f}")
        c[2].metric("Max Drawdown", f"{m_s[3]*100:.2f}%")
        c[3].metric("Volatilité", f"{m_s[1]*100:.2f}%")
        c[4].metric("Transactions", portfolio_changes)
        
        st.markdown("### 🔸 S&P 500")
        b = st.columns(5)
        b[0].metric("CAGR", f"{m_b[0]*100:.2f}%")
        b[1].metric("Ratio Sharpe", f"{m_b[2]:.2f}")
        b[2].metric("Max Drawdown", f"{m_b[3]*100:.2f}%")
        b[3].metric("Volatilité", f"{m_b[1]*100:.2f}%")

        st.divider()

        # --- GRAPHIQUES ---
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("📈 Performance Cumulée")
            st.line_chart((1 + df[['Strat', 'SPY']]).cumprod() * 100)
        with g2:
            st.subheader("📉 Risque : Drawdown Historique (%)")
            dd_strat = ((1 + df['Strat']).cumprod() / (1 + df['Strat']).cumprod().cummax() - 1) * 100
            dd_spy = ((1 + df['SPY']).cumprod() / (1 + df['SPY']).cumprod().cummax() - 1) * 100
            st.area_chart(pd.DataFrame({'Stratégie': dd_strat, 'S&P 500': dd_spy}))

        # --- TABLEAU ANNUEL AVEC SURLIGNAGE ALPHA ---
        st.subheader("📅 Détail Annuel & Alpha")
        annual = df[['Strat', 'SPY']].groupby(df.index.year).apply(lambda x: (1 + x).prod() - 1)
        annual['Alpha'] = annual['Strat'] - annual['SPY']

        # Fonction de style pour l'Alpha
        def highlight_alpha(val):
            color = 'background-color: #2ecc71; color: white' if val >= 0 else ''
            return color

        styled_annual = annual.sort_index(ascending=False).style.format("{:.2%}")\
            .applymap(highlight_alpha, subset=['Alpha'])
        
        st.table(styled_annual)

        # --- ÉTAT ACTUEL ---
        st.divider()
        st.subheader(f"🎯 État du système au {date.today().year}")
        if is_invested:
            st.success(f"✅ ÉTAT : INVESTI (SPY > MM{sma_period})")
            st.info(f"🚀 **Secteurs sélectionnés :** {', '.join(current_top)}")
        else:
            st.error(f"🛡️ ÉTAT : CASH (SPY < MM{sma_period})")
            st.warning("En attente d'un signal haussier mensuel pour réinvestir.")

        csv = df.to_csv().encode('utf-8')
        st.download_button("📥 Télécharger CSV", data=csv, file_name="backtest.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Erreur technique : {e}")

if __name__ == "__main__":
    run_momentum_pure()
