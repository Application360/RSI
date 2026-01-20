import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date

# 1. Configuration de la page
st.set_page_config(page_title="Momentum Analytics Pro", layout="wide")

def calculate_metrics(returns):
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
    st.title("🚀 Momentum Pro : Benchmark Officiel ^GSPC")
    
    sectors = ['XLK', 'XLF', 'XLV', 'XLY', 'XLI', 'XLP', 'XLE', 'XLC', 'XLB', 'XLU', 'XLRE']
    
    with st.sidebar:
        st.header("⚙️ Paramètres Stratégie")
        n_top = st.slider("Nombre de secteurs à détenir", 1, 5, 2)
        lookback = st.slider("Look-back Momentum (mois)", 1, 12, 6)
        holding_period = st.slider("Fréquence rotation secteurs (mois)", 1, 12, 9)
        fees_pct = st.slider("Frais par transaction (%)", 0.0, 0.5, 0.1, step=0.01) / 100
        
        st.divider()
        st.header("🛡️ Market Timing")
        use_market_timing = st.checkbox("Activer le filtre de tendance", value=True)
        sma_period = st.slider("Moyenne Mobile S&P 500 (jours)", 50, 250, 150, disabled=not use_market_timing)
        
        st.divider()
        st.header("📅 Période")
        min_date, max_date = date(1999, 1, 1), date(2026, 12, 31)
        start_date = st.date_input("Début", value=min_date)
        end_date = st.date_input("Fin", value=max_date)

    @st.cache_data
    def load_data(s_date, e_date, lb_period, sma_p):
        margin_start = pd.to_datetime(s_date) - pd.DateOffset(days=max(lb_period * 31, sma_p) + 60)
        data = yf.download(sectors + ['^GSPC'], start=margin_start, end=e_date, progress=False)
        
        if data.empty: return pd.DataFrame(), pd.DataFrame(), pd.Series()
        
        # Sécurisation : Extraction des prix pour les secteurs (Adj Close) et l'indice (Close)
        # On utilise .xs pour gérer le multi-index de yfinance proprement
        df_adj = data['Adj Close'][sectors].ffill()
        df_close_idx = data['Close']['^GSPC'].ffill()
        
        closes = pd.concat([df_adj, df_close_idx.rename('^GSPC')], axis=1)
        opens = data['Open'].ffill() # Pour le calcul des returns mensuels
        
        spy_sma = closes['^GSPC'].rolling(window=sma_p).mean()
        return closes, opens, spy_sma

    try:
        with st.spinner('Analyse des données en cours...'):
            close_data, open_data, spy_sma = load_data(start_date, end_date, lookback, sma_period)
            if close_data.empty: return

            monthly_close = close_data.resample('ME').last()
            momentum = monthly_close[sectors].pct_change(lookback)
            
            history = []
            pos_history = [] 
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
                
                # Market Timing
                idx_ref = spy_sma.index.get_indexer([dt_now], method='ffill')[0]
                market_is_bull = (close_data['^GSPC'].iloc[idx_ref] > spy_sma.iloc[idx_ref]) if use_market_timing else True

                # Rotation
                if (i - valid_start_idx) % holding_period == 0:
                    scores = momentum.iloc[i].dropna().sort_values(ascending=False)
                    new_top = scores.index[:n_top].tolist()
                    if is_invested and current_top:
                        num_changes = len([s for s in new_top if s not in current_top])
                        portfolio_changes += num_changes
                        monthly_fees += (num_changes / n_top) * fees_pct
                    current_top = new_top
                    
                    pos_history.append({
                        'Date': dt_now.strftime('%d/%m/%Y'),
                        'État': "INVESTI" if market_is_bull else "CASH",
                        'Secteurs': ", ".join(current_top) if market_is_bull else "---"
                    })

                if market_is_bull and not is_invested:
                    is_invested, monthly_fees = True, monthly_fees + fees_pct
                    portfolio_changes += len(current_top)
                elif not market_is_bull and is_invested:
                    is_invested, monthly_fees = False, monthly_fees + fees_pct
                    portfolio_changes += len(current_top)

                d_start, d_end = monthly_close.index[i] + pd.Timedelta(days=1), monthly_close.index[i+1]
                try:
                    idx_s = open_data.index.get_indexer([d_start], method='bfill')[0]
                    idx_e = close_data.index.get_indexer([d_end], method='ffill')[0]
                    
                    strat_ret = sum((close_data[t].iloc[idx_e] / open_data[t].iloc[idx_s]) - 1 for t in current_top) / n_top if is_invested else 0.0
                    bench_ret = (close_data['^GSPC'].iloc[idx_e] / open_data['^GSPC'].iloc[idx_s]) - 1
                    
                    history.append({
                        'Date': monthly_close.index[i+1], 
                        'Ma Stratégie': strat_ret - monthly_fees, 
                        'S&P 500': bench_ret
                    })
                except: continue

        df = pd.DataFrame(history).set_index('Date')
        m_s, m_b = calculate_metrics(df['Ma Stratégie']), calculate_metrics(df['S&P 500'])

        # --- DASHBOARD ---
        st.subheader("📊 Métriques de Performance")
        
        for label, m, trades in [("🔹 Ma Stratégie", m_s, portfolio_changes), ("🔸 S&P 500 (^GSPC)", m_b, None)]:
            st.markdown(f"**{label}**")
            cols = st.columns(6)
            cols[0].metric("Perf. Totale", f"{m[4]*100:.1f}%")
            cols[1].metric("CAGR", f"{m[0]*100:.2f}%")
            cols[2].metric("Sharpe", f"{m[2]:.2f}")
            cols[3].metric("Max DD", f"{m[3]*100:.1f}%")
            cols[4].metric("Volatilité", f"{m[1]*100:.1f}%")
            if trades is not None: cols[5].metric("Nb Trades", trades)
            st.write("")

        st.divider()
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("📈 Performance Cumulée")
            st.line_chart((1 + df[['Ma Stratégie', 'S&P 500']]).cumprod() * 100, color=["#0077b6", "#f39c12"])
        with g2:
            st.subheader("📉 Risque : Drawdown (%)")
            dd_strat = ((1 + df['Ma Stratégie']).cumprod() / (1 + df['Ma Stratégie']).cumprod().cummax() - 1) * 100
            dd_spy = ((1 + df['S&P 500']).cumprod() / (1 + df['S&P 500']).cumprod().cummax() - 1) * 100
            st.line_chart(pd.DataFrame({'Strat': dd_strat, 'S&P': dd_spy, 'Limit': -20}), color=["#0077b6", "#f39c12", "#e74c3c"])

        st.divider()
        c_t1, c_t2 = st.columns([1, 2])
        with c_t1:
            st.subheader("📅 Détail Annuel")
            annual = df[['Ma Stratégie', 'S&P 500']].groupby(df.index.year).apply(lambda x: (1 + x).prod() - 1)
            st.table(annual.sort_index(ascending=False).style.format("{:.2%}"))
        with c_t2:
            st.subheader("🔍 Historique des Tickers")
            st.dataframe(pd.DataFrame(pos_history).sort_index(ascending=False), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Erreur technique : {e}")

if __name__ == "__main__":
    run_momentum_pure()
