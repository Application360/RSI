import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

# 1. Configuration de la page
st.set_page_config(page_title="Momentum Analytics Pro - Top 30 Stocks", layout="wide")

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
    st.title("🚀 Momentum Pro : Analyse Top 30 Stocks vs S&P 500")
    
    tickers_list = [
        "NVDA", "GOOGL", "AAPL", "AMZN", "META", "AVGO", "TSLA", "BRK-B", 
        "LLY", "WMT", "JPM", "V", "XOM", "JNJ", "ORCL", "MA", "MU", "COST", 
        "AMD", "PLTR", "NFLX", "ABBV", "GE", "CSCO", "PG", "UNH", "KO", "CAT", "MS", "IBM"
    ]
    
    with st.sidebar:
        st.header("⚙️ Paramètres Stratégie")
        n_top = st.slider("Nombre d'actions à détenir", 1, 10, 5)
        lookback = st.slider("Look-back Momentum (mois)", 1, 12, 6)
        holding_period = st.slider("Fréquence rotation (mois)", 1, 12, 1)
        fees_pct = st.slider("Frais par transaction (%)", 0.0, 0.5, 0.1, step=0.01) / 100
        
        st.divider()
        st.header("🛡️ Market Timing")
        use_market_timing = st.checkbox("Activer le filtre de tendance", value=True)
        sma_period = st.slider("Moyenne Mobile S&P 500 (jours)", 50, 250, 200, disabled=not use_market_timing)
        
        st.divider()
        st.header("📅 Période")
        start_date = st.date_input("Début", value=date(2018, 1, 1))
        end_date = st.date_input("Fin", value=date.today())

    @st.cache_data
    def load_data(s_date, e_date, lb_period, sma_p):
        margin_start = pd.to_datetime(s_date) - pd.DateOffset(days=max(lb_period * 31, sma_p) + 60)
        data = yf.download(tickers_list + ['SPY'], start=margin_start, end=e_date, progress=False)
        if data.empty: return pd.DataFrame(), pd.DataFrame(), pd.Series()
        
        if isinstance(data.columns, pd.MultiIndex):
            closes = data['Adj Close'].ffill() if 'Adj Close' in data.columns.levels[0] else data['Close'].ffill()
            opens = data['Open'].ffill()
        else:
            closes = data[['Adj Close']].ffill() if 'Adj Close' in data.columns else data[['Close']].ffill()
            opens = data[['Open']].ffill()

        spy_sma = closes['SPY'].rolling(window=sma_p).mean() if 'SPY' in closes.columns else pd.Series()
        return closes, opens, spy_sma

    try:
        with st.spinner('Calcul des performances...'):
            close_data, open_data, spy_sma = load_data(start_date, end_date, lookback, sma_period)
            if close_data.empty: return

            monthly_close = close_data.resample('ME').last()
            momentum = monthly_close[tickers_list].pct_change(lookback)
            
            history = []
            pos_history = [] 
            is_invested = False 
            current_top = []
            
            start_dt = pd.to_datetime(start_date)
            valid_idx = [i for i, index in enumerate(monthly_close.index) if index >= start_dt and i >= lookback]
            if not valid_idx: return
            
            for i in range(valid_idx[0], len(monthly_close) - 1):
                dt_now = monthly_close.index[i]
                idx_ref = spy_sma.index.get_indexer([dt_now], method='ffill')[0]
                market_is_bull = (close_data['SPY'].iloc[idx_ref] > spy_sma.iloc[idx_ref]) if use_market_timing else True

                if (i - valid_idx[0]) % holding_period == 0:
                    current_top = momentum.iloc[i].dropna().sort_values(ascending=False).index[:n_top].tolist()
                    pos_history.append({'Date': dt_now.strftime('%Y-%m'), 'État': "INVESTI" if market_is_bull else "CASH", 'Actions': ", ".join(current_top) if market_is_bull else "---"})

                is_invested = market_is_bull
                d_start, d_end = monthly_close.index[i] + pd.Timedelta(days=1), monthly_close.index[i+1]
                
                try:
                    idx_s = open_data.index.get_indexer([d_start], method='bfill')[0]
                    idx_e = close_data.index.get_indexer([d_end], method='ffill')[0]
                    ret_strat = sum((close_data[t].iloc[idx_e] / open_data[t].iloc[idx_s]) - 1 for t in current_top) / n_top if is_invested else 0.0
                    ret_bench = (close_data['SPY'].iloc[idx_e] / open_data['SPY'].iloc[idx_s]) - 1
                    history.append({'Date': monthly_close.index[i+1], 'Ma Stratégie': ret_strat, 'S&P 500': ret_bench})
                except: continue

        df = pd.DataFrame(history).set_index('Date')
        m_s = calculate_metrics(df['Ma Stratégie'])
        m_b = calculate_metrics(df['S&P 500'])

        # --- Dashboard ---
        st.subheader("📊 Métriques de Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAGR Strat", f"{m_s[0]*100:.1f}%", f"{ (m_s[0]-m_b[0])*100:.1f}% Alpha")
        c2.metric("CAGR S&P 500", f"{m_b[0]*100:.1f}%")
        c3.metric("Sharpe Strat", f"{m_s[2]:.2f}")
        c4.metric("Max DD Strat", f"{m_s[3]*100:.1f}%")

        # --- Graphique Échelle Log ---
        st.subheader("📈 Évolution Comparative (Échelle Logarithmique)")
        cum_data = (1 + df[['Ma Stratégie', 'S&P 500']]).cumprod() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_data.index, y=cum_data['Ma Stratégie'], name="Ma Stratégie", line=dict(color='#0077b6', width=3)))
        fig.add_trace(go.Scatter(x=cum_data.index, y=cum_data['S&P 500'], name="S&P 500", line=dict(color='#f39c12', width=2, dash='dot')))
        
        fig.update_layout(
            yaxis_type="log",
            yaxis_title="Valeur du portefeuille (Base 100)",
            xaxis_title="Date",
            template="plotly_white",
            height=500,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔍 Dernières Positions")
        st.dataframe(pd.DataFrame(pos_history).tail(12), use_container_width=True)

    except Exception as e:
        st.error(f"Erreur : {e}")

if __name__ == "__main__":
    run_momentum_pure()
