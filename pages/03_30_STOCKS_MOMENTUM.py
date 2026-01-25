import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

# 1. Configuration de la page
st.set_page_config(page_title="Momentum Analytics Pro - Historical Top 30", layout="wide")

def calculate_metrics(returns, portfolio_changes=None):
    if returns.empty:
        return {}
    total_return = (returns + 1).prod() - 1
    days = (returns.index[-1] - returns.index[0]).days
    n_years = max(days / 365.25, 0.1)
    cagr = (total_return + 1) ** (1 / n_years) - 1
    vol = returns.std() * np.sqrt(12)
    sharpe = (cagr) / vol if vol > 0 else 0
    cum_rets = (returns + 1).cumprod()
    drawdown = (cum_rets / cum_rets.cummax() - 1).min()
    
    metrics = {
        "Performance Totale": f"{total_return * 100:.2f}%",
        "CAGR (Annuel)": f"{cagr * 100:.2f}%",
        "Max Drawdown": f"{drawdown * 100:.2f}%",
        "Volatilité": f"{vol * 100:.2f}%",
        "Ratio de Sharpe": f"{sharpe:.2f}",
    }
    if portfolio_changes is not None:
        metrics["Nombre de Transactions"] = str(portfolio_changes)
    
    return metrics

def run_momentum_pure():
    st.title("🚀 Momentum Pro : Stratégie Top 30 (Historique & Frais Réels)")
    
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
        st.header("📅 Période Historique")
        min_hist = date(1960, 1, 1)
        start_date = st.date_input("Début", value=date(1990, 1, 1), min_value=min_hist)
        end_date = st.date_input("Fin", value=date.today(), min_value=min_hist)

    @st.cache_data
    def load_data(s_date, e_date, lb_period, sma_p):
        margin_start = pd.to_datetime(s_date) - pd.DateOffset(days=max(lb_period * 31, sma_p) + 100)
        data = yf.download(tickers_list + ['^GSPC'], start=margin_start, end=e_date, progress=False)
        
        if data.empty: return pd.DataFrame(), pd.DataFrame(), pd.Series()
        
        if isinstance(data.columns, pd.MultiIndex):
            closes = data['Adj Close'].ffill() if 'Adj Close' in data.columns.levels[0] else data['Close'].ffill()
            opens = data['Open'].ffill()
        else:
            closes = data[['Adj Close']].ffill() if 'Adj Close' in data.columns else data[['Close']].ffill()
            opens = data[['Open']].ffill()

        spy_sma = closes['^GSPC'].rolling(window=sma_p).mean() if '^GSPC' in closes.columns else pd.Series()
        return closes, opens, spy_sma

    try:
        with st.spinner('Analyse des données et calcul des frais...'):
            close_data, open_data, spy_sma = load_data(start_date, end_date, lookback, sma_period)
            if close_data.empty: return

            monthly_close = close_data.resample('ME').last()
            momentum = monthly_close[tickers_list].pct_change(lookback)
            
            history = []
            pos_history = [] 
            is_invested = False 
            current_top = []
            portfolio_changes = 0
            
            start_dt = pd.to_datetime(start_date)
            valid_idx = [i for i, idx in enumerate(monthly_close.index) if idx >= start_dt and i >= lookback]
            
            if not valid_idx:
                st.error("Données insuffisantes.")
                return

            for i in range(valid_idx[0], len(monthly_close) - 1):
                dt_now = monthly_close.index[i]
                monthly_fees = 0.0 
                
                idx_ref = spy_sma.index.get_indexer([dt_now], method='ffill')[0]
                market_is_bull = (close_data['^GSPC'].iloc[idx_ref] > spy_sma.iloc[idx_ref]) if use_market_timing else True

                # --- Logique de Rotation ---
                if (i - valid_idx[0]) % holding_period == 0:
                    available_scores = momentum.iloc[i].dropna().sort_values(ascending=False)
                    new_top = available_scores.index[:n_top].tolist()
                    
                    if is_invested and current_top:
                        to_sell = [s for s in current_top if s not in new_top]
                        to_buy = [s for s in new_top if s not in current_top]
                        
                        num_transac_rotation = len(to_sell) + len(to_buy)
                        portfolio_changes += num_transac_rotation
                        monthly_fees += (num_transac_rotation / n_top) * fees_pct
                    
                    current_top = new_top
                    pos_history.append({
                        'Période': dt_now.strftime('%Y-%m'), 
                        'État': "INVESTI" if market_is_bull and current_top else "CASH", 
                        'Tickers': ", ".join(current_top) if market_is_bull and current_top else "---"
                    })

                # --- Logique de Market Timing ---
                was_invested = is_invested
                is_invested = market_is_bull and len(current_top) > 0

                if is_invested and not was_invested:
                    portfolio_changes += len(current_top)
                    monthly_fees += fees_pct 
                elif not is_invested and was_invested:
                    portfolio_changes += len(current_top)
                    monthly_fees += fees_pct 

                # --- Calcul des rendements ---
                d_start, d_end = monthly_close.index[i] + pd.Timedelta(days=1), monthly_close.index[i+1]
                try:
                    idx_s = open_data.index.get_indexer([d_start], method='bfill')[0]
                    idx_e = close_data.index.get_indexer([d_end], method='ffill')[0]
                    
                    if is_invested:
                        raw_ret = sum((close_data[t].iloc[idx_e] / open_data[t].iloc[idx_s]) - 1 for t in current_top) / len(current_top)
                        ret_strat = raw_ret - monthly_fees
                    else:
                        ret_strat = 0.0 - monthly_fees
                        
                    ret_bench = (close_data['^GSPC'].iloc[idx_e] / open_data['^GSPC'].iloc[idx_s]) - 1
                    history.append({'Date': monthly_close.index[i+1], 'Ma Stratégie': ret_strat, 'S&P 500': ret_bench})
                except: continue

        df = pd.DataFrame(history).set_index('Date')
        m_s = calculate_metrics(df['Ma Stratégie'], portfolio_changes)
        m_b = calculate_metrics(df['S&P 500'])

        st.subheader("🏁 Performance Comparative (Tableau)")
        st.table(pd.DataFrame([m_s, m_b], index=["Ma Stratégie", "S&P 500 (^GSPC)"]).T)

        st.subheader("📈 Évolution Portefeuille (Échelle Logarithmique)")
        cum_data = (1 + df[['Ma Stratégie', 'S&P 500']]).cumprod() * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_data.index, y=cum_data['Ma Stratégie'], name="Ma Stratégie", line=dict(color='#0077b6', width=2)))
        fig.add_trace(go.Scatter(x=cum_data.index, y=cum_data['S&P 500'], name="S&P 500", line=dict(color='#f39c12', width=1.5, dash='dot')))
        fig.update_layout(yaxis_type="log", template="plotly_white", height=600, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔍 Historique des Tickers")
        st.dataframe(pd.DataFrame(pos_history).sort_index(ascending=False), use_container_width=True)

    except Exception as e:
        st.error(f"Erreur : {e}")

if __name__ == "__main__":
    run_momentum_pure()
