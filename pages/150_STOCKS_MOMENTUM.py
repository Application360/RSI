import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

# Configuration stricte
st.set_page_config(page_title="Momentum 150", layout="wide")

def calculate_metrics(returns, portfolio_changes=None):
    if returns.empty: return {}
    total_return = (returns + 1).prod() - 1
    n_years = max((returns.index[-1] - returns.index[0]).days / 365.25, 0.1)
    cagr = (total_return + 1) ** (1 / n_years) - 1
    vol = returns.std() * np.sqrt(12)
    sharpe = cagr / vol if vol > 0 else 0
    cum_rets = (returns + 1).cumprod()
    dd = (cum_rets / cum_rets.cummax() - 1).min()
    return {
        "Perf": f"{total_return*100:.1f}%", "CAGR": f"{cagr*100:.1f}%",
        "Drawdown": f"{dd*100:.1f}%", "Sharpe": f"{sharpe:.2f}",
        "Transac": str(portfolio_changes)
    }

def main():
    st.title("🚀 Stratégie Momentum 150")

    universe = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "INTC", "CSCO", "ORCL", "IBM", 
        "HPQ", "TXN", "AMD", "MU", "NFLX", "TSLA", "ADBE", "CRM", "PLTR", "AVGO",
        "JPM", "BAC", "GS", "MS", "AXP", "V", "MA", "WFC", "C", "BRK-B", 
        "GE", "XOM", "CVX", "CAT", "BA", "MMM", "HON", "LMT", "DE", "F", 
        "GM", "UPS", "FDX", "RTX", "GD", "NOC", "WM", "RSG", "NSC", "UNP", 
        "WMT", "KO", "PEP", "PG", "COST", "TGT", "HD", "MCD", "NKE", "DIS", 
        "JNJ", "PFE", "LLY", "UNH", "ABBV", "MRK", "AMGN", "CVS", "ISRG", "GILD",
        "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "NEM", "FCX", "APD", "LIN",
        "X", "AA", "HAL", "BKR", "K", "GIS", "CPB", "ADM", "IP", "WY"
    ]

    with st.sidebar:
        n_top = st.slider("Nombre d'actions", 1, 20, 10)
        lookback = st.slider("Lookback (mois)", 1, 12, 6)
        fees = st.slider("Frais (%)", 0.0, 0.5, 0.1) / 100
        start_date = st.date_input("Début", date(1990, 1, 1))

    @st.cache_data
    def get_data(s_date):
        d = yf.download(universe + ["^GSPC"], start=s_date, progress=False)
        c = d['Adj Close'].ffill() if 'Adj Close' in d.columns else d['Close'].ffill()
        o = d['Open'].ffill()
        return c, o

    try:
        close_data, open_data = get_data(start_date)
        m_close = close_data.resample('ME').last()
        mom = m_close[universe].pct_change(lookback)
        
        hist, current_top, changes = [], [], 0
        
        # Simulation simplifiée pour éviter les erreurs d'index
        for i in range(lookback, len(m_close)-1):
            scores = mom.iloc[i].dropna().sort_values(ascending=False)
            new_top = scores.index[:n_top].tolist()
            
            # Calcul frais simplifié
            if current_top:
                mouvements = len([x for x in new_top if x not in current_top]) * 2
                changes += mouvements
                fee_impact = (mouvements / n_top) * fees
            else:
                fee_impact = fees
                changes += n_top
            
            current_top = new_top
            
            # Rendement
            d_s, d_e = m_close.index[i], m_close.index[i+1]
            r_strat = (close_data[current_top].loc[d_e] / open_data[current_top].loc[d_s+pd.Timedelta(days=1)]).mean() - 1 - fee_impact
            r_bench = (close_data["^GSPC"].loc[d_e] / open_data["^GSPC"].loc[d_s+pd.Timedelta(days=1)]) - 1
            
            hist.append({"Date": d_e, "Strat": r_strat, "Bench": r_bench})

        res = pd.DataFrame(hist).set_index("Date")
        st.table(pd.DataFrame([calculate_metrics(res["Strat"], changes), calculate_metrics(res["Bench"])], index=["Strat", "S&P500"]).T)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res.index, y=(1+res["Strat"]).cumprod(), name="Stratégie"))
        fig.add_trace(go.Scatter(x=res.index, y=(1+res["Bench"]).cumprod(), name="S&P 500"))
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Erreur : {e}")

if __name__ == "__main__":
    main()
