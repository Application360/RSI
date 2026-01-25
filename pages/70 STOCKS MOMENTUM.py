import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

# Configuration
st.set_page_config(page_title="Momentum Pro - Stable", layout="wide")

def calculate_metrics(returns, portfolio_changes=None):
    if returns.empty: return {}
    total_return = (returns + 1).prod() - 1
    days = (returns.index[-1] - returns.index[0]).days
    n_years = max(days / 365.25, 0.1)
    cagr = (total_return + 1) ** (1 / n_years) - 1
    vol = returns.std() * np.sqrt(12)
    sharpe = cagr / vol if vol > 0 else 0
    dd = ((returns + 1).cumprod() / (returns + 1).cumprod().cummax() - 1).min()
    return {
        "Perf. Totale": f"{total_return * 100:.1f}%",
        "CAGR": f"{cagr * 100:.1f}%",
        "Drawdown": f"{dd * 100:.1f}%",
        "Sharpe": f"{sharpe:.2f}",
        "Transactions": str(portfolio_changes)
    }

def main():
    st.title("🚀 Momentum Pro : Univers Anti-Biais")
    st.markdown("Inclusion de **Newmont (NEM)** et **AppLovin (APP)**.")

    universe = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "INTC", "CSCO", "ORCL", "IBM", 
        "HPQ", "TXN", "AMD", "MU", "NFLX", "TSLA", "ADBE", "CRM", "PLTR", "AVGO", "APP",
        "JPM", "BAC", "GS", "MS", "AXP", "V", "MA", "WFC", "C", "BRK-B",
        "GE", "XOM", "CVX", "CAT", "BA", "MMM", "HON", "LMT", "DE", "F", "GM",
        "WMT", "KO", "PEP", "PG", "JNJ", "PFE", "LLY", "UNH", "ABBV", "MRK", "AMGN", 
        "COST", "TGT", "HD", "MCD", "NKE", "DIS", "PM", "MO", "NEM",
        "T", "VZ", "UPS", "FDX", "SBUX", "LOW"
    ]

    with st.sidebar:
        st.header("⚙️ Paramètres")
        n_top = st.slider("Nombre d'actions", 1, 15, 5)
        lookback = st.slider("Look-back (mois)", 1, 12, 6)
        holding_period = st.slider("Rotation (mois)", 1, 12, 1)
        fees_pct = st.slider("Frais (%)", 0.0, 0.5, 0.1, step=0.01) / 100
        use_market_timing = st.checkbox("Filtre SMA 200 S&P 500", value=True)
        start_date = st.date_input("Début", date(1995, 1, 1), min_value=date(1960, 1, 1))

    @st.cache_data
    def load_data(s_date):
        margin = pd.to_datetime(s_date) - pd.DateOffset(days=400)
        data = yf.download(universe + ['^GSPC'], start=margin, progress=False)
        # Gestion multi-index Yahoo Finance
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
            open_p = data['Open']
        else:
            close = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
            open_p = data['Open']
        return close.ffill(), open_p.ffill()

    try:
        with st.spinner('Calculs en cours...'):
            close_data, open_data = load_data(start_date)
            spy_sma = close_data['^GSPC'].rolling(window=200).mean()
            monthly_close = close_data.resample('ME').last()
            momentum = monthly_close[universe].pct_change(lookback)
            
            history, pos_history = [], []
            is_invested, current_top, portfolio_changes = False, [], 0
            start_dt = pd.to_datetime(start_date)
            valid_months = [m for m in monthly_close.index if m >= start_dt]

            for i in range(len(valid_months) - 1):
                dt_now = valid_months[i]
                dt_next = valid_months[i+1]
                monthly_fees = 0.0

                # Market Timing avec protection contre les dates manquantes
                idx_now = close_data.index.get_indexer([dt_now], method='ffill')[0]
                market_is_bull = (close_data['^GSPC'].iloc[idx_now] > spy_sma.iloc[idx_now]) if use_market_timing else True

                # Rotation
                if i % holding_period == 0:
                    available = momentum.loc[dt_now].dropna().sort_values(ascending=False)
                    new_top = available.index[:n_top].tolist()
                    if is_invested and current_top:
                        mvt = len([s for s in new_top if s not in current_top]) + len([s for s in current_top if s not in new_top])
                        portfolio_changes += mvt
                        monthly_fees += (mvt / n_top) * fees_pct
                    current_top = new_top
                    pos_history.append({'Date': dt_now.strftime('%Y-%m'), 'Etat': "INVESTI" if market_is_bull else "CASH", 'Top': ", ".join(current_top) if market_is_bull else "---"})

                was_invested = is_invested
                is_invested = market_is_bull and len(current_top) > 0
                if is_invested != was_invested:
                    portfolio_changes += len(current_top)
                    monthly_fees += fees_pct

                # Calcul rendement avec recherche de l'index le plus proche (évite l'erreur Timestamp)
                idx_start = open_data.index.get_indexer([dt_now], method='bfill')[0]
                idx_end = close_data.index.get_indexer([dt_next], method='ffill')[0]
                
                if is_invested:
                    rets = (close_data[current_top].iloc[idx_end] / open_data[current_top].iloc[idx_start]) - 1
                    ret_strat = rets.mean() - monthly_fees
                else:
                    ret_strat = 0.0 - monthly_fees
                
                ret_bench = (close_data['^GSPC'].iloc[idx_end] / open_data['^GSPC'].iloc[idx_start]) - 1
                history.append({'Date': dt_next, 'Strat': ret_strat, 'Bench': ret_bench})

            df_res = pd.DataFrame(history).set_index('Date')
            
            # Affichage des résultats
            st.table(pd.DataFrame([calculate_metrics(df_res['Strat'], portfolio_changes), calculate_metrics(df_res['Bench'])], index=["Stratégie", "S&P 500"]).T)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_res.index, y=(1+df_res['Strat']).cumprod(), name="Stratégie"))
            fig.add_trace(go.Scatter(x=df_res.index, y=(1+df_res['Bench']).cumprod(), name="S&P 500"))
            fig.update_layout(yaxis_type="log", template="plotly_white", height=500)
            st.plotly_chart(fig, use_container_width=True)
