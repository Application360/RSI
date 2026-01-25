import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

# 1. Configuration de la page
st.set_page_config(page_title="Momentum Pro - Univers 150", layout="wide")

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
    st.title("🚀 Momentum Pro : Univers 150 Valeurs")
    st.markdown("Univers dynamique (1960-Aujourd'hui) pour éliminer le biais de survie.")
    
    # Liste des 150 tickers
    universe_150 = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "INTC", "CSCO", "ORCL", "IBM", 
        "HPQ", "TXN", "AMD", "MU", "NFLX", "TSLA", "ADBE", "CRM", "PLTR", "AVGO",
        "SAP", "ASML", "QCOM", "AMAT", "LRCX", "ADI", "PANW", "SNPS", "CDNS", "VRSN", 
        "T", "VZ", "TMUS", "ERIC", "NOK", "CSGP", "NOW", "UBER", "SHOP", "SQ",
        "JPM", "BAC", "GS", "MS", "AXP", "V", "MA", "WFC", "C", "BRK-B", 
        "PYPL", "BLK", "SPGI", "CB", "MMC", "PGR", "MET", "PRU", "AIG", "SCHW",
        "GE", "XOM", "CVX", "CAT", "BA", "MMM", "HON", "LMT", "DE", "F", 
        "GM", "UPS", "FDX", "RTX", "GD", "NOC", "WM", "RSG", "NSC", "UNP", 
        "CSX", "ETN", "PH", "EMR", "ITW", "WMT", "KO", "PEP", "PG", "COST", 
        "TGT", "HD", "MCD", "NKE", "DIS", "SBUX", "LOW", "TJX", "EL", "CL", 
        "KMB", "MO", "PM", "STZ", "MDLZ", "MAR", "HLT", "BKNG", "YUM", "JNJ", 
        "PFE", "LLY", "UNH", "ABBV", "MRK", "AMGN", "CVS", "ISRG", "GILD", "VRTX", 
        "REGN", "ZTS", "SYK", "BSX", "BDX", "HCA", "TMO", "DHR", "A", "COP", 
        "SLB", "EOG", "MPC", "PSX", "VLO", "NEM", "FCX", "APD", "LIN", "SHW", 
        "CTAS", "DD", "DOW", "ALB", "X", "WRE", "AA", "HAL", "BKR", "K", 
        "GIS", "CPB", "ADM", "IP", "WY", "M", "KSS", "JWN"
    ]
    universe_150 = list(set(universe_150))

    with st.sidebar:
        st.header("⚙️ Paramètres")
        n_top = st.slider("Actions à détenir", 1, 30, 10)
        lookback = st.slider("Look-back (mois)", 1, 12, 6)
        holding_period = st.slider("Rotation (mois)", 1, 12, 1)
        fees_pct = st.slider("Frais (%)", 0.0, 0.5, 0.1, step=0.01) / 100
        
        st.divider()
        st.header("🛡️ Market Timing")
        use_market_timing = st.checkbox("Filtre de tendance", value=True)
        sma_period = st.slider("SMA S&P 500 (jours)", 50, 250, 200, disabled=not use_market_timing)
        
        st.divider()
        st.header("📅 Période")
        min_hist = date(1960, 1, 1)
        today = date.today()
        start_date = st.date_input("Début", value=date(1980, 1, 1), min_value=min_hist, max_value=today)
        end_date = st.date_input("Fin", value=today, min_value=min_hist, max_value=today)

    @st.cache_data
    def load_data(s_date, e_date, lb_period, sma_p):
        # Marge de sécurité pour le calcul des indicateurs
        margin_start = pd.to_datetime(s_date) - pd.DateOffset(days=max(lb_period * 31, sma_p) + 100)
        data = yf.download(universe_150 + ['^GSPC'], start=margin_start, end=e_date, progress=False)
        if data.empty: 
            return pd.DataFrame(), pd.DataFrame(), pd.Series()
        
        # Extraction propre des colonnes
        if isinstance(data.columns, pd.MultiIndex):
            closes = data['Adj Close'].ffill() if 'Adj Close' in data.columns.levels[0] else data['Close'].ffill()
            opens = data['Open'].ffill()
        else:
            closes = data[['Adj Close']].ffill() if 'Adj Close' in data.columns else data[['Close']].ffill()
            opens = data[['Open']].ffill()

        spy_sma = closes['^GSPC'].rolling(window=sma_p).mean() if '^GSPC' in closes.columns else pd.Series()
        return closes, opens, spy_sma

    try:
        with st.spinner('Téléchargement des données historiques...'):
            close_data, open_data, spy_sma = load_data(start_date, end_date, lookback, sma_period)
            if close_data.empty: 
                st.warning("Aucune donnée disponible.")
                return

            monthly_close = close_data.resample('ME').last()
            valid_cols = [c for c in universe_150 if c in monthly_close.columns]
            momentum = monthly_close[valid_cols].pct_change(lookback)
            
            history, pos_history = [], []
            is_invested, current_top, portfolio_changes = False, [], 0
            
            start_dt = pd.to_datetime(start_date)
            valid_idx = [i for i, idx in enumerate(monthly_close.index) if idx >= start_dt and i >= lookback]
            
            if not valid_idx:
                st.error("Période trop courte.")
                return

            # Boucle principale du Backtest
            for i in range(valid_idx[0], len(monthly_close) - 1):
                dt_now = monthly_close.index[i]
                monthly_fees = 0.0
                
                idx_ref = spy_sma.index.get_indexer([dt_now], method='ffill')[0]
                market_is_bull = (close_data['^GSPC'].iloc[idx_ref] > spy_sma.iloc[idx_ref]) if use_market_timing else True

                # Gestion de la rotation
                if (i - valid_idx[0]) % holding_period == 0:
                    available_scores = momentum.iloc[i].dropna().sort_values(ascending=False)
                    new_top = available_scores.index[:n_top].tolist()
                    
                    if is_invested and current_top:
                        to_sell = [s for s in current_top if s not in new_top]
                        to_buy = [s for s in new_top if s not in current_top]
                        num_transac = len(to_sell) + len(to_buy)
                        portfolio_changes += num_transac
                        monthly_fees += (num_transac / n_top) * fees_pct
                    current_top = new_top
                    
                    pos_history.append({
                        'Période': dt_now.strftime('%Y-%m'),
                        'État': "INVESTI" if market_is_bull and current_top else "CASH",
                        'Actifs dispo.': len(available_scores),
                        'Top Tickers': ", ".join(current_top) if market_is_bull and current_top else "---"
                    })

                # Gestion entrée/sortie de marché
                was_invested = is_invested
                is_invested = market_is_bull and len(current_top) > 0

                if is_invested != was_invested:
                    portfolio_changes += len(current_top)
                    monthly_fees += fees_pct

                # Calcul des rendements mensuels
                d_start, d_end = monthly_close.index[i] + pd.Timedelta(days=1), monthly_close.index[i+1]
                try:
                    idx_s = open_data.index.get_indexer([d_start], method='bfill')[0]
                    idx_e = close_data.index.get_indexer([d_end], method='ffill')[0]
                    
                    if is_invested:
                        raw_ret = sum((close_data[t].iloc[idx_e] / open_data[t].iloc[idx_s]) - 1 for t in current_top) / len(current_top)
                        ret_strat = raw_ret - monthly_fees
                    else:
                        ret_strat = -monthly_fees
                        
                    ret_bench = (close_data['^GSPC'].iloc[idx_e] / open_data['^GSPC'].iloc[idx_s]) - 1
                    history.append({'Date': monthly_close.index[i+1], 'Ma Stratégie': ret_strat, 'S&P 500': ret_bench})
                except:
                    continue

        # Résultats
        df = pd.DataFrame(history).set_index('Date')
        m_s = calculate_metrics(df['Ma Stratégie'], portfolio_changes)
        m_b = calculate_metrics(df['S&P 500'])

        st.subheader("🏁 Performance")
        st.table(pd.DataFrame([m_s, m_b
