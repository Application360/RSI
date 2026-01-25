import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, datetime

# 1. Configuration de la page
st.set_page_config(page_title="Momentum Pro - 1960 Edition", layout="wide")

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
    st.title("🚀 Momentum Pro : Analyse Long-Terme (1960 - Présent)")
    
    # Univers large
    extended_universe = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "INTC", "CSCO", "ORCL", "IBM", 
        "HPQ", "TXN", "AMD", "MU", "NFLX", "TSLA", "ADBE", "CRM", "PLTR", "AVGO", "APP",
        "JPM", "BAC", "GS", "MS", "AXP", "V", "MA", "WFC", "C", "BRK-B",
        "GE", "XOM", "CVX", "CAT", "BA", "MMM", "HON", "LMT", "DE", "F", "GM",
        "WMT", "KO", "PEP", "PG", "JNJ", "PFE", "LLY", "UNH", "ABBV", "MRK", "AMGN", 
        "COST", "TGT", "HD", "MCD", "NKE", "DIS", "PM", "MO", "NEM",
        "T", "VZ", "UPS", "FDX", "SBUX", "LOW", "ABT", "LRCX", "QCOM", "PGR"
    ]
    extended_universe = list(set(extended_universe))

    with st.sidebar:
        st.header("⚙️ Paramètres Stratégie")
        n_top = st.slider("Nombre d'actions à détenir", 1, 15, 5)
        lookback = st.slider("Look-back Momentum (mois)", 1, 12, 6)
        holding_period = st.slider("Fréquence rotation (mois)", 1, 12, 1)
        fees_pct = st.slider("Frais par transaction (%)", 0.0, 0.5, 0.1, step=0.01) / 100
        
        st.divider()
        st.header("🛡️ Market Timing")
        use_market_timing = st.checkbox("Activer le filtre de tendance", value=True)
        sma_period = st.slider("Moyenne Mobile S&P 500 (jours)", 50, 250, 200)
        
        st.divider()
        st.header("📅 Période Historique")
        # --- CORRECTION DU CURSEUR ---
        # On définit explicitement les limites pour forcer l'affichage de 1960
        min_date = date(1960, 1, 1)
        max_date = date.today()
        
        start_date = st.date_input("Date de début", value=min_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("Date de fin", value=max_date, min_value=min_date, max_value=max_date)

    @st.cache_data
    def load_data(s_date, e_date, sma_p):
        # Conversion pour yfinance
        margin_start = pd.to_datetime(s_date) - pd.DateOffset(days=sma_p + 180)
        data = yf.download(extended_universe + ['^GSPC', 'SHY'], start=margin_start, end=e_date, progress=False)
        
        if data.empty: return pd.DataFrame(), pd.DataFrame(), pd.Series()
        
        # Sélection robuste des prix de clôture ajustés
        if 'Adj Close' in data.columns:
            closes = data['Adj Close'].ffill()
        else:
            closes = data['Close'].ffill()
            
        opens = data['Open'].ffill()
        spy_sma = closes['^GSPC'].rolling(window=sma_p).mean()
        
        return closes, opens, spy_sma

    try:
        if start_date >= end_date:
            st.error("La date de début doit être antérieure à la date de fin.")
            return

        with st.spinner('Extraction des données (cela peut prendre du temps pour 60 ans)...'):
            close_data, open_data, spy_sma = load_data(start_date, end_date, sma_period)
            
            if close_data.empty:
                st.error("Aucune donnée récupérée.")
                return

            # Resampling mensuel
            monthly_close = close_data.resample('ME').last()
            momentum = monthly_close[extended_universe].pct_change(lookback)
            
            history = []
            pos_history = [] 
            current_top = []
            portfolio_changes = 0
            
            # Aligner les dates
            start_dt = pd.to_datetime(start_date)
            valid_indices = monthly_close.index[monthly_close.index >= start_dt]
            
            if len(valid_indices) < lookback:
                st.warning("Plage de dates trop courte pour les paramètres sélectionnés.")
                return

            for i in range(len(monthly_close) - 1):
                dt_now = monthly_close.index[i]
                if dt_now < start_dt: continue
                
                dt_next = monthly_close.index[i+1]
                monthly_fees = 0.0 
                
                # Vérification Trend S&P 500
                idx_ref = close_data.index.get_indexer([dt_now], method='pad')[0]
                market_is_bull = (close_data['^GSPC'].iloc[idx_ref] > spy_sma.iloc[idx_ref]) if use_market_timing else True

                # Rotation Logique
                if i % holding_period == 0:
                    # On ne regarde que les actions cotées à cette date précise
                    present_tickers = close_data.iloc[idx_ref][extended_universe].dropna().index.tolist()
                    if present_tickers:
                        # Calcul Momentum sur les tickers existants
                        valid_mom = momentum.loc[dt_now, present_tickers].dropna()
                        new_ranking = valid_mom.sort_values(ascending=False).head(n_top).index.tolist()
                        
                        if current_top:
                            to_sell = [s for s in current_top if s not in new_ranking]
                            to_buy = [s for s in new_ranking if s not in current_top]
                            current_top = [s for s in current_top if s in new_ranking] + to_buy[:n_top-len([s for s in current_top if s in new_ranking])]
                            
                            change_count = len(to_sell) + len(to_buy)
                            portfolio_changes += change_count
                            monthly_fees += (change_count / n_top) * fees_pct
                        else:
                            current_top = new_ranking
                            portfolio_changes += len(current_top)
                            monthly_fees += fees_pct

                    pos_history.append({
                        'Date': dt_now.strftime('%Y-%m'),
                        'Status': "BOULLIER" if market_is_bull else "PRUDENCE (CASH)",
                        'Holdings': ", ".join(current_top) if market_is_bull else "SHY / CASH"
                    })

                # Calcul performance du mois
                idx_start_month = open_data.index.get_indexer([dt_now], method='bfill')[0]
                idx_end_month = close_data.index.get_indexer([dt_next], method='ffill')[0]
                
                if market_is_bull and current_top:
                    # Performance moyenne des actions détenues
                    month_rets = (close_data[current_top].iloc[idx_end_month] / open_data[current_top].iloc[idx_start_month]) - 1
                    ret_strat = month_rets.mean() - monthly_fees
                else:
                    # Performance SHY (ou 0 si SHY n'existe pas en 1960)
                    shy_val = (close_data['SHY'].iloc[idx_end_month] / open_data['SHY'].iloc[idx_start_month]) - 1
                    ret_strat = (shy_val if not np.isnan(shy_val) else 0.0) - monthly_fees
                
                ret_bench = (close_data['^GSPC'].iloc[idx_end_month] / open_data['^GSPC'].iloc[idx_start_month]) - 1
                history.append({'Date': dt_next, 'Stratégie': ret_strat, 'S&P 500': ret_bench})

        # --- RESTITUTION ---
        results_df = pd.DataFrame(history).set_index('Date')
        
        st.subheader("📊 Comparaison Historique (Base 100)")
        cum_rets = (1 + results_df).cumprod() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_rets.index, y=cum_rets['Stratégie'], name="Stratégie Momentum", line=dict(color='#00d1b2', width=2)))
        fig.add_trace(go.Scatter(x=cum_rets.index, y=cum_rets['S&P 500'], name="S&P 500", line=dict(color='#ff3860', dash='dot')))
        fig.update_layout(yaxis_type="log", template="plotly_white", height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.table(pd.DataFrame([
            calculate_metrics(results_df['Stratégie'], portfolio_changes),
            calculate_metrics(results_df['S&P 500'])
        ], index=["Ma Stratégie", "Benchmark S&P 500"]).T)

    except Exception as e:
        st.error(f"Erreur lors du calcul : {str(e)}")
        st.info("Note : Certaines actions de l'univers n'étaient pas cotées en 1960. Le script s'adapte automatiquement.")

if __name__ == "__main__":
    run_momentum_pure()
