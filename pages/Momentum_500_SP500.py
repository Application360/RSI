import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
import requests

# 1. Configuration de la page
st.set_page_config(page_title="S&P 500 Stock Momentum Pro", layout="wide")

@st.cache_data
def get_sp500_tickers():
    """Récupère la liste des tickers du S&P 500 de manière robuste"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # On ajoute un User-Agent pour éviter d'être bloqué par Wikipédia
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        
        # Utilisation de pandas pour lire le tableau directement (plus fiable)
        tables = pd.read_html(response.text)
        df = tables[0]
        
        tickers = df['Symbol'].tolist()
        # Nettoyage des tickers pour Yahoo Finance (BRK.B -> BRK-B)
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except Exception as e:
        st.error(f"Erreur lors de la récupération des tickers : {e}")
        return []

def calculate_metrics(returns):
    if returns.empty: return 0, 0, 0, 0, 0
    total_return = (returns + 1).prod() - 1
    days = (returns.index[-1] - returns.index[0]).days
    n_years = max(days / 365.25, 0.1)
    cagr = (total_return + 1) ** (1 / n_years) - 1
    vol = returns.std() * np.sqrt(12)
    sharpe = (cagr) / vol if vol > 0 else 0
    cum_rets = (returns + 1).cumprod()
    drawdown = (cum_rets / cum_rets.cummax() - 1).min()
    return cagr, vol, sharpe, drawdown, total_return

def run_stock_momentum():
    st.title("🦅 S&P 500 Stock Momentum : Analyse 1960-2026")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        n_top = st.slider("Nombre d'actions à détenir", 1, 10, 5)
        lookback = st.slider("Look-back Momentum (mois)", 1, 12, 12)
        holding_period = st.slider("Fréquence rotation (mois)", 1, 12, 3)
        fees_pct = st.slider("Frais par transaction (%)", 0.0, 1.0, 0.1) / 100
        
        st.divider()
        st.header("🛡️ Market Timing")
        use_market_timing = st.checkbox("Activer filtre S&P 500 (SMA)", value=True)
        sma_period = st.slider("Moyenne Mobile SPY (jours)", 50, 300, 200)
        
        st.divider()
        st.header("📅 Période")
        min_date = date(1962, 1, 1)
        start_date = st.date_input("Date de début", value=date(2015, 1, 1), min_value=min_date)
        end_date = st.date_input("Date de fin", value=date(2026, 12, 31))

    tickers = get_sp500_tickers()
    if not tickers:
        st.warning("La liste des tickers est vide. Vérifiez votre connexion.")
        return

    @st.cache_data(show_spinner=False)
    def load_stock_data(tickers_list, s_date, e_date, lb_period, sma_p):
        margin_start = pd.to_datetime(s_date) - pd.DateOffset(days=max(lb_period * 31, sma_p) + 90)
        data = yf.download(tickers_list + ['SPY'], start=margin_start, end=e_date, interval="1d", progress=False)
        
        if data.empty or 'Close' not in data:
            return pd.DataFrame(), pd.DataFrame(), pd.Series()
            
        closes = data['Close'].ffill()
        opens = data['Open'].ffill()
        spy_sma = closes['SPY'].rolling(window=sma_p).mean()
        return closes, opens, spy_sma

    try:
        with st.spinner(f'Analyse de {len(tickers)} actions en cours...'):
            close_data, open_data, spy_sma = load_stock_data(tickers, start_date, end_date, lookback, sma_period)
            
            if close_data.empty:
                st.error("Aucune donnée téléchargée.")
                return

            monthly_close = close_data.resample('ME').last()
            momentum_df = monthly_close[tickers].pct_change(lookback)
            
            history = []
            current_top = []
            is_invested = False
            start_dt = pd.to_datetime(start_date)
            
            valid_indices = [i for i, dt in enumerate(monthly_close.index) if dt >= start_dt and i >= lookback]

            for count, i in enumerate(valid_indices[:-1]):
                monthly_fees = 0
                dt_now = monthly_close.index[i]
                
                # Market Timing
                price_spy = close_data['SPY'].asof(dt_now)
                val_sma = spy_sma.asof(dt_now)
                market_is_bull = (price_spy > val_sma) if use_market_timing else True

                # Rotation
                if count % holding_period == 0:
                    scores = momentum_df.iloc[i].dropna().sort_values(ascending=False)
                    new_top = scores.index[:n_top].tolist()
                    if is_invested and current_top:
                        num_changes = len([s for s in new_top if s not in current_top])
                        monthly_fees += (num_changes / n_top) * fees_pct
                    current_top = new_top

                # Signal
                if market_is_bull and not is_invested:
                    is_invested, monthly_fees = True, monthly_fees + fees_pct
                elif not market_is_bull and is_invested:
                    is_invested, monthly_fees = False, monthly_fees + fees_pct

                # Perf
                d_start, d_end = monthly_close.index[i], monthly_close.index[i+1]
                if is_invested and current_top:
                    available_stocks = [s for s in current_top if s in close_data.columns]
                    gross_ret = (close_data[available_stocks].asof(d_end) / open_data[available_stocks].asof(d_start) - 1).mean()
                else:
                    gross_ret = 0.0
                
                history.append({
                    'Date': d_end,
                    'Stratégie': gross_ret - monthly_fees,
                    'S&P 500': (close_data['SPY'].asof(d_end) / open_data['SPY'].asof(d_start)) - 1
                })

            df_res = pd.DataFrame(history).set_index('Date')
            
            # --- Affichage des Résultats ---
            c1, c2 = st.columns(2)
            m_s = calculate_metrics(df_res['Stratégie'])
            m_b = calculate_metrics(df_res['S&P 500'])
            
            with c1:
                st.metric("CAGR Stratégie", f"{m_s[0]*100:.2f}%", f"{m_s[0]-m_b[0]:.2f}% vs Bench")
                st.line_chart((1 + df_res).cumprod())
            
            with c2:
                st.metric("Max Drawdown", f"{m_s[3]*100:.1f}%")
                st.subheader("Dernière Sélection")
                st.write(", ".join(current_top))

    except Exception as e:
        st.error(f"Erreur d'exécution : {e}")

if __name__ == "__main__":
    run_stock_momentum()
