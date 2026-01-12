import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
import bs4 as bs
import requests

# 1. Configuration de la page
st.set_page_config(page_title="S&P 500 Stock Momentum Pro", layout="wide")

@st.cache_data
def get_sp500_tickers():
    """Récupère la liste actuelle des tickers du S&P 500 via Wikipedia"""
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.replace('\n', '')
        # Ajustement pour les tickers contenant des points (BRK.B -> BRK-B)
        ticker = ticker.replace('.', '-')
        tickers.append(ticker)
    return tickers

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
    st.info("Note : Les données antérieures à 1980 peuvent être fragmentaires sur Yahoo Finance pour certains tickers.")

    # --- BARRE LATÉRALE ---
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
        min_date = date(1962, 1, 1) # Date min Yahoo Finance pour bcp de titres
        start_date = st.date_input("Date de début", value=date(1990, 1, 1), min_value=min_date)
        end_date = st.date_input("Date de fin", value=date(2026, 12, 31))

    # --- CHARGEMENT DES DONNÉES ---
    tickers = get_sp500_tickers()
    
    @st.cache_data(show_spinner=False)
    def load_stock_data(tickers, s_date, e_date, lb_period, sma_p):
        margin_start = pd.to_datetime(s_date) - pd.DateOffset(days=max(lb_period * 31, sma_p) + 90)
        
        # Téléchargement par blocs pour éviter les plantages
        data = yf.download(tickers + ['SPY'], start=margin_start, end=e_date, interval="1d", progress=False)
        
        closes = data['Close'].ffill()
        opens = data['Open'].ffill()
        spy_sma = closes['SPY'].rolling(window=sma_p).mean()
        return closes, opens, spy_sma

    try:
        with st.spinner(f'Analyse de {len(tickers)} actions en cours...'):
            close_data, open_data, spy_sma = load_stock_data(tickers, start_date, end_date, lookback, sma_period)
            
            if close_data.empty:
                st.error("Impossible de récupérer les données.")
                return

            # Resampling mensuel
            monthly_close = close_data.resample('ME').last()
            # Momentum basé sur le rendement cumulé
            momentum_df = monthly_close[tickers].pct_change(lookback)
            
            history = []
            current_top = []
            is_invested = False
            
            start_dt = pd.to_datetime(start_date)
            valid_indices = [i for i, dt in enumerate(monthly_close.index) if dt >= start_dt and i >= lookback]
            
            if not valid_indices:
                st.warning("Période trop courte pour les paramètres sélectionnés.")
                return

            # --- BACKTEST CORE ---
            for count, i in enumerate(valid_indices[:-1]):
                monthly_fees = 0
                dt_now = monthly_close.index[i]
                
                # Market Timing
                price_spy = close_data['SPY'].asof(dt_now)
                val_sma = spy_sma.asof(dt_now)
                market_is_bull = (price_spy > val_sma) if use_market_timing else True

                # Rotation
                if count % holding_period == 0:
                    # On ne prend que les actions qui ont des données à cette date
                    scores = momentum_df.iloc[i].dropna().sort_values(ascending=False)
                    new_top = scores.index[:n_top].tolist()
                    
                    if is_invested and current_top:
                        num_changes = len([s for s in new_top if s not in current_top])
                        monthly_fees += (num_changes / n_top) * fees_pct
                    current_top = new_top

                # In/Out Market
                if market_is_bull and not is_invested:
                    is_invested = True
                    monthly_fees += fees_pct
                elif not market_is_bull and is_invested:
                    is_invested = False
                    monthly_fees += fees_pct

                # Calcul du rendement du mois suivant
                d_start = monthly_close.index[i]
                d_end = monthly_close.index[i+1]
                
                try:
                    # Rendement moyen des N actions sélectionnées
                    if is_invested and current_top:
                        # Utilisation des prix d'ouverture pour l'entrée
                        ret_stocks = (close_data[current_top].asof(d_end) / open_data[current_top].asof(d_start)) - 1
                        gross_ret = ret_stocks.mean()
                    else:
                        gross_ret = 0.0
                    
                    history.append({
                        'Date': d_end,
                        'Ma Stratégie': gross_ret - monthly_fees,
                        'S&P 500': (close_data['SPY'].asof(d_end) / open_data['SPY'].asof(d_start)) - 1
                    })
                except: continue

            df = pd.DataFrame(history).set_index('Date')
            
            # --- AFFICHAGE ---
            m_s = calculate_metrics(df['Ma Stratégie'])
            m_b = calculate_metrics(df['S&P 500'])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("CAGR Stratégie", f"{m_s[0]*100:.2f}%", f"{m_s[0]-m_b[0]:.2f}% Alpha")
                st.line_chart((1 + df['Ma Stratégie']).cumprod())
            with col2:
                st.metric("Max Drawdown", f"{m_s[3]*100:.1f}%")
                st.line_chart(df[['Ma Stratégie', 'S&P 500']].rolling(12).std())

            st.subheader("Détail Annuel")
            annual = df.groupby(df.index.year).apply(lambda x: (1 + x).prod() - 1)
            st.dataframe(annual.style.format("{:.2%}").background_gradient(cmap='RdYlGn'))

            st.success(f"Dernière sélection : {', '.join(current_top)}")

    except Exception as e:
        st.error(f"Erreur de données : {e}")
        st.info("Ceci peut arriver si trop de tickers n'ont pas de données pour la période choisie.")

if __name__ == "__main__":
    run_stock_momentum()
