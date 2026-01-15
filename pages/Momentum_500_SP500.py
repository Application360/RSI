import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# --- CONFIGURATION ---
st.set_page_config(page_title="Momentum 500", layout="wide")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Paramètres")
num_assets = st.sidebar.slider("Nombre d'actions", 5, 50, 20)
lookback_months = st.sidebar.slider("Look-back (mois)", 1, 12, 6)
rotation_freq = st.sidebar.slider("Rotation (mois)", 1, 12, 3)
fees_pct = st.sidebar.slider("Frais (%)", 0.0, 0.5, 0.1, 0.01) / 100

st.sidebar.markdown("---")
enable_filter = st.sidebar.checkbox("Filtre de tendance MM", value=True)
ma_window = st.sidebar.slider("MM (jours)", 50, 250, 150)

st.sidebar.markdown("---")
start_input = st.sidebar.text_input("Début", "2018/01/01") # Période plus courte pour tester
end_input = st.sidebar.text_input("Fin", "2026/12/31")

st.title("🚀 Momentum 500 Actions SP500")

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        df = pd.read_html(response.text)[0]
        return df['Symbol'].str.replace('.', '-', regex=True).tolist()
    except:
        # Liste de secours si Wikipédia bloque
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B", "JNJ", "V"]

@st.cache_data(show_spinner=False)
def download_data(tickers, start, end):
    # Téléchargement de l'indice
    mkt = yf.download("^GSPC", start=start, end=end, interval="1mo", progress=False)['Close']
    if isinstance(mkt, pd.DataFrame): mkt = mkt.iloc[:, 0]
    
    # Téléchargement des actions (on limite pour la stabilité)
    # yfinance gère mieux les listes propres
    data = yf.download(tickers, start=start, end=end, interval="1mo", progress=False)['Close']
    return data, mkt

# --- EXECUTION ---
try:
    tickers = get_sp500_tickers()
    s_date = start_input.replace('/', '-')
    e_date = end_input.replace('/', '-')

    with st.spinner("Chargement des données financières..."):
        all_data, mkt_price = download_data(tickers, s_date, e_date)

    if all_data is not None and not all_data.empty:
        # 1. Calculs
        momentum = all_data.pct_change(lookback_months)
        monthly_rets = all_data.pct_change()
        
        mkt_ma = mkt_price.rolling(window=max(2, ma_window // 21)).mean()
        trend_signal = (mkt_price > mkt_ma).shift(1) if enable_filter else pd.Series(True, index=mkt_price.index)
        
        strat_returns = pd.Series(0.0, index=monthly_rets.index)
        current_portfolio = []
        trade_count = 0
        
        # 2. Simulation
        for i in range(len(monthly_rets)):
            if i % rotation_freq == 0:
                new_portfolio = []
                if trend_signal.iloc[i]:
                    top = momentum.iloc[i].dropna().nlargest(num_assets)
                    new_portfolio = top.index.tolist()
                
                if set(new_portfolio) != set(current_portfolio):
                    if i > 0: trade_count += 1
                    strat_returns.iloc[i] -= fees_pct
                    current_portfolio = new_portfolio
            
            if current_portfolio:
                strat_returns.iloc[i] += monthly_returns[current_portfolio].iloc[i].mean()

        # 3. Métriques
        results = pd.DataFrame(index=monthly_rets.index)
        results['mkt_cum'] = (1 + mkt_price.pct_change().fillna(0)).cumprod()
        results['strat_cum'] = (1 + strat_returns.fillna(0)).cumprod()
        
        # Calcul KPI
        def calc_kpi(cum_series, ret_series):
            yrs = max(0.1, (cum_series.index[-1] - cum_series.index[0]).days / 365.25)
            cagr = (cum_series.iloc[-1] ** (1/yrs) - 1) * 100
            mdd = ((cum_series - cum_series.cummax()) / cum_series.cummax()).min() * 100
            vol = ret_series.std() * np.sqrt(12) * 100
            sharpe = (ret_series.mean() * 12) / (ret_series.std() * np.sqrt(12)) if ret_series.std() != 0 else 0
            return cagr, mdd, sharpe, vol

        c_s, m_s, sh_s, v_s = calc_kpi(results['strat_cum'], strat_returns)
        c_m, m_m, sh_m, v_m = calc_kpi(results['mkt_cum'], mkt_price.pct_change())

        # --- AFFICHAGE ---
        st.subheader("📊 Comparaison Performances")
        c1, c2 = st.columns(2)
        
        with c1:
            st.info(f"**Ma Stratégie ({num_assets} actions)**")
            st.metric("CAGR Net", f"{c_s:.2f}%")
            st.metric("Max Drawdown", f"{m_s:.1f}%")
            st.metric("Trades", trade_count)
            
        with c2:
            st.warning("**S&P 500 (^GSPC)**")
            st.metric("CAGR", f"{c_m:.2f}%")
            st.metric("Max Drawdown", f"{m_m:.1f}%")
            st.metric("Sharpe", f"{sh_m:.2f}")

        st.plotly_chart(go.Figure(data=[
            go.Scatter(x=results.index, y=results['strat_cum'], name="Stratégie", line=dict(color='#3366CC')),
            go.Scatter(x=results.index, y=results['mkt_cum'], name="S&P 500", line=dict(color='#FF6633', dash='dot'))
        ]).update_layout(template="plotly_white", yaxis_type="log"), use_container_width=True)

    else:
        st.warning("⚠️ Les données n'ont pas pu être téléchargées. Essayez de rafraîchir la page ou de réduire la période.")

except Exception as e:
    st.error(f"Erreur : {e}")
