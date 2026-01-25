import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
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
    
    # Remplacement des secteurs par vos 30 tickers
    tickers_list = [
        "NVDA", "GOOGL", "AAPL", "AMZN", "META", "AVGO", "TSLA", "BRK-B", 
        "LLY", "WMT", "JPM", "V", "XOM", "JNJ", "ORCL", "MA", "MU", "COST", 
        "AMD", "PLTR", "NFLX", "ABBV", "GE", "CSCO", "PG", "UNH", "KO", "CAT", "MS", "IBM"
    ]
    
    with st.sidebar:
        st.header("⚙️ Paramètres Stratégie")
        n_top = st.slider("Nombre d'actions à détenir", 1, 10, 5) # Augmenté à 10 max vu le panier de 30
        lookback = st.slider("Look-back Momentum (mois)", 1, 12, 6)
        holding_period = st.slider("Fréquence rotation (mois)", 1, 12, 1) # Par défaut 1 mois pour les actions
        fees_pct = st.slider("Frais par transaction (%)", 0.0, 0.5, 0.1, step=0.01) / 100
        
        st.divider()
        st.header("🛡️ Market Timing")
        use_market_timing = st.checkbox("Activer le filtre de tendance", value=True)
        sma_period = st.slider("Moyenne Mobile S&P 500 (jours)", 50, 250, 200, disabled=not use_market_timing)
        
        st.divider()
        st.header("📅 Période")
        min_date, max_date = date(2010, 1, 1), date(2026, 12, 31)
        start_date = st.date_input("Début", value=date(2018, 1, 1), min_value=min_date, max_value=max_date)
        end_date = st.date_input("Fin", value=max_date, min_value=min_date, max_value=max_date)

 @st.cache_data
    def load_data(s_date, e_date, lb_period, sma_p):
        margin_start = pd.to_datetime(s_date) - pd.DateOffset(days=max(lb_period * 31, sma_p) + 60)
        # On télécharge avec group_by='column' pour s'assurer de la structure
        data = yf.download(tickers_list + ['SPY'], start=margin_start, end=e_date, progress=False)
        
        if data.empty: 
            return pd.DataFrame(), pd.DataFrame(), pd.Series()
        
        # Correction robuste pour extraire les colonnes
        if isinstance(data.columns, pd.MultiIndex):
            # Si Yahoo renvoie un MultiIndex (standard pour plusieurs tickers)
            closes = data['Adj Close'].ffill() if 'Adj Close' in data.level_0 else data['Close'].ffill()
            opens = data['Open'].ffill()
        else:
            # Cas de secours si un seul ticker ou format plat
            closes = data[['Adj Close']].ffill() if 'Adj Close' in data.columns else data[['Close']].ffill()
            opens = data[['Open']].ffill()

        # Sécurité : on s'assure que SPY est présent pour le calcul de la SMA
        if 'SPY' in closes.columns:
            spy_sma = closes['SPY'].rolling(window=sma_p).mean()
        else:
            spy_sma = pd.Series()

        return closes, opens, spy_sma

    try:
        with st.spinner('Calcul des performances historiques sur le Top 30...'):
            close_data, open_data, spy_sma = load_data(start_date, end_date, lookback, sma_period)
            if close_data.empty: return

            monthly_close = close_data.resample('ME').last()
            # Calcul du momentum sur les tickers uniquement
            momentum = monthly_close[tickers_list].pct_change(lookback)
            
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
                
                # Market Timing Logic
                idx_ref = spy_sma.index.get_indexer([dt_now], method='ffill')[0]
                price_spy = close_data['SPY'].iloc[idx_ref]
                val_sma = spy_sma.iloc[idx_ref]
                market_is_bull = (price_spy > val_sma) if use_market_timing else True

                # Rotation de portefeuille
                if (i - valid_start_idx) % holding_period == 0:
                    scores = momentum.iloc[i].dropna().sort_values(ascending=False)
                    new_top = scores.index[:n_top].tolist()
                    
                    if is_invested and current_top:
                        num_changes = len([s for s in new_top if s not in current_top])
                        portfolio_changes += num_changes
                        monthly_fees += (num_changes / n_top) * fees_pct
                    current_top = new_top
                    
                    pos_history.append({
                        'Période': dt_now.strftime('%b %Y'),
                        'État': "INVESTI" if market_is_bull else "CASH (Sécurité)",
                        'Actions': ", ".join(current_top) if market_is_bull else "---"
                    })

                # Gestion Entrée/Sortie Cash
                if market_is_bull and not is_invested:
                    is_invested = True
                    portfolio_changes += len(current_top)
                    monthly_fees += fees_pct
                elif not market_is_bull and is_invested:
                    is_invested = False
                    portfolio_changes += len(current_top)
                    monthly_fees += fees_pct

                # Calcul des rendements
                d_start, d_end = monthly_close.index[i] + pd.Timedelta(days=1), monthly_close.index[i+1]
                try:
                    idx_s = open_data.index.get_indexer([d_start], method='bfill')[0]
                    idx_e = close_data.index.get_indexer([d_end], method='ffill')[0]
                    
                    # Rendement moyen des n actions sélectionnées
                    gross_ret = sum((close_data[t].iloc[idx_e] / open_data[t].iloc[idx_s]) - 1 for t in current_top) / n_top if is_invested else 0.0
                    
                    history.append({
                        'Date': monthly_close.index[i+1], 
                        'Ma Stratégie': gross_ret - monthly_fees, 
                        'S&P 500': (close_data['SPY'].iloc[idx_e] / open_data['SPY'].iloc[idx_s]) - 1
                    })
                except: continue

        df = pd.DataFrame(history).set_index('Date')
        m_s = calculate_metrics(df['Ma Stratégie'])
        m_b = calculate_metrics(df['S&P 500'])

        # --- DASHBOARD ---
        st.subheader("📊 Métriques de Performance (Panier 30 Actions)")
        
        st.markdown("#### 🔹 Ma Stratégie")
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("Perf. Totale", f"{m_s[4]*100:.1f}%")
        s2.metric("CAGR Net", f"{m_s[0]*100:.2f}%")
        s3.metric("Ratio Sharpe", f"{m_s[2]:.2f}")
        s4.metric("Max Drawdown", f"{m_s[3]*100:.1f}%")
        s5.metric("Volatilité", f"{m_s[1]*100:.1f}%")
        s6.metric("Nb Trades", portfolio_changes)

        st.markdown("#### 🔸 S&P 500 (Benchmark)")
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.metric("Perf. Totale", f"{m_b[4]*100:.1f}%")
        b2.metric("CAGR", f"{m_b[0]*100:.2f}%")
        b3.metric("Ratio Sharpe", f"{m_b[2]:.2f}")
        b4.metric("Max Drawdown", f"{m_b[3]*100:.1f}%")
        b5.metric("Volatilité", f"{m_b[1]*100:.1f}%")

        st.divider()

        # --- GRAPHIQUES ---
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("📈 Performance Cumulée")
            st.line_chart((1 + df[['Ma Stratégie', 'S&P 500']]).cumprod() * 100, color=["#0077b6", "#f39c12"])
        with g2:
            st.subheader("📉 Risque : Drawdown (%)")
            dd_strat = ((1 + df['Ma Stratégie']).cumprod() / (1 + df['Ma Stratégie']).cumprod().cummax() - 1) * 100
            dd_spy = ((1 + df['S&P 500']).cumprod() / (1 + df['S&P 500']).cumprod().cummax() - 1) * 100
            st.line_chart(pd.DataFrame({'Ma Stratégie': dd_strat, 'S&P 500': dd_spy, 'Seuil -20%': -20}), color=["#0077b6", "#f39c12", "#e74c3c"])

        # --- TABLES ---
        st.divider()
        col_tab1, col_tab2 = st.columns([1, 2])
        
        with col_tab1:
            st.subheader("📅 Détail Annuel")
            annual = df[['Ma Stratégie', 'S&P 500']].groupby(df.index.year).apply(lambda x: (1 + x).prod() - 1)
            annual['Alpha'] = annual['Ma Stratégie'] - annual['S&P 500']
            st.table(annual.sort_index(ascending=False).style.format("{:.2%}").applymap(lambda x: 'background-color: #2ecc71; color: white' if x > 0 else '', subset=['Alpha']))

        with col_tab2:
            st.subheader("🔍 Historique des Actions investies")
            st.dataframe(pd.DataFrame(pos_history).sort_index(ascending=False), use_container_width=True, hide_index=True)

        # --- SIGNAL ---
        st.divider()
        if is_invested:
            st.success(f"✅ ÉTAT ACTUEL : INVESTI | Actions sélectionnées : {', '.join(current_top)}")
        else:
            st.error(f"🛡️ ÉTAT ACTUEL : CASH (Filtre de tendance actif)")

    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    run_momentum_pure()
