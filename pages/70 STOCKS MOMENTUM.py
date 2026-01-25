import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

# 1. Configuration de la page
st.set_page_config(page_title="Momentum Pro - Univers Dynamique", layout="wide")

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
    st.title("🚀 Momentum Pro : Univers 70+ avec Optimisation Cash & Laggards")
    st.markdown("""
    **Boosters activés :**
    * **Laggard Logic :** Une action n'est vendue que si elle sort du classement Top N.
    * **SHY Cash Management :** Capital placé sur l'ETF SHY (Bons du Trésor 1-3 ans) si le S&P 500 < SMA 200.
    """)
    
    # Univers large pour limiter le biais de survie
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
        start_date = st.date_input("Début", value=date(2005, 1, 1))
        end_date = st.date_input("Fin", value=date.today())

    @st.cache_data
    def load_data(s_date, e_date, sma_p):
        # Marge pour calcul de la SMA et du momentum initial
        margin_start = pd.to_datetime(s_date) - pd.DateOffset(days=sma_p + 150)
        data = yf.download(extended_universe + ['^GSPC', 'SHY'], start=margin_start, end=e_date, progress=False)
        
        if data.empty: return pd.DataFrame(), pd.DataFrame(), pd.Series()
        
        # Gestion du MultiIndex de yfinance
        if isinstance(data.columns, pd.MultiIndex):
            closes = data['Adj Close'].ffill() if 'Adj Close' in data.columns.levels[0] else data['Close'].ffill()
            opens = data['Open'].ffill()
        else:
            closes = data[['Adj Close']].ffill() if 'Adj Close' in data.columns else data[['Close']].ffill()
            opens = data[['Open']].ffill()

        spy_sma = closes['^GSPC'].rolling(window=sma_p).mean()
        return closes, opens, spy_sma

    try:
        with st.spinner('Chargement des données et calcul du SHY...'):
            close_data, open_data, spy_sma = load_data(start_date, end_date, sma_period)
            if close_data.empty: 
                st.error("Échec du téléchargement.")
                return

            # Calcul Momentum Mensuel
            monthly_close = close_data.resample('ME').last()
            momentum = monthly_close[extended_universe].pct_change(lookback)
            
            history = []
            pos_history = [] 
            is_invested = False 
            current_top = []
            portfolio_changes = 0
            
            start_dt = pd.to_datetime(start_date)
            # On commence après la période de lookback nécessaire
            valid_idx = [i for i, idx in enumerate(monthly_close.index) if idx >= start_dt and i >= lookback]
            
            if not valid_idx:
                st.warning("Période trop courte pour les paramètres sélectionnés.")
                return

            for i in range(valid_idx[0], len(monthly_close) - 1):
                dt_now = monthly_close.index[i]
                dt_next = monthly_close.index[i+1]
                monthly_fees = 0.0 
                
                # Check Market Timing (SMA 200)
                idx_ref = spy_sma.index.get_indexer([dt_now], method='ffill')[0]
                market_is_bull = (close_data['^GSPC'].iloc[idx_ref] > spy_sma.iloc[idx_ref]) if use_market_timing else True

                # --- LOGIQUE LAGGARD (Rotation) ---
                if (i - valid_idx[0]) % holding_period == 0:
                    # On ne regarde que les actions qui ont des données
                    valid_universe = momentum.iloc[i].dropna().index.tolist()
                    scores = momentum.iloc[i][valid_universe].sort_values(ascending=False)
                    new_ranking = scores.index[:n_top].tolist()
                    
                    if current_top:
                        # On ne vend que si l'action sort du classement Top N
                        to_sell = [s for s in current_top if s not in new_ranking]
                        to_keep = [s for s in current_top if s in new_ranking]
                        # On remplace les places vides par les meilleurs du classement non possédés
                        needed = n_top - len(to_keep)
                        to_buy = [s for s in new_ranking if s not in to_keep][:needed]
                        
                        final_top = to_keep + to_buy
                        
                        num_transac = len(to_sell) + len(to_buy)
                        portfolio_changes += num_transac
                        monthly_fees += (num_transac / n_top) * fees_pct
                        current_top = final_top
                    else:
                        current_top = new_ranking
                        portfolio_changes += len(current_top)
                        monthly_fees += fees_pct

                    pos_history.append({
                        'Période': dt_now.strftime('%Y-%m'), 
                        'État': "INVESTI" if market_is_bull else "CASH (SHY)", 
                        'Actifs': len(valid_universe),
                        'Holdings': ", ".join(current_top) if market_is_bull else "---"
                    })

                # Gestion des entrées/sorties de marché totales
                was_invested = is_invested
                is_invested = market_is_bull and len(current_top) > 0
                if is_invested != was_invested:
                    portfolio_changes += len(current_top)
                    monthly_fees += fees_pct 

                # Calcul des rendements mensuels
                # On utilise get_indexer pour éviter les erreurs de jours fériés
                idx_s = open_data.index.get_indexer([dt_now + pd.Timedelta(days=1)], method='bfill')[0]
                idx_e = close_data.index.get_indexer([dt_next], method='ffill')[0]
                
                if is_invested:
                    # Performance moyenne des actions du top
                    raw_ret = close_data[current_top].iloc[idx_e] / open_data[current_top].iloc[idx_s] - 1
                    ret_strat = raw_ret.mean() - monthly_fees
                else:
                    # --- BOOST CASH MANAGEMENT (SHY) ---
                    ret_shy = (close_data['SHY'].iloc[idx_e] / open_data['SHY'].iloc[idx_s]) - 1
                    ret_strat = ret_shy - monthly_fees
                
                ret_bench = (close_data['^GSPC'].iloc[idx_e] / open_data['^GSPC'].iloc[idx_s]) - 1
                history.append({'Date': dt_next, 'Ma Stratégie': ret_strat, 'S&P 500': ret_bench})

        # Finalisation des calculs
        df = pd.DataFrame(history).set_index('Date')
        metrics_s = calculate_metrics(df['Ma Stratégie'], portfolio_changes)
        metrics_b = calculate_metrics(df['S&P 500'])

        # AFFICHAGE DES RÉSULTATS
        st.subheader("🏁 Performance Comparative")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.write("**Ma Stratégie (Boostée)**")
            st.table(pd.Series(metrics_s))
        with col_m2:
            st.write("**S&P 500 (^GSPC)**")
            st.table(pd.Series(metrics_b))

        st.subheader("📈 Évolution du Capital (Base 100)")
        cum_data = (1 + df[['Ma Stratégie', 'S&P 500']]).cumprod() * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_data.index, y=cum_data['Ma Stratégie'], name="Stratégie (Laggards + SHY)", line=dict(color='#00d1b2', width=2.5)))
        fig.add_trace(go.Scatter(x=cum_data.index, y=cum_data['S&P 500'], name="S&P 500", line=dict(color='#ff3860', width=1.5, dash='dot')))
        fig.update_layout(yaxis_type="log", template="plotly_white", height=500, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔍 Détails Historiques")
        st.dataframe(pd.DataFrame(pos_history).sort_index(ascending=False), use_container_width=True)

    except Exception as e:
        st.error(f"Une erreur est survenue lors du calcul : {e}")

if __name__ == "__main__":
    run_momentum_pure()
      
