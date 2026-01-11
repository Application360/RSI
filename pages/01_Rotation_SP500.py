import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Momentum Pro SaaS", layout="wide")

def calculate_metrics(returns):
    """Calcule les métriques clés de performance"""
    total_return = (returns + 1).prod() - 1
    n_years = len(returns) / 12
    cagr = (total_return + 1) ** (1 / n_years) - 1
    vol = returns.std() * np.sqrt(12)
    sharpe = (cagr - 0.02) / vol if vol > 0 else 0
    cum_rets = (returns + 1).cumprod()
    rolling_max = cum_rets.cummax()
    drawdown = (cum_rets / rolling_max - 1).min()
    return cagr, vol, sharpe, drawdown, total_return

def run_momentum_pro():
    st.title("🛡️ Momentum Pro : Stratégie & Analytics")
    
    sectors = ['XLK', 'XLF', 'XLV', 'XLY', 'XLI', 'XLP', 'XLE', 'XLC', 'XLB', 'XLU', 'XLRE']
    
    @st.cache_data
    def load_data():
        # On télécharge avec une marge pour le lookback
        data = yf.download(sectors + ['SPY'], start="1998-10-01")
        # Nettoyage MultiIndex si présent
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data['Close']
            open_prices = data['Open']
        else:
            close_prices = data['Close']
            open_prices = data['Open']
        return close_prices.ffill(), open_prices.ffill()

    try:
        close_data, open_data = load_data()
        
        # Resample mensuel
        monthly_close = close_data.resample('ME').last()
        momentum = monthly_close[sectors].pct_change(3)
        
        history = []
        current_portfolio = {} 
        
        # On itère sur les mois
        for i in range(3, len(monthly_close) - 1):
            # 1. Calcul des signaux à la clôture du mois i
            scores = momentum.iloc[i].sort_values(ascending=False)
            top_2 = scores.index[:2].tolist()
            top_4 = scores.index[:4].tolist()
            
            # 2. Application BUFFER & PYRAMID
            new_portfolio = {}
            for ticker, lots in current_portfolio.items():
                if ticker in top_4:
                    new_portfolio[ticker] = lots
                    if ticker in top_2 and lots == 1:
                        new_portfolio[ticker] = 2
            
            for ticker in top_2:
                if ticker not in new_portfolio:
                    new_portfolio[ticker] = 1
            
            # 3. Calcul de la performance du mois i+1
            # On cherche l'indice de la première bougie du mois suivant
            next_month_start = monthly_close.index[i] + pd.Timedelta(days=1)
            next_month_end = monthly_close.index[i+1]
            
            # Sécurité pour trouver les prix réels les plus proches (évite l'erreur Timestamp)
            idx_start = open_data.index.get_indexer([next_month_start], method='bfill')[0]
            idx_end = close_data.index.get_indexer([next_month_end], method='ffill')[0]
            
            month_return = 0
            num_lots_total = sum(new_portfolio.values())
            
            if num_lots_total > 0:
                for ticker, lots in new_portfolio.items():
                    p_open = open_data[ticker].iloc[idx_start]
                    p_close = close_data[ticker].iloc[idx_end]
                    ret = (p_close / p_open) - 1
                    month_return += (ret * (lots / num_lots_total))
            
            # Benchmark SPY
            spy_open = open_data['SPY'].iloc[idx_start]
            spy_close = close_data['SPY'].iloc[idx_end]
            
            history.append({
                'Date': monthly_close.index[i+1],
                'Strat_Ret': month_return,
                'SPY_Ret': (spy_close / spy_open) - 1
            })
            current_portfolio = new_portfolio

        df_res = pd.DataFrame(history).set_index('Date')
        
        # --- METRIQUES & AFFICHAGE ---
        m_strat = calculate_metrics(df_res['Strat_Ret'])
        m_spy = calculate_metrics(df_res['SPY_Ret'])

        st.subheader("📊 Comparatif de Performance (Backtest Historique)")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAGR Stratégie", f"{m_strat[0]*100:.2f}%", f"{(m_strat[0]-m_spy[0])*100:.2f}% vs Index")
        c2.metric("Sharpe Ratio", f"{m_strat[2]:.2f}")
        c3.metric("Max Drawdown", f"{m_strat[3]*100:.2f}%", delta_color="inverse")
        c4.metric("Performance Totale", f"{m_strat[4]*100:.1f}%")

        # Table
        st.table(pd.DataFrame({
            'Métrique': ['Total Return', 'CAGR', 'Sharpe', 'Max Drawdown'],
            'Stratégie': [f"{m_strat[4]*100:.1f}%", f"{m_strat[0]*100:.2f}%", f"{m_strat[2]:.2f}", f"{m_strat[3]*100:.2f}%"],
            'S&P 500': [f"{m_spy[4]*100:.1f}%", f"{m_spy[0]*100:.2f}%", f"{m_spy[2]:.2f}", f"{m_spy[3]*100:.2f}%"]
        }))

        st.line_chart(pd.DataFrame({
            'Stratégie': (1 + df_res['Strat_Ret']).cumprod() * 100,
            'S&P 500': (1 + df_res['SPY_Ret']).cumprod() * 100
        }))

        # Signaux actuels
        st.divider()
        st.subheader("🎯 Signaux Prochains (Mise à jour mensuelle)")
        last_m = momentum.iloc[-1].sort_values(ascending=False)
        
        sc1, sc2 = st.columns(2)
        sc1.success(f"🚀 **ACHAT/PYRAMIDE (Top 2) :** {last_m.index[0]}, {last_m.index[1]}")
        sc2.info(f"📋 **WATCHLIST (Top 4) :** {', '.join(last_m.index[:4])}")

    except Exception as e:
        st.error(f"Détails de l'erreur : {e}")

if __name__ == "__main__":
    run_momentum_pro()
