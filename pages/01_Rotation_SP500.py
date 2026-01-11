import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(page_title="Momentum Pro SaaS", layout="wide")

def calculate_metrics(returns):
    """Calcule les métriques clés de performance"""
    # Performance totale
    total_return = (returns + 1).prod() - 1
    # CAGR (Taux de croissance annuel composé)
    n_years = len(returns) / 12
    cagr = (total_return + 1) ** (1 / n_years) - 1
    # Volatilité annualisée
    vol = returns.std() * np.sqrt(12)
    # Ratio de Sharpe (sans risque estimé à 2%)
    sharpe = (cagr - 0.02) / vol if vol != 0 else 0
    # Max Drawdown
    cum_rets = (returns + 1).cumprod()
    rolling_max = cum_rets.cummax()
    drawdown = (cum_rets / rolling_max - 1).min()
    
    return cagr, vol, sharpe, drawdown, total_return

def run_momentum_pro():
    st.title("🛡️ Momentum Pro : Stratégie & Analytics")
    
    sectors = ['XLK', 'XLF', 'XLV', 'XLY', 'XLI', 'XLP', 'XLE', 'XLC', 'XLB', 'XLU', 'XLRE']
    
    @st.cache_data
    def load_data():
        # On télécharge un peu avant 1999 pour avoir les 3 mois de look-back dès le départ
        data = yf.download(sectors + ['SPY'], start="1998-10-01")
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data['Close']
            open_prices = data['Open']
        else:
            close_prices = data['Close']
            open_prices = data['Open']
        return close_prices.ffill(), open_prices.ffill()

    try:
        close_data, open_data = load_data()
        monthly_close = close_data.resample('ME').last()
        monthly_open = open_data.resample('MS').first()
        momentum = monthly_close[sectors].pct_change(3)
        
        history = []
        current_portfolio = {} 
        
        for i in range(3, len(monthly_close) - 1):
            date_signal = monthly_close.index[i]
            date_invest = monthly_open.index[i+1]
            
            scores = momentum.iloc[i].sort_values(ascending=False)
            top_2 = scores.index[:2].tolist()
            top_4 = scores.index[:4].tolist()
            
            new_portfolio = {}
            for ticker, lots in current_portfolio.items():
                if ticker in top_4:
                    new_portfolio[ticker] = lots
                    if ticker in top_2 and lots == 1:
                        new_portfolio[ticker] = 2
            
            for ticker in top_2:
                if ticker not in new_portfolio:
                    new_portfolio[ticker] = 1
            
            month_return = 0
            num_lots_total = sum(new_portfolio.values())
            
            if num_lots_total > 0:
                for ticker, lots in new_portfolio.items():
                    # Rendement mensuel calculé d'ouverture à clôture suivante
                    try:
                        ret = (close_data[ticker].loc[monthly_close.index[i+1]] / open_data[ticker].loc[date_invest]) - 1
                        month_return += (ret * (lots / num_lots_total))
                    except: continue
            
            history.append({
                'Date': monthly_close.index[i+1],
                'Strat_Ret': month_return,
                'SPY_Ret': (close_data['SPY'].loc[monthly_close.index[i+1]] / open_data['SPY'].loc[date_invest]) - 1
            })
            current_portfolio = new_portfolio

        df_res = pd.DataFrame(history).set_index('Date')
        
        # --- CALCUL DES METRIQUES ---
        m_strat = calculate_metrics(df_res['Strat_Ret'])
        m_spy = calculate_metrics(df_res['SPY_Ret'])

        # --- AFFICHAGE DASHBOARD ---
        st.subheader("📊 Comparatif de Performance (1999 - Présent)")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CAGR Stratégie", f"{m_strat[0]*100:.2f}%", f"{(m_strat[0]-m_spy[0])*100:.2f}% vs SPY")
        col2.metric("Sharpe Ratio", f"{m_strat[2]:.2f}", f"{(m_strat[2]-m_spy[2]):.2f} pts")
        col3.metric("Max Drawdown", f"{m_strat[3]*100:.2f}%", f"{(m_strat[3]-m_spy[3])*100:.2f}% pts", delta_color="inverse")
        col4.metric("Performance Totale", f"{m_strat[4]*100:.1f}%")

        # --- TABLEAU RECAPITULATIF ---
        metrics_table = pd.DataFrame({
            'Métrique': ['Performance Totale', 'CAGR (Rendement Annuel)', 'Ratio de Sharpe', 'Max Drawdown (Risque)'],
            'Stratégie Momentum': [f"{m_strat[4]*100:.1f}%", f"{m_strat[0]*100:.2f}%", f"{m_strat[2]:.2f}", f"{m_strat[3]*100:.2f}%"],
            'S&P 500 (Buy & Hold)': [f"{m_spy[4]*100:.1f}%", f"{m_spy[0]*100:.2f}%", f"{m_spy[2]:.2f}", f"{m_spy[3]*100:.2f}%"]
        })
        st.table(metrics_table)

        # Graphique
        st.line_chart(pd.DataFrame({
            'Stratégie': (1 + df_res['Strat_Ret']).cumprod() * 100,
            'S&P 500': (1 + df_res['SPY_Ret']).cumprod() * 100
        }))

        # Signaux du mois (le plus important pour l'abonnement)
        st.divider()
        st.subheader("🎯 Signaux pour le mois suivant")
        last_m = momentum.iloc[-1].sort_values(ascending=False)
        st.info(f"Analyse basée sur la clôture du {monthly_close.index[-1].date()}")
        
        c1, c2 = st.columns(2)
        c1.success(f"🔥 **PYRAMIDE (Top 2) :** {last_m.index[0]}, {last_m.index[1]}")
        c2.warning(f"🛡️ **BUFFER (Top 4) :** {', '.join(last_m.index[:4])}")

    except Exception as e:
        st.error(f"Erreur : {e}")

if __name__ == "__main__":
    run_momentum_pro()
