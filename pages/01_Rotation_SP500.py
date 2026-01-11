import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_rotation_page():
    st.title("🚀 Stratégie de Rotation Sectorielle S&P 500")
    st.write("Cet outil analyse les 11 secteurs du S&P 500 pour sélectionner les 3 plus performants par rapport au risque.")

    # Sidebar pour les paramètres utilisateur (Donne un aspect SaaS pro)
    with st.sidebar:
        st.header("Paramètres")
        lookback_period = st.slider("Période d'analyse (mois)", 1, 12, 3)
        start_date = st.date_input("Date de début du backtest", pd.to_datetime("2019-01-01"))

    sectors = ['XLK', 'XLF', 'XLV', 'XLY', 'XLI', 'XLP', 'XLE', 'XLC', 'XLB', 'XLU', 'XLRE']
    
    if st.button('Lancer l\'analyse et le Backtest'):
        with st.spinner('Récupération des données boursières...'):
            data = yf.download(sectors + ['SPY'], start=start_date)['Adj Close']
            
            # Logique de calcul (identique au script précédent)
            monthly_data = data.resample('ME').last()
            monthly_returns = monthly_data.pct_change().dropna()
            strategy_returns = []

            for i in range(1, len(monthly_returns)):
                lookback = monthly_returns.iloc[max(0, i-lookback_period):i]
                sharpe_scores = lookback[sectors].mean() / lookback[sectors].std()
                top_sectors = sharpe_scores.nlargest(3).index
                next_month_ret = monthly_returns.iloc[i][top_sectors].mean()
                strategy_returns.append(next_month_ret)

            # Préparation des résultats
            results = pd.DataFrame({
                'Stratégie': strategy_returns,
                'S&P 500': monthly_returns['SPY'].iloc[1:].values
            }, index=monthly_returns.index[1:])
            
            cum_results = (1 + results).cumprod()

            # --- AFFICHAGE STREAMLIT ---
            col1, col2 = st.columns(2)
            final_strat = (cum_results['Stratégie'].iloc[-1] - 1) * 100
            final_spy = (cum_results['S&P 500'].iloc[-1] - 1) * 100
            
            col1.metric("Performance Stratégie", f"{final_strat:.2f}%", f"{final_strat - final_spy:.2f}% vs SPY")
            col2.metric("Performance S&P 500", f"{final_spy:.2f}%")

            # Graphique Interactif
            st.line_chart(cum_results)

            # Conseil Actionnable (Le cœur du SaaS)
            st.subheader("🎯 Signal pour le mois actuel")
            current_sharpe = monthly_returns[sectors].tail(lookback_period).mean() / monthly_returns[sectors].tail(lookback_period).std()
            best_now = current_sharpe.nlargest(3)
            
            st.success(f"Les secteurs recommandés sont : **{', '.join(best_now.index.tolist())}**")

# Si tu l'utilises en page unique ou import
if __name__ == "__main__":
    run_rotation_page()
