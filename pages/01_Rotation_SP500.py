import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_rotation_page():
    st.title("🚀 Stratégie de Rotation Sectorielle S&P 500")
    st.write("Cet outil analyse les 11 secteurs du S&P 500 pour sélectionner les leaders par rapport au risque.")

    # Sidebar pour les paramètres
    with st.sidebar:
        st.header("Paramètres")
        lookback_period = st.slider("Période d'analyse (mois)", 1, 12, 3)
        start_date = st.date_input("Date de début du backtest", pd.to_datetime("2019-01-01"))

    sectors = ['XLK', 'XLF', 'XLV', 'XLY', 'XLI', 'XLP', 'XLE', 'XLC', 'XLB', 'XLU', 'XLRE']
    
    if st.button('Lancer l\'analyse et le Backtest'):
        with st.spinner('Récupération des données boursières...'):
            # Téléchargement sécurisé
            all_tickers = sectors + ['SPY']
            raw_data = yf.download(all_tickers, start=start_date, interval="1d")
            
            # Correction de l'erreur KeyError : On vérifie la structure des colonnes
            if 'Adj Close' in raw_data.columns:
                data = raw_data['Adj Close']
            else:
                data = raw_data['Close'] # Backup si Adj Close n'est pas dispo

            # Nettoyage des données
            data = data.ffill().dropna()

            # Calcul des rendements mensuels
            monthly_data = data.resample('ME').last()
            monthly_returns = monthly_data.pct_change().dropna()
            
            strategy_returns = []

            # Simulation du Backtest
            for i in range(1, len(monthly_returns)):
                # Calcul sur la période de lookback choisie
                lookback = monthly_returns.iloc[max(0, i-lookback_period):i]
                
                # Calcul du score (Rendement / Volatilité)
                # On ajoute une petite valeur pour éviter la division par zéro
                sharpe_scores = lookback[sectors].mean() / (lookback[sectors].std() + 1e-6)
                
                # Top 3 secteurs
                top_sectors = sharpe_scores.nlargest(3).index
                
                # Performance le mois suivant
                next_month_ret = monthly_returns.iloc[i][top_sectors].mean()
                strategy_returns.append(next_month_ret)

            # Résultats finaux
            results = pd.DataFrame({
                'Stratégie': strategy_returns,
                'S&P 500': monthly_returns['SPY'].iloc[1:].values
            }, index=monthly_returns.index[1:])
            
            cum_results = (1 + results).cumprod()

            # --- AFFICHAGE ---
            col1, col2 = st.columns(2)
            final_strat = (cum_results['Stratégie'].iloc[-1] - 1) * 100
            final_spy = (cum_results['S&P 500'].iloc[-1] - 1) * 100
            
            col1.metric("Performance Stratégie", f"{final_strat:.2f}%", f"{final_strat - final_spy:.2f}% vs SPY")
            col2.metric("Performance S&P 500", f"{final_spy:.2f}%")

            # Graphique
            st.subheader("Courbe de croissance cumulée")
            st.line_chart(cum_results)

            # Signal Actuel
            st.divider()
            st.subheader("🎯 Signal pour le mois actuel")
            last_lookback = monthly_returns[sectors].tail(lookback_period)
            current_scores = last_lookback.mean() / (last_lookback.std() + 1e-6)
            best_now = current_scores.nlargest(3)
            
            cols = st.columns(3)
            for i, (ticker, score) in enumerate(best_now.items()):
                cols[i].success(f"Secteur {i+1} : **{ticker}**")

# Lancement
if __name__ == "__main__":
    run_rotation_page()
