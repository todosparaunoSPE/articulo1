# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:56:21 2024

@author: jperezr
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Funciones auxiliares
def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_std_dev

def optimize_portfolio(mean_returns, cov_matrix, num_portfolios=10000):
    results = np.zeros((3, num_portfolios))
    weights_record = np.zeros((num_portfolios, len(mean_returns)))  # Para almacenar los pesos
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        
        portfolio_return, portfolio_std_dev = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = (results[0, i] - 0.01) / results[1, i]  # Sharpe ratio
        weights_record[i, :] = weights  # Guardar los pesos
    return results, weights_record

# 2. Configuración de Streamlit
st.title("Optimización de Portafolios usando Media-Varianza (MVP)")

# 3. Parámetros de entrada
tickers = ['RELIANCE.NS', 'ULTRACEMCO.NS', 'TATASTEEL.NS', 'NTPC.NS', 
           'JSWSTEEL.NS', 'ONGC.NS', 'GRASIM.NS', 'HINDALCO.NS', 
           'COALINDIA.NS', 'UPL.NS']

# 4. Paso 0: Selección de Tickers y Fechas
st.header("Paso 1: Selección de Tickers y Fechas")

# Combobox para selección de tickers
selected_tickers = st.multiselect("Selecciona los activos:", tickers, default=tickers)

# Combobox para selección de fecha de inicio
start_date = st.date_input("Fecha de inicio", pd.to_datetime('2018-01-01'))
end_date = st.date_input("Fecha de fin", pd.to_datetime('2022-12-31'))

# 5. Paso 1: Obtener los datos de Yahoo Finance
st.header("Paso 2: Datos de entrada (Rendimientos y Correlaciones)")

# Descargar datos
data = yf.download(selected_tickers, start=start_date, end=end_date)['Adj Close']

# Mostrar los datos descargados
st.subheader("Precios ajustados")
st.dataframe(data)

# Calcular rendimientos logarítmicos diarios
returns = np.log(data / data.shift(1)).dropna()

st.subheader("Rendimientos diarios (logarítmicos)")
st.dataframe(returns)

# Calcular la matriz de covarianza y la media de rendimientos
cov_matrix = returns.cov()
mean_returns = returns.mean()

st.subheader("Matriz de Covarianza")
st.dataframe(cov_matrix)

# Mostrar correlaciones
correlations = returns.corr()
st.subheader("Matriz de correlaciones")
st.dataframe(correlations)

# Visualización de la matriz de correlaciones
st.subheader("Visualización de la Matriz de Correlaciones")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlations, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# 6. Paso 2: Optimización del Portafolio
st.header("Paso 3: Optimización del Portafolio")

# Optimizar el portafolio
num_portfolios = 10000
results, weights_record = optimize_portfolio(mean_returns, cov_matrix, num_portfolios)

# Crear un DataFrame para los resultados
results_df = pd.DataFrame(results.T, columns=['Returns', 'Volatility', 'Sharpe Ratio'])

# Visualizar los resultados
st.subheader("Resultados de la Optimización del Portafolio")
st.dataframe(results_df)

# Visualización de la frontera eficiente
st.subheader("Frontera Eficiente")
fig, ax = plt.subplots(figsize=(10, 8))

# Graficar todos los puntos, usando el Sharpe Ratio como el color
sc = ax.scatter(results_df['Volatility'], results_df['Returns'], 
                 c=results_df['Sharpe Ratio'], cmap='viridis', marker='o')

# Encontrar el portafolio con el mejor Sharpe Ratio
optimal_index = results_df['Sharpe Ratio'].idxmax()
optimal_return = results_df['Returns'].iloc[optimal_index]
optimal_volatility = results_df['Volatility'].iloc[optimal_index]

# Agregar el punto óptimo en rojo
ax.scatter(optimal_volatility, optimal_return, color='red', s=100, label='Portafolio Óptimo')

# Etiquetas y título
ax.set_xlabel('Riesgo (Volatilidad)')
ax.set_ylabel('Rendimiento Esperado')
ax.set_title('Frontera Eficiente')

# Crear la barra de colores
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Sharpe Ratio')  # Etiqueta de la barra de colores

# Añadir leyenda
ax.legend()

# Mostrar la figura
st.pyplot(fig)

# 7. Paso 3: Pesos del Portafolio Óptimo
st.header("Paso 4: Pesos del Portafolio Óptimo")

# Obtener los pesos del portafolio óptimo
optimal_weights = weights_record[optimal_index]

st.subheader("Pesos óptimos del portafolio")
final_weights_df = pd.DataFrame({'Activos': selected_tickers, 'Pesos': optimal_weights})
st.dataframe(final_weights_df)

# 8. Paso 4: Prueba del portafolio en 2022
st.header("Prueba del portafolio con datos de 2022")
test_data = yf.download(selected_tickers, start='2022-01-01', end='2022-12-31')['Adj Close']
test_returns = np.log(test_data / test_data.shift(1)).dropna()

# Calcular el rendimiento del portafolio en 2022
portfolio_return_2022 = (test_returns * optimal_weights).sum(axis=1)

st.subheader("Rendimiento del portafolio en 2022")
if not portfolio_return_2022.empty:
    st.line_chart(portfolio_return_2022.cumsum())
else:
    st.write("No hay datos para mostrar el rendimiento del portafolio.")