# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:07:25 2024

@author: jperezr
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Funciones auxiliares
def calculate_weights(cov_matrix, clusters):
    cluster_var = {}
    for cluster in np.unique(clusters):
        # Identificación de activos en el clúster
        cluster_assets = [cov_matrix.columns[i] for i in range(len(clusters)) if clusters[i] == cluster]
        # Submatriz de covarianza para el clúster
        cov_cluster = cov_matrix.loc[cluster_assets, cluster_assets]
        # Suma de la inversa de la matriz de covarianza
        cluster_var[cluster] = np.sum(np.linalg.inv(cov_cluster))
    
    # Suma total de la varianza
    total_var = sum(cluster_var.values())
    # Pesos por clúster
    cluster_weights = {cluster: var / total_var for cluster, var in cluster_var.items()}
    return cluster_weights

def allocate_cluster_weights(cov_matrix, clusters, cluster_weights):
    asset_weights = {}
    for cluster in np.unique(clusters):
        # Identificación de activos en el clúster
        cluster_assets = [cov_matrix.columns[i] for i in range(len(clusters)) if clusters[i] == cluster]
        # Submatriz de covarianza para el clúster
        cov_cluster = cov_matrix.loc[cluster_assets, cluster_assets]
        # Cálculo de la inversa de la matriz de covarianza
        inverse_var = np.linalg.inv(cov_cluster).sum(axis=1)
        total_inv_var = inverse_var.sum()
        # Cálculo de pesos
        asset_weights.update({asset: cluster_weights[cluster] * inv_var / total_inv_var for asset, inv_var in zip(cluster_assets, inverse_var)})
    
    return asset_weights

# 2. Configuración de Streamlit
st.title("Optimización de Portafolios usando Paridad de Riesgo Jerárquica (HRP)")

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

# Calcular la matriz de covarianza
cov_matrix = returns.cov()
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

# 6. Paso 2: Agrupamiento jerárquico
st.header("Paso 3: Agrupamiento Jerárquico")

# Usar la distancia de correlación (1 - correlación) para agrupar
corr_distance = 1 - correlations
linkage_matrix = linkage(corr_distance, method='ward')

# Mostrar el dendrograma
st.subheader("Dendrograma de Agrupamiento Jerárquico")
fig, ax = plt.subplots(figsize=(12, 8))  # Aumenta el tamaño del gráfico
dendrogram(linkage_matrix, labels=correlations.columns, leaf_rotation=90)  # Rota las etiquetas 90 grados
plt.title("Dendrograma de Agrupamiento Jerárquico")
plt.xlabel("Activos")
plt.ylabel("Distancia")
st.pyplot(fig)

# Mostrar la matriz de agrupamiento (Linkage)
st.subheader("Matriz de agrupamiento (Linkage)")
st.dataframe(pd.DataFrame(linkage_matrix, columns=['Índice1', 'Índice2', 'Distancia', 'Cantidad de elementos']))

# 7. Paso 3: Asignación recursiva de pesos basada en el riesgo
st.header("Paso 4: Asignación Recursiva de Pesos")

# Agrupamos los activos en 3 clústeres (por ejemplo)
clusters = fcluster(linkage_matrix, 3, criterion='maxclust')

# Calcular la matriz de covarianza
cov_matrix = returns.cov()

# Asignación recursiva de pesos por clúster
cluster_weights = calculate_weights(cov_matrix, clusters)

# Asegúrate de convertir los pesos a un formato serializable
serializable_weights = {int(k): float(v) for k, v in cluster_weights.items()}  # Convertir a float
weights_df = pd.DataFrame(list(serializable_weights.items()), columns=['Cluster', 'Weight'])

# Pesos por clúster basado en el riesgo
st.subheader("Pesos por clúster basado en el riesgo")
st.dataframe(weights_df)

# 8. Paso 4: Asignación de pesos a los activos
st.header("Paso 5: Asignación de Pesos a los Activos")

# Calcular los pesos finales para cada activo
final_weights = allocate_cluster_weights(cov_matrix, clusters, cluster_weights)

st.subheader("Pesos finales por activo")
final_weights_df = pd.DataFrame(list(final_weights.items()), columns=['Asset', 'Weight'])
st.dataframe(final_weights_df)

# 9. Paso 5: Pesos del portafolio optimizado
st.header("Paso 6: Pesos del Portafolio Optimizado")

# Mostrar los pesos finales del portafolio
st.subheader("Portafolio optimizado - Pesos finales")
st.dataframe(final_weights_df)

# 10. Paso 6: Prueba del portafolio en 2022
st.header("Prueba del portafolio con datos de 2022")
test_data = yf.download(selected_tickers, start='2022-01-01', end='2022-12-31')['Adj Close']
test_returns = np.log(test_data / test_data.shift(1)).dropna()

# Calcular el rendimiento del portafolio en 2022
portfolio_return_2022 = (test_returns * pd.Series(final_weights)).sum(axis=1)

st.subheader("Rendimiento del portafolio en 2022")
st.line_chart(portfolio_return_2022.cumsum())