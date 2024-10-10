# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:20:23 2024

@author: jperezr
"""

import streamlit as st
import base64



# Título de la aplicación
st.title("Optimización de Portafolios")

# Resumen del Artículo
st.header("Resumen del Artículo: Portfolio Optimization – A Comparative Study")
st.write("""
Este artículo presenta un estudio comparativo sobre diversas técnicas de optimización de portafolios. Se analizan métodos clásicos como la optimización de media-varianza, así como enfoques más recientes basados en técnicas de aprendizaje automático. 
El objetivo principal es evaluar la efectividad de cada método en la maximización del rendimiento ajustado al riesgo, considerando la volatilidad de los activos y la correlación entre ellos. 
Se utilizan datos históricos de diferentes activos para ilustrar la aplicación práctica de estos métodos y se discuten las implicaciones para los inversores en la toma de decisiones.
""")

#st.sidebar.title("Información Personal")
st.sidebar.write("Nombre: Javier Horacio Pérez Ricárdez")

# Sección del proyecto final
st.header("Escrito")

# Ruta del primer archivo PDF
pdf_file_path_1 = "escrito.pdf"  # Cambia esto a la ruta de tu primer archivo PDF

# Leer el primer archivo PDF
with open(pdf_file_path_1, "rb") as pdf_file_1:
    pdf_bytes_1 = pdf_file_1.read()

# Ofrecer la descarga del primer archivo PDF
b64_pdf_1 = base64.b64encode(pdf_bytes_1).decode("utf-8")
href_1 = f'<a href="data:application/octet-stream;base64,{b64_pdf_1}" download="escrito.pdf">Descargar PDF, escrito</a>'
st.markdown(href_1, unsafe_allow_html=True)
