import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt 




st.title("Primera app en línea con Streamlit    y Python")

ticker = st.text_input("Ingrese el símbolo de la acción (por ejemplo, AAPL para Apple):", "AAPL")

if ticker:
    data = yf.download(ticker, period="1mo", interval="1d")
    pi