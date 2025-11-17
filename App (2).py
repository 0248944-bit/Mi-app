import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import numpy as np
import os

# Configuración de la página (debe ser lo primero)
st.set_page_config(
    page_title="FinAnalyzer Pro - Análisis Financiero Inteligente",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título de la app
st.title("📊 FinAnalyzer Pro - Análisis Financiero Inteligente")

# Clave de API de Gemini - REEMPLAZA CON TU API KEY REAL
API_KEY = os.getenv('API_KEY', 'fallback_key_si_no_existe')

# Estilos CSS mejorados
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem !important;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1f77b4, #2e86ab);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .section-header {
        font-size: 2rem !important;
        color: #2e86ab;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.8rem;
        margin-top: 3rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .info-box {
        background: linear-gradient(135deg, #e8f4f8, #d4edda);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #1f77b4;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .company-section {
        background: linear-gradient(135deg, #f0f8ff, #e6f3ff);
        padding: 2rem;
        border-radius: 20px;
        border: 3px solid #1f77b4;
        margin-bottom: 2.5rem;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    .ai-analysis {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        padding: 2.5rem;
        border-radius: 20px;
        border: 3px solid #ffc107;
        margin-bottom: 2.5rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        padding: 2rem;
        border-radius: 15px;
        border: 3px solid #28a745;
        margin-top: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #dc3545;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #e9ecef;
        border-radius: 8px;
        gap: 1px;
        padding: 15px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1f77b4, #2e86ab);
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stButton button {
        background: linear-gradient(135deg, #1f77b4, #2e86ab);
        color: white;
        font-weight: bold;
        padding: 12px 30px;
        border-radius: 10px;
        border: none;
        font-size: 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #1668a3, #1a759f);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Configuración de Gemini (SINTAXIS ACTUALIZADA)
try:
    from google import genai
    client = genai.Client(api_key=API_KEY)
    gemini_configured = True
    
except Exception as e:
    st.sidebar.warning(f"⚠️ Error configurando Gemini: {e}")
    client = None
    gemini_configured = False

# HEADER PRINCIPAL MEJORADO
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="main-header">🚀 FinAnalyzer Pro</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #6c757d; font-size: 1.2rem; margin-bottom: 2rem;'>
        Plataforma de análisis financiero inteligente con IA integrada
    </div>
    """, unsafe_allow_html=True)

# BARRA LATERAL MEJORADA
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #1f77b4, #2e86ab); 
                border-radius: 15px; margin-bottom: 2rem; color: white;'>
        <h2>⚙️ Configuración</h2>
        <p style='margin: 0; font-size: 0.9rem;'>Personaliza tu análisis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input del ticker principal
    st.markdown("### 📈 Acción Principal")
    stonk = st.text_input(
        "**Ticker de la acción principal**", 
        value="MSFT",
        help="Ejemplos: AAPL, TSLA, GOOGL, AMZN, NVDA"
    )
    
    # Selector de acciones para comparar
    st.markdown("### 🔄 Comparar Con")
    comparar_tickers = st.text_input(
        "**Tickers para comparar (separados por coma)**", 
        value="AAPL, GOOGL",
        help="Ingresa hasta 5 tickers adicionales para comparar"
    )
    
    # Procesar los tickers para comparar
    tickers_comparar = [ticker.strip().upper() for ticker in comparar_tickers.split(",") if ticker.strip()]
    
    # Selector de período
    st.markdown("### 📅 Período de Análisis")
    periodo = st.selectbox(
        "**Selecciona el período**", 
        ["1 mes", "3 meses", "6 meses", "1 año", "3 años", "5 años"], 
        index=4
    )
    
    # Botón para análisis de IA
    st.markdown("---")
    st.markdown("### 🤖 Análisis IA")
    
    if gemini_configured:
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🎯 Opinión de la IA", use_container_width=True, type="primary"):
                st.session_state.analisis_ia = True
        with col_btn2:
            if st.button("🔄 Actualizar Datos", use_container_width=True):
                st.rerun()
        
        if 'analisis_ia' not in st.session_state:
            st.session_state.analisis_ia = False
            
        st.markdown("""
        <div class='success-box'>
            <strong>💡 Tip:</strong> La IA analizará todas las empresas y te dará recomendaciones específicas.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='warning-box'>
            <strong>⚠️ Atención:</strong> Configura correctamente la API de Gemini para usar el análisis IA.
        </div>
        """, unsafe_allow_html=True)

# CÁLCULO DE FECHAS SEGÚN PERIODO SELECCIONADO
fecha_actual = datetime.now()
periodo_map = {
    "1 mes": timedelta(days=30),
    "3 meses": timedelta(days=90),
    "6 meses": timedelta(days=180),
    "1 año": timedelta(days=365),
    "3 años": timedelta(days=3*365),
    "5 años": timedelta(days=5*365)
}
fecha_inicio = fecha_actual - periodo_map[periodo]

# Función para calcular rendimientos porcentuales (CORREGIDA)
def calcular_rendimientos(data):
    """
    Calcula los rendimientos porcentuales diarios y acumulados
    """
    data = data.copy()
    
    # Asegurarse de que Close es numérico
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data.dropna(subset=['Close'])
    
    # Rendimiento diario porcentual
    data['Rendimiento_Diario'] = data['Close'].pct_change() * 100
    
    # Rendimiento acumulado (desde el primer día del dataset)
    if len(data) > 0:
        precio_inicial = data['Close'].iloc[0]
        data['Rendimiento_Acumulado'] = (data['Close'] / precio_inicial - 1) * 100
    
    # Rendimiento rolling (promedio móvil de 30 días)
    data['Rendimiento_Rolling_30d'] = data['Rendimiento_Diario'].rolling(window=30).mean()
    
    # Volatilidad rolling (desviación estándar de 30 días)
    data['Volatilidad_30d'] = data['Rendimiento_Diario'].rolling(window=30).std()
    
    return data

# Función para obtener análisis comparativo de Gemini (ACTUALIZADA)
def obtener_analisis_ia(tickers, info_tickers, datos_tickers):
    """
    Obtiene análisis comparativo de Gemini basado en la información fundamental
    """
    try:
        # Construir prompt con información de todas las empresas
        prompt = """
        Eres un analista financiero senior. Analiza las siguientes empresas y proporciona un análisis completo:
        
        **INFORMACIÓN DE LAS EMPRESAS:**
        """
        
        for ticker in tickers:
            if ticker in info_tickers:
                info = info_tickers[ticker]
                data = datos_tickers[ticker]
                
                # Calcular métricas adicionales (manejar errores)
                try:
                    rendimiento_total = data['Rendimiento_Acumulado'].iloc[-1] if not data.empty and 'Rendimiento_Acumulado' in data.columns else 0
                    volatilidad_promedio = data['Volatilidad_30d'].mean() if not data.empty and 'Volatilidad_30d' in data.columns else 0
                except:
                    rendimiento_total = 0
                    volatilidad_promedio = 0
                
                prompt += f"""
                
                📊 **{ticker} - {info.get('longName', 'N/A')}**
                **Sector:** {info.get('sector', 'N/A')}
                **Valoración:**
                • Capitalización: ${info.get('marketCap', 0):,.0f}
                • P/E Ratio: {info.get('trailingPE', 'N/A')}
                • Dividend Yield: {info.get('dividendYield', 0)*100:.2f}%
                • Margen de Beneficio: {info.get('profitMargins', 0)*100:.2f}%
                • Return on Equity: {info.get('returnOnEquity', 0)*100:.2f}%
                • Rendimiento Total: {rendimiento_total:.2f}%
                • Volatilidad: {volatilidad_promedio:.2f}%
                """
        
        prompt += """
        
        **POR FAVOR PROPORCIONA:**
        
        🎯 **ANÁLISIS COMPARATIVO**
        Compara las empresas en valoración, crecimiento y rentabilidad

        📈 **OPINIÓN SOBRE COMPORTAMIENTO**
        Analiza tendencias y patrones de precio

        💡 **RECOMENDACIONES**
        Recomendaciones específicas para cada empresa

        🏆 **MEJOR OPCIÓN**
        Cuál empresa consideras mejor para invertir y por qué

        ⚠️ **RIESGOS**
        Principales riesgos a considerar

        **IMPORTANTE:** Sé detallado y usa números concretos.
        """
        
        with st.spinner('🤖 Gemini está realizando análisis...'):
            response = client.models.generate_content(
                model="gemini-2.0-flash",  # MODELO ACTUALIZADO
                contents=prompt
            )
        
        return response.text
        
    except Exception as e:
        return f"❌ Error al obtener análisis de IA: {str(e)}"

# Función para descargar y procesar datos (CORREGIDA)
def descargar_datos(ticker, fecha_inicio, fecha_actual):
    try:
        with st.spinner(f'📥 Descargando {ticker}...'):
            data = yf.download(ticker, start=fecha_inicio.strftime('%Y-%m-%d'), 
                              end=fecha_actual.strftime('%Y-%m-%d'), interval='1d')
        
        if data.empty:
            return None
        
        # Procesar datos
        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in data.columns]
        
        # Mapeo de columnas
        column_mapping = {'Date': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 
                         'Close': 'Close', 'Volume': 'Volume', 'Adj Close': 'Close'}
        
        processed_data = pd.DataFrame()
        for standard_name, possible_names in column_mapping.items():
            for col_name in data.columns:
                if col_name in possible_names or col_name.startswith(possible_names + '_'):
                    processed_data[standard_name] = data[col_name]
                    break
        
        data = processed_data
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Convertir columnas numéricas de forma segura
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data = data.dropna()
        
        # Calcular rendimientos
        if not data.empty:
            data = calcular_rendimientos(data)
        
        return data
    except Exception as e:
        st.error(f"Error descargando {ticker}: {e}")
        return None

# Función para obtener información de la empresa
def obtener_info_empresa(ticker):
    try:
        with st.spinner(f'🔍 Obteniendo información de {ticker}...'):
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
        return info
    except Exception as e:
        st.error(f"Error obteniendo información de {ticker}: {e}")
        return {}

# Función para mostrar tarjeta de métricas (CORREGIDA)
def mostrar_metric_card(label, value, delta=None):
    # Limpiar el valor si es un string con porcentaje
    if isinstance(value, str) and '%' in value:
        # Extraer solo el número para mostrar
        clean_value = value.replace('%', '').replace('+', '')
        try:
            numeric_value = float(clean_value)
            value = f"{numeric_value:+.2f}%"
        except:
            pass
    
    st.metric(label=label, value=value, delta=delta)

# Función para mostrar información corporativa (CORREGIDA)
def mostrar_info_corporativa(ticker, info, es_principal=True):
    if es_principal:
        st.markdown(f'<div class="section-header">🏢 {ticker} - Información Corporativa</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="company-section">')
        st.markdown(f'<h3 style="color: #1f77b4; margin-bottom: 1.5rem;">🏢 {ticker} - Información Corporativa</h3>', unsafe_allow_html=True)
    
    # Header de la empresa
    nombre = info.get("longName", "No disponible")
    st.markdown(f"### {nombre}")
    
    # Métricas principales en cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        market_cap = info.get('marketCap', 0)
        mostrar_metric_card("💰 Capitalización", f"${market_cap/1e9:.2f}B")
    
    with col2:
        pe_ratio = info.get('trailingPE', 'N/A')
        pe_display = f"{pe_ratio:.1f}" if pe_ratio != 'N/A' else 'N/A'
        mostrar_metric_card("📊 P/E Ratio", pe_display)
    
    with col3:
        dividend_yield = info.get('dividendYield', 0) * 100
        mostrar_metric_card("💸 Dividend Yield", f"{dividend_yield:.2f}%")
    
    with col4:
        sector = info.get('sector', 'N/A')
        mostrar_metric_card("🏭 Sector", sector)
    
    # Segunda fila de métricas
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        beta = info.get('beta', 'N/A')
        beta_display = f"{beta:.2f}" if beta != 'N/A' else 'N/A'
        mostrar_metric_card("📈 Beta", beta_display)
    
    with col6:
        profit_margin = info.get('profitMargins', 0) * 100
        mostrar_metric_card("🎯 Margen Beneficio", f"{profit_margin:.1f}%")
    
    with col7:
        roe = info.get('returnOnEquity', 0) * 100
        mostrar_metric_card("🚀 ROE", f"{roe:.1f}%")
    
    with col8:
        employees = info.get('fullTimeEmployees', 'N/A')
        emp_display = f"{employees:,}" if employees != 'N/A' else 'N/A'
        mostrar_metric_card("👥 Empleados", emp_display)
    
    # Descripción de la empresa
    descripcion = info.get("longBusinessSummary", "No disponible")
    if descripcion != "No disponible":
        st.markdown("#### 📝 Descripción de la Empresa")
        st.markdown(f'<div class="info-box">{descripcion[:800]}...</div>', unsafe_allow_html=True)
    
    if not es_principal:
        st.markdown('</div>', unsafe_allow_html=True)

# CONTENIDO PRINCIPAL
try:
    # Lista de todos los tickers a procesar
    todos_tickers = [stonk] + tickers_comparar
    
    # Mostrar progreso de descarga
    with st.status("📥 Descargando datos de mercado...", expanded=True) as status:
        st.write("Iniciando descarga de datos financieros...")
        
        # Descargar datos para todos los tickers
        datos_tickers = {}
        info_tickers = {}
        
        progress_bar = st.progress(0)
        for i, ticker in enumerate(todos_tickers):
            st.write(f"📊 Procesando {ticker}...")
            data = descargar_datos(ticker, fecha_inicio, fecha_actual)
            if data is not None:
                datos_tickers[ticker] = data
                info_tickers[ticker] = obtener_info_empresa(ticker)
                st.success(f"✅ {ticker} - Datos descargados correctamente")
            else:
                st.error(f"❌ {ticker} - Error en descarga")
            
            progress_bar.progress((i + 1) / len(todos_tickers))
        
        status.update(label="✅ ¡Todos los datos descargados!", state="complete", expanded=False)
    
    if not datos_tickers:
        st.error("❌ No se encontraron datos para ningún ticker ingresado")
        st.stop()
    
    # SECCIÓN: ANÁLISIS DE IA (si se solicitó)
    if st.session_state.get('analisis_ia', False) and gemini_configured:
        st.markdown('<div class="section-header">🤖 Análisis Inteligente por IA</div>', unsafe_allow_html=True)
        
        with st.spinner('🚀 Ejecutando análisis avanzado con Gemini...'):
            analisis_ia = obtener_analisis_ia(todos_tickers, info_tickers, datos_tickers)
        
        st.markdown(f'<div class="ai-analysis">{analisis_ia}</div>', unsafe_allow_html=True)
        
        # Botones de acción
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Cerrar Análisis IA", use_container_width=True):
                st.session_state.analisis_ia = False
                st.rerun()
    
    # RESUMEN EJECUTIVO (CORREGIDO)
    st.markdown('<div class="section-header">📋 Resumen Ejecutivo</div>', unsafe_allow_html=True)
    
    # Crear resumen con métricas clave
    resumen_data = []
    for ticker in todos_tickers:
        if ticker in datos_tickers and ticker in info_tickers:
            data = datos_tickers[ticker]
            info = info_tickers[ticker]
            
            if not data.empty and 'Rendimiento_Acumulado' in data.columns:
                precio_actual = data['Close'].iloc[-1]
                rendimiento_total = data['Rendimiento_Acumulado'].iloc[-1]
                market_cap = info.get('marketCap', 0)
                pe_ratio = info.get('trailingPE', 'N/A')
                
                resumen_data.append({
                    'Ticker': ticker,
                    'Precio': f"${precio_actual:.2f}",
                    'Rendimiento': f"{rendimiento_total:+.2f}%",
                    'Market Cap': f"${market_cap/1e9:.1f}B",
                    'P/E': f"{pe_ratio:.1f}" if pe_ratio != 'N/A' else 'N/A',
                    'Sector': info.get('sector', 'N/A')
                })
    
    if resumen_data:
        df_resumen = pd.DataFrame(resumen_data)
        
        # Función para aplicar colores al rendimiento
        def color_rendimiento(val):
            if '%' in str(val):
                try:
                    num = float(str(val).replace('%', '').replace('+', ''))
                    if num > 0:
                        return 'background-color: #d4edda; color: #155724;'
                    elif num < 0:
                        return 'background-color: #f8d7da; color: #721c24;'
                except:
                    pass
            return ''
        
        styled_df = df_resumen.style.applymap(color_rendimiento, subset=['Rendimiento'])
        st.dataframe(styled_df, use_container_width=True, height=200)
    
    # GRÁFICOS PRINCIPALES CON MATPLOTLIB
    col_graf1, col_graf2 = st.columns(2)
    
    with col_graf1:
        st.markdown('<div class="section-header">📈 Comparación de Precios</div>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, (ticker, data) in enumerate(datos_tickers.items()):
            color = colores[i % len(colores)]
            
            # Muestreo para mejor performance
            if len(data) > 100:
                data_plot = data.iloc[::5]
            else:
                data_plot = data
            
            ax.plot(data_plot['Date'], data_plot['Close'], 
                   label=ticker, color=color, linewidth=2.5, alpha=0.8)
        
        ax.set_title(f"Evolución de Precios - {periodo}", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Fecha", fontsize=12)
        ax.set_ylabel("Precio (USD)", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
    
    with col_graf2:
        st.markdown('<div class="section-header">📊 Rendimientos Acumulados</div>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for i, (ticker, data) in enumerate(datos_tickers.items()):
            color = colores[i % len(colores)]
            
            if len(data) > 100:
                data_plot = data.iloc[::5]
            else:
                data_plot = data
            
            if 'Rendimiento_Acumulado' in data_plot.columns:
                ax.plot(data_plot['Date'], data_plot['Rendimiento_Acumulado'], 
                       label=ticker, color=color, linewidth=2.5, alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(f"Rendimientos Acumulados - {periodo}", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Fecha", fontsize=12)
        ax.set_ylabel("Rendimiento (%)", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # INFORMACIÓN DETALLADA POR EMPRESA
    st.markdown('<div class="section-header">🏢 Análisis por Empresa</div>', unsafe_allow_html=True)
    
    # Usar pestañas para cada empresa
    tabs = st.tabs([f"📊 {ticker}" for ticker in todos_tickers if ticker in info_tickers])
    
    for i, ticker in enumerate([t for t in todos_tickers if t in info_tickers]):
        with tabs[i]:
            if ticker in info_tickers and info_tickers[ticker]:
                mostrar_info_corporativa(ticker, info_tickers[ticker], es_principal=(i == 0))
    
    # FOOTER MEJORADO
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 3rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); 
                border-radius: 15px; margin-top: 2rem;'>
        <h3 style='color: #1f77b4; margin-bottom: 1rem;'>🚀 FinAnalyzer Pro</h3>
        <p style='margin-bottom: 0.5rem;'><strong>Plataforma de análisis financiero avanzado</strong></p>
        <p style='margin-bottom: 1rem; font-size: 0.9rem;'>
            Desarrollado con Streamlit • Integración con Yahoo Finance • Análisis IA con Gemini
        </p>
        <p style='margin-top: 1.5rem; font-size: 0.8rem; color: #868e96;'>
            Última actualización: {} | © 2024 FinAnalyzer Pro
        </p>
    </div>
    """.format(fecha_actual.strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

except Exception as e:
    st.error(f"""
    ❌ Error al procesar los datos: {str(e)}
    
    **Solución de problemas:**
    - Verifica tu conexión a internet
    - Revisa que los tickers sean válidos
    - Intenta con un período de tiempo más corto
    """)