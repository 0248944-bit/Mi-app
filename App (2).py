import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import google.generativeai as genai  # ← CORREGIDO: "genai" con N
import matplotlib.dates as mdates
import numpy as np



# Clave de API de Gemini
API_KEY = "AIzaSyCMqbyl7yrGtY-Os1BPgoOJpRgnX49E_Wv0"

# Configuración de la página
st.set_page_config(
    page_title="FinAnalyzer Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS mejorados (Responsive)
st.markdown("""
<style>
    .main-header {
        font-size: clamp(2rem, 5vw, 3.5rem) !important;
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
        font-size: clamp(1.5rem, 3vw, 2rem) !important;
        color: #2e86ab;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.8rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 1rem;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    @media (max-width: 768px) {
        .metric-card {
            padding: 0.8rem;
            min-height: 100px;
        }
    }
    
    .info-box {
        background: linear-gradient(135deg, #e8f4f8, #d4edda);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #1f77b4;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .company-section {
        background: linear-gradient(135deg, #f0f8ff, #e6f3ff);
        padding: 1.5rem;
        border-radius: 20px;
        border: 3px solid #1f77b4;
        margin-bottom: 2rem;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    .ai-analysis {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        padding: 2rem;
        border-radius: 20px;
        border: 3px solid #ffc107;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    
    .portfolio-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        padding: 1.5rem;
        border-radius: 15px;
        border: 3px solid #28a745;
        margin-top: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        padding: 1.2rem;
        border-radius: 10px;
        border: 2px solid #dc3545;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        padding: 1.2rem;
        border-radius: 10px;
        border: 2px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f0f2f6;
        padding: 8px;
        border-radius: 10px;
        flex-wrap: wrap;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: nowrap;
        background-color: #e9ecef;
        border-radius: 8px;
        padding: 10px 15px;
        font-weight: 600;
        transition: all 0.3s ease;
        flex: 1;
        min-width: 120px;
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
        padding: 10px 20px;
        border-radius: 10px;
        border: none;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        width: 100%;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #1668a3, #1a759f);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    /* Mejoras para móviles */
    @media (max-width: 768px) {
        .section-header {
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .ai-analysis {
            padding: 1.5rem;
        }
        
        .portfolio-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Configuración de Gemini 
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    gemini_configured = True
except Exception as e:
    st.sidebar.warning(f"⚠️ Error configurando Gemini: {e}")
    model = None
    gemini_configured = False

# Inicializar session state para portafolio
if 'portfolios' not in st.session_state:
    st.session_state.portfolios = {}
if 'current_portfolio' not in st.session_state:
    st.session_state.current_portfolio = None

# HEADER PRINCIPAL MEJORADO
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="main-header">🚀 FinAnalyzer Pro+</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #6c757d; font-size: clamp(1rem, 2vw, 1.2rem); margin-bottom: 2rem;'>
        Plataforma avanzada de análisis financiero con gestión de portafolios
    </div>
    """, unsafe_allow_html=True)

# BARRA LATERAL MEJORADA
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #1f77b4, #2e86ab); 
                border-radius: 15px; margin-bottom: 2rem; color: white;'>
        <h2 style='margin: 0; font-size: 1.5rem;'>⚙️ Configuración</h2>
        <p style='margin: 0; font-size: 0.9rem;'>Personaliza tu análisis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Selector de modo de análisis
    modo_analisis = st.radio(
        "**🎯 Modo de Análisis**",
        ["📈 Análisis de Mercado", "💼 Gestión de Portafolios"],
        index=0
    )
    
    if modo_analisis == "📈 Análisis de Mercado":
        st.markdown("### 📈 Acción Principal")
        stonk = st.text_input(
            "**Ticker de la acción principal**", 
            value="MSFT",
            help="Ejemplos: AAPL, TSLA, GOOGL, AMZN, NVDA"
        )
        
        st.markdown("### 🔄 Comparar Con")
        comparar_tickers = st.text_input(
            "**Tickers para comparar (separados por coma)**", 
            value="AAPL, GOOGL",
            help="Ingresa hasta 5 tickers adicionales para comparar"
        )
        
        tickers_comparar = [ticker.strip().upper() for ticker in comparar_tickers.split(",") if ticker.strip()]
        
        st.markdown("### 📅 Período de Análisis")
        periodo = st.selectbox(
            "**Selecciona el período**", 
            ["1 mes", "3 meses", "6 meses", "1 año", "3 años", "5 años"], 
            index=4
        )
    
    else:  # Gestión de Portafolios
        st.markdown("### 💼 Gestión de Portafolios")
        
        # Crear nuevo portafolio
        with st.expander("➕ Crear Nuevo Portafolio", expanded=True):
            nuevo_portfolio_nombre = st.text_input("Nombre del portafolio")
            nuevo_portfolio_desc = st.text_area("Descripción (opcional)")
            
            if st.button("🎯 Crear Portafolio", use_container_width=True):
                if nuevo_portfolio_nombre:
                    if nuevo_portfolio_nombre not in st.session_state.portfolios:
                        st.session_state.portfolios[nuevo_portfolio_nombre] = {
                            'descripcion': nuevo_portfolio_desc,
                            'inversiones': [],
                            'fecha_creacion': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.session_state.current_portfolio = nuevo_portfolio_nombre
                        st.success(f"✅ Portafolio '{nuevo_portfolio_nombre}' creado!")
                        st.rerun()
                else:
                    st.error("❌ Ingresa un nombre para el portafolio")
        
        # Selector de portafolio existente
        if st.session_state.portfolios:
            st.markdown("### 📂 Portafolios Existentes")
            portfolio_seleccionado = st.selectbox(
                "Selecciona un portafolio",
                list(st.session_state.portfolios.keys())
            )
            st.session_state.current_portfolio = portfolio_seleccionado
            
            # Gestión de inversiones
            if st.session_state.current_portfolio:
                with st.expander("💰 Agregar Inversión", expanded=True):
                    col_inv1, col_inv2 = st.columns(2)
                    with col_inv1:
                        ticker_inversion = st.text_input("Ticker", placeholder="AAPL")
                        cantidad = st.number_input("Cantidad de acciones", min_value=1, value=100)
                    with col_inv2:
                        precio_compra = st.number_input("Precio de compra ($)", min_value=0.01, value=150.0)
                        fecha_compra = st.date_input("Fecha de compra", value=datetime.now())
                    
                    if st.button("💾 Agregar Inversión", use_container_width=True):
                        if ticker_inversion:
                            nueva_inversion = {
                                'ticker': ticker_inversion.upper(),
                                'cantidad': cantidad,
                                'precio_compra': precio_compra,
                                'fecha_compra': fecha_compra.strftime("%Y-%m-%d"),
                                'id': f"{ticker_inversion}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                            }
                            st.session_state.portfolios[st.session_state.current_portfolio]['inversiones'].append(nueva_inversion)
                            st.success(f"✅ Inversión en {ticker_inversion} agregada!")
                            st.rerun()
    
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

# FUNCIONES MEJORADAS
def calcular_rendimientos(data):
    """Calcula los rendimientos porcentuales diarios y acumulados"""
    data = data.copy()
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data.dropna(subset=['Close'])
    
    data['Rendimiento_Diario'] = data['Close'].pct_change() * 100
    
    if len(data) > 0:
        precio_inicial = data['Close'].iloc[0]
        data['Rendimiento_Acumulado'] = (data['Close'] / precio_inicial - 1) * 100
    
    data['Rendimiento_Rolling_30d'] = data['Rendimiento_Diario'].rolling(window=30).mean()
    data['Volatilidad_30d'] = data['Rendimiento_Diario'].rolling(window=30).std()
    
    # Indicadores técnicos adicionales
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calcular_rsi(data['Close'])
    
    return data

def calcular_rsi(precios, periodo=14):
    """Calcula el RSI (Relative Strength Index)"""
    delta = precios.diff()
    ganancia = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
    perdida = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
    rs = ganancia / perdida
    rsi = 100 - (100 / (1 + rs))
    return rsi

def obtener_analisis_ia(tickers, info_tickers, datos_tickers):
    """Obtiene análisis comparativo de Gemini"""
    try:
        prompt = "Eres un analista financiero senior. Analiza las siguientes empresas:\n\n"
        
        for ticker in tickers:
            if ticker in info_tickers:
                info = info_tickers[ticker]
                data = datos_tickers[ticker]
                
                try:
                    rendimiento_total = data['Rendimiento_Acumulado'].iloc[-1] if not data.empty and 'Rendimiento_Acumulado' in data.columns else 0
                    volatilidad_promedio = data['Volatilidad_30d'].mean() if not data.empty and 'Volatilidad_30d' in data.columns else 0
                    rsi_actual = data['RSI'].iloc[-1] if not data.empty and 'RSI' in data.columns else 50
                except:
                    rendimiento_total = 0
                    volatilidad_promedio = 0
                    rsi_actual = 50
                
                prompt += f"""
📊 {ticker} - {info.get('longName', 'N/A')}
Sector: {info.get('sector', 'N/A')}
Valoración:
• Capitalización: ${info.get('marketCap', 0):,.0f}
• P/E Ratio: {info.get('trailingPE', 'N/A')}
• Dividend Yield: {info.get('dividendYield', 0)*100:.2f}%
• Margen: {info.get('profitMargins', 0)*100:.2f}%
• ROE: {info.get('returnOnEquity', 0)*100:.2f}%
• Rendimiento: {rendimiento_total:.2f}%
• Volatilidad: {volatilidad_promedio:.2f}%
• RSI: {rsi_actual:.1f}

"""
        
        prompt += """
Proporciona:
🎯 ANÁLISIS COMPARATIVO - Valoración, crecimiento, rentabilidad
📈 OPINIÓN TÉCNICA - Tendencias y patrones
💡 RECOMENDACIONES - Específicas por empresa
🏆 MEJOR OPCIÓN - Justificación detallada
⚠️ RIESGOS - Principales riesgos

Sé detallado y usa números concretos.
"""
        
        with st.spinner('🤖 Gemini está realizando análisis...'):
            response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        return f"❌ Error al obtener análisis de IA: {str(e)}"

def descargar_datos(ticker, fecha_inicio, fecha_actual):
    """Descarga y procesa datos financieros"""
    try:
        with st.spinner(f'📥 Descargando {ticker}...'):
            data = yf.download(ticker, start=fecha_inicio, end=fecha_actual, interval='1d')
        
        if data.empty:
            return None
        
        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in data.columns]
        
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
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data = data.dropna()
        
        if not data.empty:
            data = calcular_rendimientos(data)
        
        return data
    except Exception as e:
        st.error(f"Error descargando {ticker}: {e}")
        return None

def obtener_info_empresa(ticker):
    """Obtiene información fundamental de la empresa"""
    try:
        with st.spinner(f'🔍 Obteniendo información de {ticker}...'):
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
        return info
    except Exception as e:
        st.error(f"Error obteniendo información de {ticker}: {e}")
        return {}

def analizar_portafolio(portfolio):
    """Analiza el rendimiento del portafolio"""
    if not portfolio or not portfolio['inversiones']:
        return None
    
    analisis = {
        'inversiones': [],
        'total_invertido': 0,
        'valor_actual': 0,
        'rendimiento_total': 0,
        'composicion': {}
    }
    
    for inversion in portfolio['inversiones']:
        try:
            # Obtener precio actual
            ticker_data = descargar_datos(inversion['ticker'], 
                                        fecha_actual - timedelta(days=30), 
                                        fecha_actual)
            
            if ticker_data is not None and not ticker_data.empty:
                precio_actual = ticker_data['Close'].iloc[-1]
                inversion_actual = inversion['cantidad'] * precio_actual
                inversion_original = inversion['cantidad'] * inversion['precio_compra']
                rendimiento = ((precio_actual - inversion['precio_compra']) / inversion['precio_compra']) * 100
                ganancia_perdida = inversion_actual - inversion_original
                
                analisis['inversiones'].append({
                    **inversion,
                    'precio_actual': precio_actual,
                    'valor_actual': inversion_actual,
                    'valor_original': inversion_original,
                    'rendimiento': rendimiento,
                    'ganancia_perdida': ganancia_perdida
                })
                
                analisis['total_invertido'] += inversion_original
                analisis['valor_actual'] += inversion_actual
                analisis['composicion'][inversion['ticker']] = inversion_actual
        except Exception as e:
            st.error(f"Error analizando {inversion['ticker']}: {e}")
    
    if analisis['total_invertido'] > 0:
        analisis['rendimiento_total'] = ((analisis['valor_actual'] - analisis['total_invertido']) / analisis['total_invertido']) * 100
    
    return analisis

# CONTENIDO PRINCIPAL
try:
    if modo_analisis == "💼 Gestión de Portafolios":
        # SECCIÓN DE PORTAFOLIOS
        st.markdown('<div class="section-header">💼 Gestión de Portafolios</div>', unsafe_allow_html=True)
        
        if not st.session_state.portfolios:
            st.info("""
            **📋 No tienes portafolios creados**
            
            Para comenzar:
            1. Ve a la barra lateral
            2. En "Gestión de Portafolios"
            3. Crea tu primer portafolio
            4. Agrega tus inversiones
            """)
        else:
            portfolio_actual = st.session_state.portfolios[st.session_state.current_portfolio]
            analisis_portfolio = analizar_portafolio(portfolio_actual)
            
            if analisis_portfolio:
                # MÉTRICAS DEL PORTAFOLIO
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; color: #6c757d;">Total Invertido</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #1f77b4;">
                            ${analisis_portfolio['total_invertido']:,.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; color: #6c757d;">Valor Actual</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #28a745;">
                            ${analisis_portfolio['valor_actual']:,.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    color_rendimiento = "#28a745" if analisis_portfolio['rendimiento_total'] >= 0 else "#dc3545"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; color: #6c757d;">Rendimiento Total</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: {color_rendimiento};">
                            {analisis_portfolio['rendimiento_total']:+.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; color: #6c757d;">Número de Activos</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #6f42c1;">
                            {len(analisis_portfolio['inversiones'])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # GRÁFICO DE COMPOSICIÓN
                st.markdown("### 📊 Composición del Portafolio")
                
                if analisis_portfolio['composicion']:
                    fig_composicion = px.pie(
                        values=list(analisis_portfolio['composicion'].values()),
                        names=list(analisis_portfolio['composicion'].keys()),
                        title="Distribución de Activos"
                    )
                    st.plotly_chart(fig_composicion, use_container_width=True)
                
                # DETALLE DE INVERSIONES
                st.markdown("### 📋 Detalle de Inversiones")
                
                inversiones_data = []
                for inv in analisis_portfolio['inversiones']:
                    inversiones_data.append({
                        'Ticker': inv['ticker'],
                        'Cantidad': inv['cantidad'],
                        'Precio Compra': f"${inv['precio_compra']:.2f}",
                        'Precio Actual': f"${inv['precio_actual']:.2f}",
                        'Inversión': f"${inv['valor_original']:,.2f}",
                        'Valor Actual': f"${inv['valor_actual']:,.2f}",
                        'Rendimiento': f"{inv['rendimiento']:+.2f}%",
                        'Ganancia/Pérdida': f"${inv['ganancia_perdida']:+.2f}"
                    })
                
                if inversiones_data:
                    df_inversiones = pd.DataFrame(inversiones_data)
                    st.dataframe(df_inversiones, use_container_width=True)
            
            # GESTIÓN DE INVERSIONES EXISTENTES
            if portfolio_actual['inversiones']:
                st.markdown("### 🛠️ Gestionar Inversiones")
                
                for i, inversion in enumerate(portfolio_actual['inversiones']):
                    with st.expander(f"📈 {inversion['ticker']} - {inversion['cantidad']} acciones", expanded=False):
                        col_gest1, col_gest2, col_gest3 = st.columns([2, 1, 1])
                        
                        with col_gest1:
                            st.write(f"**Comprado:** {inversion['fecha_compra']}")
                            st.write(f"**Precio compra:** ${inversion['precio_compra']:.2f}")
                        
                        with col_gest2:
                            if st.button(f"✏️ Editar", key=f"edit_{i}"):
                                st.info("Función de edición en desarrollo")
                        
                        with col_gest3:
                            if st.button(f"🗑️ Eliminar", key=f"del_{i}"):
                                st.session_state.portfolios[st.session_state.current_portfolio]['inversiones'].pop(i)
                                st.success("Inversión eliminada!")
                                st.rerun()
    
    else:  # MODO ANÁLISIS DE MERCADO
        fecha_inicio = fecha_actual - periodo_map[periodo]
        
        # Lista de todos los tickers a procesar
        todos_tickers = [stonk] + tickers_comparar
        
        # Mostrar progreso de descarga
        with st.status("📥 Descargando datos de mercado...", expanded=True) as status:
            st.write("Iniciando descarga de datos financieros...")
            
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
        
        # SECCIÓN: ANÁLISIS DE IA
        if st.session_state.get('analisis_ia', False) and gemini_configured:
            st.markdown('<div class="section-header">🤖 Análisis Inteligente por IA</div>', unsafe_allow_html=True)
            
            with st.spinner('🚀 Ejecutando análisis avanzado con Gemini...'):
                analisis_ia = obtener_analisis_ia(todos_tickers, info_tickers, datos_tickers)
            
            st.markdown(f'<div class="ai-analysis">{analisis_ia}</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🗑️ Cerrar Análisis IA", use_container_width=True):
                    st.session_state.analisis_ia = False
                    st.rerun()
        
        # RESUMEN EJECUTIVO
        st.markdown('<div class="section-header">📋 Resumen Ejecutivo</div>', unsafe_allow_html=True)
        
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
                    rsi_actual = data['RSI'].iloc[-1] if 'RSI' in data.columns else 'N/A'
                    
                    resumen_data.append({
                        'Ticker': ticker,
                        'Precio': f"${precio_actual:.2f}",
                        'Rendimiento': f"{rendimiento_total:+.2f}%",
                        'Market Cap': f"${market_cap/1e9:.1f}B",
                        'P/E': f"{pe_ratio:.1f}" if pe_ratio != 'N/A' else 'N/A',
                        'RSI': f"{rsi_actual:.1f}" if rsi_actual != 'N/A' else 'N/A',
                        'Sector': info.get('sector', 'N/A')
                    })
        
        if resumen_data:
            df_resumen = pd.DataFrame(resumen_data)
            
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
            
            def color_rsi(val):
                if val != 'N/A':
                    try:
                        num = float(val)
                        if num > 70:
                            return 'background-color: #f8d7da; color: #721c24;'
                        elif num < 30:
                            return 'background-color: #d4edda; color: #155724;'
                    except:
                        pass
                return ''
            
            styled_df = df_resumen.style.map(color_rendimiento, subset=['Rendimiento']).map(color_rsi, subset=['RSI'])
            st.dataframe(styled_df, use_container_width=True, height=200)
        
        # GRÁFICOS INTERACTIVOS CON PLOTLY
        col_graf1, col_graf2 = st.columns(2)
        
        with col_graf1:
            st.markdown('<div class="section-header">📈 Comparación de Precios</div>', unsafe_allow_html=True)
            
            fig = go.Figure()
            colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            for i, (ticker, data) in enumerate(datos_tickers.items()):
                color = colores[i % len(colores)]
                
                if len(data) > 100:
                    data_plot = data.iloc[::3]
                else:
                    data_plot = data
                
                fig.add_trace(go.Scatter(
                    x=data_plot['Date'],
                    y=data_plot['Close'],
                    name=ticker,
                    line=dict(color=color, width=2),
                    hovertemplate='<b>%{x}</b><br>Precio: $%{y:.2f}<extra></extra>'
                ))
            
            fig.update_layout(
                title=f"Evolución de Precios - {periodo}",
                xaxis_title="Fecha",
                yaxis_title="Precio (USD)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col_graf2:
            st.markdown('<div class="section-header">📊 Rendimientos Acumulados</div>', unsafe_allow_html=True)
            
            fig = go.Figure()
            
            for i, (ticker, data) in enumerate(datos_tickers.items()):
                color = colores[i % len(colores)]
                
                if len(data) > 100:
                    data_plot = data.iloc[::3]
                else:
                    data_plot = data
                
                if 'Rendimiento_Acumulado' in data_plot.columns:
                    fig.add_trace(go.Scatter(
                        x=data_plot['Date'],
                        y=data_plot['Rendimiento_Acumulado'],
                        name=ticker,
                        line=dict(color=color, width=2),
                        hovertemplate='<b>%{x}</b><br>Rendimiento: %{y:.2f}%<extra></extra>'
                    ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
            fig.update_layout(
                title=f"Rendimientos Acumulados - {periodo}",
                xaxis_title="Fecha",
                yaxis_title="Rendimiento (%)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # INDICADORES TÉCNICOS
        st.markdown('<div class="section-header">🔬 Análisis Técnico Avanzado</div>', unsafe_allow_html=True)
        
        ticker_seleccionado = st.selectbox("Seleccionar ticker para análisis técnico", todos_tickers)
        
        if ticker_seleccionado in datos_tickers:
            data_tecnico = datos_tickers[ticker_seleccionado]
            
            fig_tecnico = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Precio con Medias Móviles', 'RSI'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Gráfico de precios con medias móviles
            fig_tecnico.add_trace(
                go.Scatter(x=data_tecnico['Date'], y=data_tecnico['Close'], 
                          name='Precio Cierre', line=dict(color='#1f77b4')),
                row=1, col=1
            )
            
            if 'SMA_20' in data_tecnico.columns:
                fig_tecnico.add_trace(
                    go.Scatter(x=data_tecnico['Date'], y=data_tecnico['SMA_20'], 
                              name='SMA 20', line=dict(color='orange', dash='dash')),
                    row=1, col=1
                )
            
            if 'SMA_50' in data_tecnico.columns:
                fig_tecnico.add_trace(
                    go.Scatter(x=data_tecnico['Date'], y=data_tecnico['SMA_50'], 
                              name='SMA 50', line=dict(color='red', dash='dash')),
                    row=1, col=1
                )
            
            # Gráfico RSI
            if 'RSI' in data_tecnico.columns:
                fig_tecnico.add_trace(
                    go.Scatter(x=data_tecnico['Date'], y=data_tecnico['RSI'], 
                              name='RSI', line=dict(color='purple')),
                    row=2, col=1
                )
                
                # Líneas de sobrecompra/sobreventa
                fig_tecnico.add_hline(y=70, line_dash="dash", line_color="red", 
                                     annotation_text="Sobrecompra", row=2, col=1)
                fig_tecnico.add_hline(y=30, line_dash="dash", line_color="green", 
                                     annotation_text="Sobreventa", row=2, col=1)
            
            fig_tecnico.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig_tecnico, use_container_width=True)

except Exception as e:
    st.error(f"""
    ❌ Error al procesar los datos: {str(e)}
    
    **Solución de problemas:**
    - Verifica tu conexión a internet
    - Revisa que los tickers sean válidos
    - Intenta con un período de tiempo más corto
    - Recarga la página
    """)

# FOOTER MEJORADO
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; padding: 2rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); 
            border-radius: 15px; margin-top: 2rem;'>
    <h3 style='color: #1f77b4; margin-bottom: 1rem;'>🚀 FinAnalyzer Pro+</h3>
    <p style='margin-bottom: 0.5rem;'><strong>Plataforma avanzada de análisis financiero</strong></p>
    <p style='margin-bottom: 1rem; font-size: 0.9rem;'>
        Desarrollado con Streamlit • Yahoo Finance • Gemini AI • Análisis Técnico • Gestión de Portafolios
    </p>
    <p style='margin-top: 1.5rem; font-size: 0.8rem; color: #868e96;'>
        Última actualización: {} | © 2024 FinAnalyzer Pro+
    </p>
</div>
""".format(fecha_actual.strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)