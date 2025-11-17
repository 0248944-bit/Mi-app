import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import io
import math

# Intenta configurar Gemini, sigue siendo opcional
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

# Configuración de la página (debe ser lo primero)
st.set_page_config(
    page_title="FinAnalyzer Pro - Análisis Financiero Inteligente",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 FinAnalyzer Pro - Análisis Financiero Inteligente")

# --- CSS (como en tu versión original) ---
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
    .info-box {
        background: linear-gradient(135deg, #e8f4f8, #d4edda);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #1f77b4;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Gemini configuration (safe) ---
gemini_configured = False
model = None
if genai is not None:
    try:
        API_KEY = st.secrets.get("GEMINI_API_KEY", None)
        if API_KEY:
            genai.configure(api_key=API_KEY)
            # No todas las instalaciones tendrán este objeto; se usa condicionalmente
            model = genai.GenerativeModel('gemini-2.0-flash')
            gemini_configured = True
            st.sidebar.success("✅ Gemini configurado correctamente")
        else:
            st.sidebar.warning("⚠️ GEMINI_API_KEY no encontrada en st.secrets")
    except Exception as e:
        st.sidebar.warning(f"Error configurando Gemini: {e}")
else:
    st.sidebar.info("Gemini SDK no instalado; análisis IA deshabilitado")

# HEADER
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="main-header">🚀 FinAnalyzer Pro</div>', unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;color:#6c757d'>Plataforma de análisis financiero inteligente con IA integrada</div>", unsafe_allow_html=True)

# SIDEBAR - Inputs mejorados
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("#### 📈 Acción Principal")
    stonk = st.text_input("Ticker principal:", value="MSFT")
    st.markdown("#### 🔄 Comparar con (coma separado)")
    comparar_tickers = st.text_input("Tickers:", value="AAPL, GOOGL")
    tickers_comparar = [t.strip().upper() for t in comparar_tickers.split(",") if t.strip()]
    st.markdown("#### 📅 Rango de Fechas")
    rango_personalizado = st.checkbox("Usar rango personalizado", value=False)
    if rango_personalizado:
        fecha_inicio = st.date_input("Fecha inicio", value=datetime.now().date() - timedelta(days=365))
        fecha_fin = st.date_input("Fecha fin", value=datetime.now().date())
        fecha_inicio = datetime.combine(fecha_inicio, datetime.min.time())
        fecha_actual = datetime.combine(fecha_fin, datetime.max.time())
    else:
        periodo = st.selectbox("Selecciona el período", ["1 mes", "3 meses", "6 meses", "1 año", "3 años", "5 años"], index=3)
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
    st.markdown("#### ⏱️ Intervalo")
    intervalo = st.selectbox("Intervalo de datos", ["1d", "1wk", "1mo"], index=0)
    st.markdown("#### 🧾 Opciones")
    max_tickers = st.slider("Máx tickers para análisis IA (si activado)", min_value=1, max_value=6, value=4)
    show_indicators = st.checkbox("Mostrar indicadores técnicos (SMA/EMA/MACD/RSI)", value=True)
    show_correlation = st.checkbox("Mostrar heatmap de correlación", value=True)
    benchmark = st.text_input("Benchmark (opcional, e.g. ^GSPC)", value="^GSPC")
    st.markdown("---")
    st.markdown("### 🤖 Análisis IA")
    if gemini_configured:
        if st.button("🎯 Opinión de la IA"):
            st.session_state.analisis_ia = True
    else:
        st.markdown("<div class='info-box'><strong>⚠️</strong> Gemini no configurado. El análisis IA está deshabilitado.</div>", unsafe_allow_html=True)

# Utilities: caching para descargas e info
@st.cache_data(ttl=3600)
def cached_download(tickers, start, end, interval):
    # usa yfinance.download en lote para mejorar performance
    try:
        df = yf.download(tickers, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), interval=interval, group_by='ticker', threads=True)
        return df
    except Exception as e:
        st.error(f"Error en descarga en lote: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def cached_ticker_info(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.info
    except Exception:
        return {}

# Técnical indicators
def add_technical_indicators(df, sma_short=20, sma_long=50, ema_span=20):
    df = df.copy()
    df['SMA_short'] = df['Close'].rolling(window=sma_short, min_periods=1).mean()
    df['SMA_long'] = df['Close'].rolling(window=sma_long, min_periods=1).mean()
    df['EMA'] = df['Close'].ewm(span=ema_span, adjust=False).mean()
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(com=13, adjust=False).mean()
    roll_down = down.ewm(com=13, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# Performance metrics
def compute_performance_metrics(df):
    # Devuelve un dict con claves siempre presentes (posiblemente np.nan)
    metrics = {
        'CAGR': np.nan,
        'Cumulative': np.nan,
        'Annualized Volatility': np.nan,
        'Sharpe': np.nan,
        'Max Drawdown': np.nan
    }
    try:
        if df is None or df.empty:
            return metrics
        rtn = df['Close'].pct_change().dropna()
        if rtn.empty:
            return metrics
        total_days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
        years = total_days / 365.25 if total_days > 0 else 1/252
        cumulative_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        # CAGR
        metrics['CAGR'] = (1 + cumulative_return) ** (1 / years) - 1 if years > 0 else np.nan
        # Volatilidad anualizada
        metrics['Annualized Volatility'] = rtn.std() * np.sqrt(252)
        # Sharpe simple (sin rf)
        metrics['Sharpe'] = (rtn.mean() / rtn.std()) * np.sqrt(252) if rtn.std() != 0 else np.nan
        # Max drawdown
        cum = (1 + rtn).cumprod()
        peak = cum.cummax()
        drawdown = (cum / peak) - 1
        metrics['Max Drawdown'] = drawdown.min()
        metrics['Cumulative'] = cumulative_return
    except Exception:
        # en caso de error dejamos np.nan ya definidos
        pass
    return metrics

# Helper para exportar CSV
def df_to_csv_bytes(df):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

# Preparar lista de tickers
todos_tickers = [stonk.upper()] + [t.upper() for t in tickers_comparar]
# Limpiar duplicados y limitar tamaño razonable
seen = []
for t in todos_tickers:
    if t not in seen:
        seen.append(t)
todos_tickers = seen[:8]  # límite de 8 para UI y performance

# Descarga en lote
with st.spinner("📥 Descargando datos..."):
    raw_data = cached_download(todos_tickers + ([benchmark] if benchmark else []), fecha_inicio, fecha_actual, intervalo)

# Procesar datos por ticker en dict
datos_tickers = {}
for t in todos_tickers:
    try:
        if len(todos_tickers) == 1:
            # raw_data is simple DF
            df_t = raw_data.copy()
        else:
            # yfinance group_by='ticker' estructura: columns multiindex
            if isinstance(raw_data.columns, pd.MultiIndex):
                if t in raw_data.columns.levels[0]:
                    df_t = raw_data[t].reset_index()
                else:
                    df_t = pd.DataFrame()
            else:
                # si sólo devolvió una columna agrupada
                df_t = raw_data.reset_index()
        if df_t is None or df_t.empty:
            datos_tickers[t] = pd.DataFrame()
            continue
        # Normalizar nombres (algunos endpoints devuelven 'Adj Close' como No MultiIndex)
        if 'Date' not in df_t.columns and 'index' in df_t.columns:
            df_t = df_t.rename(columns={'index': 'Date'})
        if 'Date' not in df_t.columns:
            df_t = df_t.reset_index().rename(columns={'index': 'Date'})
        # Asegurar columnas numeric
        for c in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']:
            if c in df_t.columns:
                df_t[c] = pd.to_numeric(df_t[c], errors='coerce')
        # Preferir 'Adj Close' si existe
        if 'Adj Close' in df_t.columns and 'Close' in df_t.columns:
            df_t['Close'] = df_t['Adj Close'].combine_first(df_t['Close'])
        df_t['Date'] = pd.to_datetime(df_t['Date'])
        df_t = df_t.sort_values('Date').dropna(subset=['Close'])
        if not df_t.empty:
            df_t = add_technical_indicators(df_t) if show_indicators else df_t
            df_t['Rendimiento_Diario'] = df_t['Close'].pct_change() * 100
            df_t['Rendimiento_Acumulado'] = (df_t['Close'] / df_t['Close'].iloc[0] - 1) * 100
        datos_tickers[t] = df_t
    except Exception as e:
        st.warning(f"Error procesando {t}: {e}")
        datos_tickers[t] = pd.DataFrame()

# Obtener info de tickers (cacheada) - nota: puede ser lenta para muchos tickers
info_tickers = {}
for t in todos_tickers:
    info_tickers[t] = cached_ticker_info(t)

# Mostrar resumen ejecutivo
st.markdown('<div class="section-header">📋 Resumen Ejecutivo</div>', unsafe_allow_html=True)
resumen_data = []
for ticker in todos_tickers:
    df = datos_tickers.get(ticker, pd.DataFrame())
    info = info_tickers.get(ticker, {})
    last_price = 'N/A'
    rendimiento_total = np.nan
    metrics = compute_performance_metrics(df)

    if df is not None and not df.empty:
        last_price = df['Close'].iloc[-1]
        if 'Rendimiento_Acumulado' in df.columns:
            rendimiento_total = df['Rendimiento_Acumulado'].iloc[-1]

    resumen_data.append({
        'Ticker': ticker,
        'Precio': f"${last_price:.2f}" if isinstance(last_price, (int, float, np.floating)) else 'N/A',
        'Rendimiento': f"{rendimiento_total:+.2f}%" if not pd.isna(rendimiento_total) else 'N/A',
        'CAGR': f"{metrics.get('CAGR', np.nan)*100:+.2f}%" if not pd.isna(metrics.get('CAGR', np.nan)) else 'N/A',
        'Sharpe': f"{metrics.get('Sharpe', np.nan):.2f}" if not pd.isna(metrics.get('Sharpe', np.nan)) else 'N/A',
        'Max DD': f"{metrics.get('Max Drawdown', np.nan)*100:.2f}%" if not pd.isna(metrics.get('Max Drawdown', np.nan)) else 'N/A',
        'Market Cap': f"${info.get('marketCap', 'N/A')/1e9:.1f}B" if isinstance(info.get('marketCap', None), (int, float)) else info.get('marketCap','N/A'),
        'Sector': info.get('sector', 'N/A')
    })

df_resumen = pd.DataFrame(resumen_data)

# Asegurar que las columnas que vamos a formatear existen
expected_cols = ['Precio','Rendimiento','CAGR','Sharpe','Max DD']
subset_cols = [c for c in expected_cols if c in df_resumen.columns]

if subset_cols:
    # Aplicar formato solo a las columnas existentes
    styled = df_resumen.style.format({col: "{}" for col in subset_cols}, na_rep='N/A')
    st.dataframe(styled, use_container_width=True, height=220)
else:
    st.dataframe(df_resumen, use_container_width=True, height=220)

# GRÁFICOS
col_g1, col_g2 = st.columns([2,1])
with col_g1:
    st.markdown('<div class="section-header">📈 Evolución de Precios (normalizado opcional)</div>', unsafe_allow_html=True)
    normalizar = st.checkbox("Normalizar precios al 100% al inicio para comparar tendencias", value=True)
    fig, ax = plt.subplots(figsize=(12,6))
    for t in todos_tickers:
        df = datos_tickers.get(t, pd.DataFrame())
        if df is None or df.empty:
            continue
        plot_df = df.copy()
        if normalizar:
            plot_df['Norm'] = plot_df['Close'] / plot_df['Close'].iloc[0] * 100
            ax.plot(plot_df['Date'], plot_df['Norm'], label=t)
        else:
            ax.plot(plot_df['Date'], plot_df['Close'], label=t)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (normalizado)" if normalizar else "Precio (USD)")
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with col_g2:
    st.markdown('<div class="section-header">📊 Correlación y Heatmap</div>', unsafe_allow_html=True)
    if show_correlation:
        # crear dataframe de returns
        returns = pd.DataFrame()
        for t in todos_tickers:
            df = datos_tickers.get(t, pd.DataFrame())
            if df is None or df.empty: continue
            returns[t] = df.set_index('Date')['Close'].pct_change().rename(t)
        if not returns.empty:
            corr = returns.corr()
            fig2, ax2 = plt.subplots(figsize=(6,5))
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax2)
            ax2.set_title("Correlación de retornos")
            st.pyplot(fig2)
        else:
            st.info("No hay suficientes datos para calcular correlación.")

# INDICADORES EN DETALLE (pestañas por ticker)
st.markdown('<div class="section-header">🏢 Análisis por Empresa</div>', unsafe_allow_html=True)
tabs = st.tabs([f"{t}" for t in todos_tickers])
for i, t in enumerate(todos_tickers):
    with tabs[i]:
        df = datos_tickers.get(t, pd.DataFrame())
        info = info_tickers.get(t, {})
        if df is None or df.empty:
            st.warning(f"No hay datos para {t}")
            continue
        # Titular y botones de exportación
        colA, colB = st.columns([3,1])
        with colA:
            st.subheader(f"{t} - {info.get('longName','')}")
            st.write(f"Sector: {info.get('sector','N/A')} | Market cap: {info.get('marketCap','N/A')}")
        with colB:
            csv_bytes = df_to_csv_bytes(df)
            st.download_button(label="📥 Descargar CSV", data=csv_bytes, file_name=f"{t}_data.csv", mime="text/csv")
        # Mostrar métricas
        metrics = compute_performance_metrics(df)
        cols = st.columns(4)
        cols[0].metric("Precio", f"${df['Close'].iloc[-1]:.2f}", "")
        cols[1].metric("CAGR", f"{metrics.get('CAGR',0)*100:+.2f}%" if metrics else "N/A")
        cols[2].metric("Sharpe", f"{metrics.get('Sharpe',np.nan):.2f}" if metrics else "N/A")
        cols[3].metric("Max Drawdown", f"{metrics.get('Max Drawdown',0)*100:.2f}%" if metrics else "N/A")
        # Price chart con overlays
        fig3, ax3 = plt.subplots(figsize=(12,5))
        ax3.plot(df['Date'], df['Close'], label='Close', color='#1f77b4')
        if show_indicators and 'SMA_short' in df.columns:
            ax3.plot(df['Date'], df['SMA_short'], label='SMA short', linestyle='--', alpha=0.8)
            ax3.plot(df['Date'], df['SMA_long'], label='SMA long', linestyle='--', alpha=0.8)
            ax3.plot(df['Date'], df['EMA'], label='EMA', linestyle=':', alpha=0.8)
        ax3.set_title(f"{t} - Precio e indicadores")
        ax3.set_xlabel("Fecha")
        ax3.set_ylabel("Precio (USD)")
        ax3.legend(loc='upper left')
        ax3.grid(alpha=0.2)
        st.pyplot(fig3)
        # MACD y RSI si existen
        if show_indicators and 'MACD' in df.columns:
            fig4, (axm, axr) = plt.subplots(2,1, figsize=(12,6), gridspec_kw={'height_ratios':[2,1]})
            axm.plot(df['Date'], df['MACD'], label='MACD')
            axm.plot(df['Date'], df['MACD_signal'], label='MACD signal')
            axm.axhline(0, color='black', linestyle='--', alpha=0.5)
            axm.legend()
            axm.set_title("MACD")
            axr.plot(df['Date'], df['RSI'], label='RSI', color='purple')
            axr.axhline(70, color='red', linestyle='--', alpha=0.5)
            axr.axhline(30, color='green', linestyle='--', alpha=0.5)
            axr.set_ylim(0,100)
            axr.set_title("RSI (14)")
            st.pyplot(fig4)
        # Tabla de últimos valores
        st.dataframe(df[['Date','Close','Rendimiento_Diario']].tail(10), use_container_width=True)

# Backtest simple: cartera igual ponderada
st.markdown('<div class="section-header">💼 Backtest simple - Cartera igual ponderada</div>', unsafe_allow_html=True)
if len(todos_tickers) >= 1:
    # construir returns aligned
    price_df = pd.DataFrame()
    for t in todos_tickers:
        df = datos_tickers.get(t, pd.DataFrame())
        if df is None or df.empty: continue
        tmp = df.set_index('Date')['Close'].rename(t)
        price_df = pd.concat([price_df, tmp], axis=1)
    price_df = price_df.dropna(axis=0, how='any')
    if price_df.empty:
        st.info("No hay datos comunes para realizar backtest")
    else:
        returns = price_df.pct_change().dropna()
        weights = np.array([1/returns.shape[1]]*returns.shape[1])
        portfolio_returns = returns.dot(weights)
        portfolio_cum = (1 + portfolio_returns).cumprod()
        bh_returns = (1 + returns.mean(axis=1)).cumprod()
        fig5, ax5 = plt.subplots(figsize=(10,5))
        ax5.plot(portfolio_cum.index, portfolio_cum.values, label='Cartera igual ponderada')
        ax5.plot(bh_returns.index, bh_returns.values, label='Promedio Buy&Hold (por ticker)')
        ax5.legend()
        ax5.set_title("Backtest simplificado")
        ax5.set_ylabel("Crecimiento acumulado")
        st.pyplot(fig5)
        # métricas
        port_metrics = {}
        port_metrics['Cumulative'] = portfolio_cum.values[-1]-1
        port_metrics['CAGR'] = (portfolio_cum.values[-1])**(1/((portfolio_cum.index[-1]-portfolio_cum.index[0]).days/365.25)) - 1
        port_metrics['Volatility'] = portfolio_returns.std() * np.sqrt(252)
        st.write("Métricas de cartera:")
        st.json(port_metrics)

# Análisis IA (limitado por tokens y número de tickers)
if st.session_state.get('analisis_ia', False) and gemini_configured:
    st.markdown('<div class="section-header">🤖 Análisis IA (Gemini)</div>', unsafe_allow_html=True)
    # Construir prompt acotado
    limited_tickers = todos_tickers[:max_tickers]
    prompt_lines = ["Eres un analista financiero senior. Resume y compara brevemente las siguientes empresas con métricas claves:"]
    for t in limited_tickers:
        info = info_tickers.get(t, {})
        df = datos_tickers.get(t, pd.DataFrame())
        last_price = df['Close'].iloc[-1] if not df.empty else 'N/A'
        cagr = compute_performance_metrics(df).get('CAGR', None)
        prompt_lines.append(f"- {t}: nombre={info.get('longName','N/A')}, sector={info.get('sector','N/A')}, price={last_price}, CAGR={cagr}")
    prompt_lines.append("Proporciona: 1) resumen, 2) riesgos, 3) recomendación (compáralas). Usa números concretos cuando existan.")
    prompt = "\n".join(prompt_lines)[:5000]  # acotar longitud
    try:
        with st.spinner("🤖 Gemini generando análisis..."):
            response = model.generate_content(prompt)
            st.markdown(f"<div class='info-box'>{response.text}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error pidiendo análisis a Gemini: {e}")

# FOOTER
st.markdown("---")
st.markdown(f"<div style='text-align:center;color:#6c757d;padding:2rem'>Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • © FinAnalyzer Pro</div>", unsafe_allow_html=True)
