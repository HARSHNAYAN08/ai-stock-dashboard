import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import requests
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="AI Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Dark Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0F172A;
        color: #E2E8F0;
    }
    
    .stApp {
        background-color: #0F172A;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #00C896;
        font-weight: 600;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #00C896;
    }
    
    .metric-label {
        font-size: 14px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .positive {
        color: #10B981;
    }
    
    .negative {
        color: #EF4444;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #00C896;
        margin: 20px 0;
    }
    
    .stButton>button {
        background-color: #00C896;
        color: #0F172A;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #00A87E;
        box-shadow: 0 4px 12px rgba(0, 200, 150, 0.3);
    }
    
    .sidebar .sidebar-content {
        background-color: #1E293B;
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: #64748B;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'sentiment_cache' not in st.session_state:
    st.session_state.sentiment_cache = {}

# Header
st.markdown("<h1 style='text-align: center;'>📈 AI-Powered Stock Market Insight Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8;'>Real-time Analysis • AI Insights • Predictive Forecasting</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Dashboard Controls")
    
    # Stock Selection
    popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'JPM', 'V']
    selected_stocks = st.multiselect(
        "Select Stocks",
        options=popular_stocks,
        default=['AAPL', 'MSFT'],
        help="Choose one or more stocks to analyze"
    )
    
    custom_ticker = st.text_input("Or enter custom ticker:", "").upper()
    if custom_ticker and custom_ticker not in selected_stocks:
        selected_stocks.append(custom_ticker)
    
    # Date Range
    st.markdown("### 📅 Time Period")
    date_range = st.selectbox(
        "Select Range",
        ["1W", "1M", "3M", "6M", "1Y", "5Y"],
        index=4
    )
    
    period_map = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "5Y": 1825}
    start_date = datetime.now() - timedelta(days=period_map[date_range])
    
    # Features Toggle
    st.markdown("### ⚙️ Features")
    show_sentiment = st.checkbox("Sentiment Analysis", value=True)
    show_forecast = st.checkbox("Price Forecast", value=True)
    show_portfolio = st.checkbox("Portfolio Metrics", value=True)
    
    # NewsAPI Key
    st.markdown("### 🔑 API Keys")
    news_api_key = st.text_input("NewsAPI Key (optional)", type="password", help="Get free key at newsapi.org")
    
    # Refresh Button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Helper Functions
# Helper Functions
@st.cache_data(ttl=300)
def fetch_stock_data(ticker, start):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start)
        return df, stock.info
    except Exception as e:
        st.error(f"Error fetching {ticker}: {str(e)}")
        return None, None

@st.cache_resource
def load_sentiment_model():
    """Load FinBERT sentiment model"""
    try:
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        return sentiment_pipeline
    except:
        # Fallback to distilbert
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def fetch_news(ticker, api_key):
    """Fetch news articles for a stock"""
    if not api_key:
        return []
    
    try:
        url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&pageSize=10&apiKey={api_key}"
        response = requests.get(url, timeout=5)
        data = response.json()
        return data.get('articles', [])
    except:
        return []

def analyze_sentiment(articles, sentiment_pipeline):
    """Analyze sentiment of news articles"""
    if not articles:
        return {"positive": 0, "negative": 0, "neutral": 0}, []
    
    sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    results = []
    
    for article in articles[:10]:
        text = f"{article.get('title', '')} {article.get('description', '')}"[:512]
        try:
            result = sentiment_pipeline(text)[0]
            label = result['label'].lower()
            
            if 'positive' in label or result['score'] > 0.6:
                sentiments['positive'] += 1
                sentiment_label = 'positive'
            elif 'negative' in label or result['score'] < 0.4:
                sentiments['negative'] += 1
                sentiment_label = 'negative'
            else:
                sentiments['neutral'] += 1
                sentiment_label = 'neutral'
            
            results.append({
                'title': article.get('title', ''),
                'sentiment': sentiment_label,
                'score': result['score']
            })
        except:
            continue
    
    return sentiments, results

def calculate_metrics(df):
    """Calculate stock metrics"""
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
    price_change = current_price - prev_price
    pct_change = (price_change / prev_price) * 100
    
    returns = df['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    
    daily_return = returns.mean()
    sharpe_ratio = (daily_return * 252) / (volatility) if volatility > 0 else 0
    
    return {
        'current_price': current_price,
        'price_change': price_change,
        'pct_change': pct_change,
        'volume': df['Volume'].iloc[-1],
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'daily_return': daily_return * 100
    }

def forecast_stock(df, ticker, days=30):
    """Forecast stock prices using Prophet - FIXED VERSION"""
    try:
        # Reset index and prepare data
        df_prophet = df.reset_index()
        
        # Ensure Date column exists and is timezone-naive
        if 'Date' in df_prophet.columns:
            df_prophet['Date'] = pd.to_datetime(df_prophet['Date']).dt.tz_localize(None)
        else:
            # If Date is the index, reset it
            df_prophet['Date'] = df_prophet.index
            df_prophet['Date'] = pd.to_datetime(df_prophet['Date']).dt.tz_localize(None)
        
        # Rename columns for Prophet (ds = datestamp, y = value)
        df_prophet = df_prophet[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        
        # Remove any NaN values
        df_prophet = df_prophet.dropna()
        
        # Ensure we have enough data points
        if len(df_prophet) < 30:
            st.warning(f"Not enough data for {ticker} forecast (need at least 30 days)")
            return None
        
        # Ensure ds column is proper datetime
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        
        # Create Prophet model with adjusted parameters
        model = Prophet(
            daily_seasonality=False,
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05,
            interval_width=0.95
        )
        
        # Suppress Prophet's verbose logging
        import logging
        logging.getLogger('prophet').setLevel(logging.ERROR)
        logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
        
        # Fit the model
        model.fit(df_prophet)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=days)
        
        # Make predictions
        forecast = model.predict(future)
        
        return forecast
        
    except Exception as e:
        st.warning(f"Forecast error for {ticker}: {str(e)}")
        # Uncomment below to see full error traceback during debugging
        # import traceback
        # st.error(traceback.format_exc())
        return None

# Main Application
if not selected_stocks:
    st.warning("⚠️ Please select at least one stock from the sidebar")
    st.stop()

# Load sentiment model
if show_sentiment and news_api_key:
    with st.spinner("Loading AI models..."):
        sentiment_pipeline = load_sentiment_model()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🧠 AI Insights", "💼 Portfolio", "⚙️ Advanced"])

with tab1:
    st.markdown("## Market Overview")
    
    for ticker in selected_stocks:
        with st.expander(f"📈 {ticker} Analysis", expanded=True):
            df, info = fetch_stock_data(ticker, start_date)
            
            if df is None or df.empty:
                st.error(f"Could not fetch data for {ticker}")
                continue
            
            metrics = calculate_metrics(df)
            
            # Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Current Price</div>
                    <div class="metric-value">${metrics['current_price']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                change_class = "positive" if metrics['pct_change'] >= 0 else "negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Change</div>
                    <div class="metric-value {change_class}">{metrics['pct_change']:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Volume</div>
                    <div class="metric-value">{metrics['volume']:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Volatility</div>
                    <div class="metric-value">{metrics['volatility']:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Price Chart - FIXED: Added unique key
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#00C896', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 200, 150, 0.1)'
            ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0F172A',
                plot_bgcolor='#1E293B',
                title=f'{ticker} Price Trend',
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"price_chart_{ticker}")
            
            # Volume Chart - FIXED: Added unique key
            fig_vol = go.Figure()
            colors = ['#10B981' if row['Close'] >= row['Open'] else '#EF4444' for _, row in df.iterrows()]
            fig_vol.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                marker_color=colors,
                name='Volume'
            ))
            
            fig_vol.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0F172A',
                plot_bgcolor='#1E293B',
                title=f'{ticker} Trading Volume',
                xaxis_title='Date',
                yaxis_title='Volume',
                height=300
            )
            
            st.plotly_chart(fig_vol, use_container_width=True, key=f"volume_chart_{ticker}")

with tab2:
    st.markdown("## 🧠 AI-Powered Insights")
    
    if show_sentiment and news_api_key:
        for ticker in selected_stocks:
            st.markdown(f"### {ticker} Sentiment Analysis")
            
            articles = fetch_news(ticker, news_api_key)
            
            if articles:
                sentiments, results = analyze_sentiment(articles, sentiment_pipeline)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Sentiment Pie Chart - FIXED: Added unique key
                    fig_sent = go.Figure(data=[go.Pie(
                        labels=['Positive', 'Neutral', 'Negative'],
                        values=[sentiments['positive'], sentiments['neutral'], sentiments['negative']],
                        marker=dict(colors=['#10B981', '#FCD34D', '#EF4444']),
                        hole=0.4
                    )])
                    
                    fig_sent.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='#0F172A',
                        title='News Sentiment Distribution',
                        height=300
                    )
                    
                    st.plotly_chart(fig_sent, use_container_width=True, key=f"sentiment_pie_{ticker}")
                
                with col2:
                    # AI Summary
                    total = sum(sentiments.values())
                    if total > 0:
                        pos_pct = (sentiments['positive'] / total) * 100
                        
                        if pos_pct > 60:
                            trend = "strongly positive"
                            emoji = "🚀"
                        elif pos_pct > 40:
                            trend = "mixed with cautious optimism"
                            emoji = "📊"
                        else:
                            trend = "bearish with concerns"
                            emoji = "⚠️"
                        
                        st.markdown(f"""
                        <div class="insight-box">
                            <h4>{emoji} AI Market Summary</h4>
                            <p style="font-size: 16px; line-height: 1.6;">
                            {ticker} is showing <strong>{trend}</strong> sentiment based on recent news analysis.
                            {sentiments['positive']} out of {total} articles show positive sentiment ({pos_pct:.1f}%).
                            Current market indicators suggest {'' if pos_pct > 50 else 'careful consideration for'} investment opportunities.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Recent Headlines
                st.markdown("#### Recent Headlines")
                for result in results[:5]:
                    sentiment_color = {'positive': '#10B981', 'neutral': '#FCD34D', 'negative': '#EF4444'}
                    st.markdown(f"""
                    <div style="padding: 10px; margin: 5px 0; background: #1E293B; border-radius: 8px; border-left: 3px solid {sentiment_color[result['sentiment']]};">
                        {result['title']} 
                        <span style="color: {sentiment_color[result['sentiment']]}; font-weight: 600;">
                            [{result['sentiment'].upper()}]
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"No recent news found for {ticker}")
    else:
        st.info("💡 Add your NewsAPI key in the sidebar to enable sentiment analysis")
    
    # Forecast Section
        # Forecast Section
    if show_forecast:
        st.markdown("## 📈 Predictive Forecasting")
        
        for ticker in selected_stocks:
            df, _ = fetch_stock_data(ticker, start_date)
            
            if df is not None and len(df) > 30:
                with st.spinner(f"Generating forecast for {ticker}..."):
                    forecast = forecast_stock(df, ticker, days=30)
                    
                    if forecast is not None:
                        fig_forecast = go.Figure()
                        
                        # Create a copy and ensure timezone-naive
                        df_plot = df.copy()
                        if hasattr(df_plot.index, 'tz') and df_plot.index.tz is not None:
                            df_plot.index = df_plot.index.tz_localize(None)
                        
                        # Historical data
                        fig_forecast.add_trace(go.Scatter(
                            x=df_plot.index,
                            y=df_plot['Close'],
                            mode='lines',
                            name='Historical',
                            line=dict(color='#00C896', width=2)
                        ))
                        
                        # Get the last date and convert to proper format
                        last_historical_date = pd.Timestamp(df_plot.index[-1])
                        
                        # Filter forecast for future dates only
                        # Ensure both are timezone-naive datetimes
                        forecast['ds'] = pd.to_datetime(forecast['ds']).dt.tz_localize(None)
                        future_dates = forecast[forecast['ds'] > last_historical_date]
                        
                        # Forecast line
                        fig_forecast.add_trace(go.Scatter(
                            x=future_dates['ds'],
                            y=future_dates['yhat'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='#FCD34D', width=2, dash='dash')
                        ))
                        
                        # Confidence interval - upper bound
                        fig_forecast.add_trace(go.Scatter(
                            x=future_dates['ds'],
                            y=future_dates['yhat_upper'],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        # Confidence interval - lower bound with fill
                        fig_forecast.add_trace(go.Scatter(
                            x=future_dates['ds'],
                            y=future_dates['yhat_lower'],
                            mode='lines',
                            line=dict(width=0),
                            fillcolor='rgba(252, 211, 77, 0.2)',
                            fill='tonexty',
                            name='Confidence Interval'
                        ))
                        
                        fig_forecast.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='#0F172A',
                            plot_bgcolor='#1E293B',
                            title=f'{ticker} 30-Day Price Forecast',
                            xaxis_title='Date',
                            yaxis_title='Price (USD)',
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig_forecast, use_container_width=True, key=f"forecast_chart_{ticker}")


with tab3:
    st.markdown("## 💼 Portfolio Management")
    
    # Add to Portfolio
    col1, col2 = st.columns([3, 1])
    with col1:
        portfolio_ticker = st.selectbox("Add Stock to Portfolio", selected_stocks)
    with col2:
        if st.button("➕ Add"):
            if portfolio_ticker not in st.session_state.portfolio:
                st.session_state.portfolio.append(portfolio_ticker)
                st.success(f"Added {portfolio_ticker}")
    
    if st.session_state.portfolio:
        st.markdown("### Your Portfolio")
        
        portfolio_data = []
        total_value = 0
        
        for ticker in st.session_state.portfolio:
            df, info = fetch_stock_data(ticker, start_date)
            if df is not None:
                metrics = calculate_metrics(df)
                portfolio_data.append({
                    'Ticker': ticker,
                    'Price': f"${metrics['current_price']:.2f}",
                    'Change %': f"{metrics['pct_change']:.2f}%",
                    'Volatility': f"{metrics['volatility']:.2%}",
                    'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
                    'Daily Return': f"{metrics['daily_return']:.2f}%"
                })
                total_value += metrics['current_price']
        
        if portfolio_data:
            df_portfolio = pd.DataFrame(portfolio_data)
            st.dataframe(df_portfolio, use_container_width=True)
            
            # Portfolio Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Portfolio Size", f"{len(st.session_state.portfolio)} stocks")
            with col2:
                st.metric("Total Value", f"${total_value:.2f}")
            with col3:
                avg_return = df_portfolio['Daily Return'].str.rstrip('%').astype(float).mean()
                st.metric("Avg Daily Return", f"{avg_return:.2f}%")
        
        if st.button("🗑️ Clear Portfolio"):
            st.session_state.portfolio = []
            st.rerun()
    else:
        st.info("Your portfolio is empty. Add stocks from the dropdown above.")
    
    # Investment Simulator
    st.markdown("### 🎯 What-If Investment Simulator")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        sim_ticker = st.selectbox("Select Stock", selected_stocks, key='sim')
    with col2:
        investment = st.number_input("Investment Amount ($)", min_value=100, value=10000, step=100)
    with col3:
        years_ago = st.selectbox("Years Ago", [1, 2, 3, 5], index=0)
    
    if st.button("Calculate ROI"):
        past_date = datetime.now() - timedelta(days=years_ago*365)
        df_sim, _ = fetch_stock_data(sim_ticker, past_date)
        
        if df_sim is not None and len(df_sim) > 0:
            initial_price = df_sim['Close'].iloc[0]
            current_price = df_sim['Close'].iloc[-1]
            
            shares = investment / initial_price
            current_value = shares * current_price
            profit = current_value - investment
            roi = (profit / investment) * 100
            cagr = (((current_value / investment) ** (1/years_ago)) - 1) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Value", f"${current_value:.2f}", f"+${profit:.2f}")
            with col2:
                st.metric("ROI", f"{roi:.2f}%")
            with col3:
                st.metric("CAGR", f"{cagr:.2f}%")
            
            # Growth Chart - FIXED: Added unique key
            growth = (df_sim['Close'] / initial_price) * investment
            fig_growth = go.Figure()
            fig_growth.add_trace(go.Scatter(
                x=df_sim.index,
                y=growth,
                mode='lines',
                fill='tozeroy',
                line=dict(color='#00C896', width=2),
                fillcolor='rgba(0, 200, 150, 0.2)'
            ))
            
            fig_growth.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0F172A',
                plot_bgcolor='#1E293B',
                title='Investment Growth Over Time',
                xaxis_title='Date',
                yaxis_title='Portfolio Value (USD)',
                height=400
            )
            
            st.plotly_chart(fig_growth, use_container_width=True, key=f"growth_chart_{sim_ticker}_{years_ago}")

with tab4:
    st.markdown("## ⚙️ Advanced Features")
    
    # Multi-Stock Comparison
    if len(selected_stocks) > 1:
        st.markdown("### 📊 Multi-Stock Comparison")
        
        comparison_data = {}
        for ticker in selected_stocks:
            df, _ = fetch_stock_data(ticker, start_date)
            if df is not None:
                # Normalize to percentage change
                df['Normalized'] = (df['Close'] / df['Close'].iloc[0] - 1) * 100
                comparison_data[ticker] = df['Normalized']
        
        if comparison_data:
            fig_comp = go.Figure()
            for ticker, data in comparison_data.items():
                fig_comp.add_trace(go.Scatter(
                    x=data.index,
                    y=data,
                    mode='lines',
                    name=ticker,
                    line=dict(width=2)
                ))
            
            fig_comp.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0F172A',
                plot_bgcolor='#1E293B',
                title='Normalized Returns Comparison',
                xaxis_title='Date',
                yaxis_title='Return (%)',
                hovermode='x unified',
                height=500
            )
            
            # FIXED: Added unique key
            st.plotly_chart(fig_comp, use_container_width=True, key="comparison_chart")
    
    # Data Export
    st.markdown("### 💾 Export Data")
    
    export_ticker = st.selectbox("Select Stock for Export", selected_stocks, key='export')
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download CSV"):
            df_export, _ = fetch_stock_data(export_ticker, start_date)
            if df_export is not None:
                csv = df_export.to_csv()
                st.download_button(
                    label="Download",
                    data=csv,
                    file_name=f"{export_ticker}_data.csv",
                    mime="text/csv"
                )
    
    with col2:
        if st.button("📊 Download Metrics"):
            metrics_data = []
            for ticker in selected_stocks:
                df, _ = fetch_stock_data(ticker, start_date)
                if df is not None:
                    metrics = calculate_metrics(df)
                    metrics['Ticker'] = ticker
                    metrics_data.append(metrics)
            
            if metrics_data:
                df_metrics = pd.DataFrame(metrics_data)
                csv = df_metrics.to_csv(index=False)
                st.download_button(
                    label="Download",
                    data=csv,
                    file_name="portfolio_metrics.csv",
                    mime="text/csv"
                )
    
    # Technical Indicators
    st.markdown("### 📈 Technical Indicators")
    
    tech_ticker = st.selectbox("Select Stock", selected_stocks, key='tech')
    df_tech, _ = fetch_stock_data(tech_ticker, start_date)
    
    if df_tech is not None:
        # Moving Averages
        df_tech['MA20'] = df_tech['Close'].rolling(window=20).mean()
        df_tech['MA50'] = df_tech['Close'].rolling(window=50).mean()
        
        fig_tech = go.Figure()
        fig_tech.add_trace(go.Scatter(
            x=df_tech.index, y=df_tech['Close'],
            mode='lines', name='Close', line=dict(color='#00C896', width=2)
        ))
        fig_tech.add_trace(go.Scatter(
            x=df_tech.index, y=df_tech['MA20'],
            mode='lines', name='MA20', line=dict(color='#FCD34D', width=1, dash='dash')
        ))
        fig_tech.add_trace(go.Scatter(
            x=df_tech.index, y=df_tech['MA50'],
            mode='lines', name='MA50', line=dict(color='#EF4444', width=1, dash='dash')
        ))
        
        fig_tech.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0F172A',
            plot_bgcolor='#1E293B',
            title=f'{tech_ticker} Technical Analysis',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=500
        )
        
        # FIXED: Added unique key
        st.plotly_chart(fig_tech, use_container_width=True, key=f"technical_chart_{tech_ticker}")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <strong>📈 AI-Powered Stock Market Dashboard</strong><br>
    Built with Streamlit • Powered by yfinance, Hugging Face & Prophet<br>
    Made for recruiters and investors • Data refreshes every 5 minutes
</div>
""", unsafe_allow_html=True)
