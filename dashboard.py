import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from datetime import datetime, timedelta
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
import google.generativeai as genai
import json
import os
import joblib
import ta
from streamlit_autorefresh import st_autorefresh
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_sortables import sort_items
import requests
from GoogleNews import GoogleNews
from yahooquery import Ticker
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from supabase import create_client, Client
from extra_streamlit_components import CookieManager
import time
from streamlit_lottie import st_lottie

st.set_page_config(
    page_title="Stock Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize CookieManager in session state if not present
if "cookie_manager" not in st.session_state:
    st.session_state.cookie_manager = CookieManager()

st.markdown("""
<style>
    body {
        background-color: #ffffff;
        color: #000000;
    }
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stPlotlyChart {
        border-radius: 10px;
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------
# Constants & Configuration
# ---------------------
genai.configure(api_key="AIzaSyCMCH3o8F-vIbGK3fdy2JJTZv2gVtHN6_E")  # <-- Replace with your valid key
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

SUPABASE_URL = "https://qjhuloguigjgtlqysjpz.supabase.co"  # Replace with your project URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFqaHVsb2d1aWdqZ3RscXlzanB6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDI2NDkyNDAsImV4cCI6MjA1ODIyNTI0MH0.I1gaQtPHw3I9GVejPtplQmwIdYDqPYZuFA4lpOSGNWM"  # Replace with your anon key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------
# Helper Functions for Supabase Authentication & Portfolio Management
# ---------------------
def login_form():
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Login"):
        try:
            # First get email associated with username
            profile_response = supabase.table("profiles") \
                .select("email") \
                .eq("username", username) \
                .execute()
            
            if not profile_response.data:
                st.error("Username not found")
                return
                
            user_email = profile_response.data[0]['email']
            
            # Authenticate with email from profile
            response = supabase.auth.sign_in_with_password({
                "email": user_email,
                "password": password
            })
            
            if response.user:
                st.session_state.user = response.user
                st.session_state.cookie_manager.set(
                    'supabase_refresh_token',
                    response.session.refresh_token,
                    expires_at=datetime.fromtimestamp(time.time() + 604800)
                )
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials")
                
        except Exception as e:
            st.error(f"Login error: {str(e)}")

def signup_form():
    st.subheader("Sign Up")
    username = st.text_input("Username", key="signup_username")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_password")
    confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
    
    if st.button("Create Account"):
        if password != confirm:
            st.error("Passwords do not match!")
        elif not username:
            st.error("Username is required!")
        else:
            try:
                # Check if username already exists
                existing_user = supabase.table("profiles") \
                    .select("username") \
                    .eq("username", username) \
                    .execute()
                
                if existing_user.data:
                    st.error("Username already taken")
                    return
                    
                # Create auth user
                auth_response = supabase.auth.sign_up({
                    "email": email,
                    "password": password
                })
                
                if auth_response.user:
                    # Create profile with username
                    profile_response = supabase.table("profiles").insert({
                        "id": auth_response.user.id,
                        "email": email,
                        "username": username
                    }).execute()
                    
                    if profile_response.data:
                        st.success("Account created! Please login.")
                        st.session_state.show_login = True
                        st.rerun()
                    else:
                        st.error("Profile creation failed")
                        
            except Exception as e:
                st.error(f"Signup error: {str(e)}")

def logout():
    if st.button("Logout"):
        try:
            # Clear cookie and session
            st.session_state.cookie_manager.delete('supabase_refresh_token')
            supabase.auth.sign_out()
        finally:
            st.session_state.clear()
            st.success("Logged out successfully!")
            st.rerun()

def load_portfolio():
    """Load the logged-in user's portfolio from Supabase."""
    user_id = st.session_state.user.id  # Using attribute notation
    response = supabase.table("portfolio").select("*").eq("user_id", user_id).execute()
    data = response.data
    portfolio = {}
    if data:
        for row in data:
            portfolio[row["stock_symbol"]] = row["shares"]
    return portfolio

def add_to_portfolio(stock_symbol, shares):
    """Add or update a portfolio record in Supabase."""
    user_id = st.session_state.user.id
    response = supabase.table("portfolio").select("*").eq("user_id", user_id).eq("stock_symbol", stock_symbol).execute()
    if response.data:
        # Update existing record
        current = response.data[0]["shares"]
        new_total = current + shares
        supabase.table("portfolio").update({"shares": new_total}).eq("id", response.data[0]["id"]).execute()
    else:
        # Insert new record; use current market price as buying_price if available.
        try:
            current_price = yf.Ticker(stock_symbol).history(period="1d")['Close'].iloc[-1]
        except Exception:
            current_price = 0
        supabase.table("portfolio").insert({
            "user_id": user_id,
            "stock_symbol": stock_symbol,
            "shares": shares,
            "buying_price": current_price
        }).execute()

def remove_from_portfolio(stock_symbol):
    """Remove a stock from the user's portfolio in Supabase."""
    user_id = st.session_state.user.id
    supabase.table("portfolio").delete().eq("user_id", user_id).eq("stock_symbol", stock_symbol).execute()

STOCK_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "HCLTECH.NS", "MARUTI.NS",
    "WIPRO.NS", "ONGC.NS", "ULTRACEMCO.NS", "TITAN.NS", "ADANIENT.NS",
    "NTPC.NS", "POWERGRID.NS", "SUNPHARMA.NS"
]

SECTOR_SYMBOLS = {
    "NIFTY IT": "^CNXIT",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY FIN SERVICE": "^NIFTY_FIN_SERVICE.NS",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY MEDIA": "^CNXMEDIA",
    "NIFTY PVT BANK": "^CNXPVTBNK",
}

INDICES = {
    "NIFTY 50": "^NSEI",
    "NIFTY INFRA": "^CNXINFRA",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY ENERGY": "^CNXENERGY"
}

# Initialize additional session state defaults
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if "active_mode" not in st.session_state:
    st.session_state.active_mode = "dashboard"
if "messages" not in st.session_state:
    st.session_state.messages = {}

def apply_theme():
    theme = "dark" if st.session_state.dark_mode else "light"
    bg_color = "#0e1117" if theme == "dark" else "#ffffff"
    text_color = "#ffffff" if theme == "dark" else "#000000"
    secondary_bg = "#1f1f1f" if theme == "dark" else "#f0f2f6"
    st.markdown(f"""
    <style>
        body {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .metric-card {{
            padding: 20px;
            border-radius: 10px;
            background-color: {secondary_bg};
            margin: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .insight-card {{
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid;
            background-color: {secondary_bg};
        }}
        .stPlotlyChart {{
            border-radius: 10px;
            background-color: {bg_color};
        }}
        .js-plotly-plot .plotly, .js-plotly-plot .plotly .modebar {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
        }}
        input, textarea, .stTextInput input {{
            color: {text_color} !important;
            background-color: {bg_color} !important;
            border: 1px solid #ccc;
        }}
    </style>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, change=None):
    if change is not None:
        try:
            change_value = float(str(change).replace('%', ''))
            color = "green" if change_value >= 0 else "red"
            change_html = f"<p style='color:{color}; font-size:14px;'>{change}%</p>"
        except:
            color = "gray"
            change_html = f"<p style='color:{color}; font-size:14px;'>N/A</p>"
    else:
        change_html = ""
    card_html = f"""
    <div class='metric-card'>
        <h3>{title}</h3>
        <h2>{value}</h2>
        {change_html}
    </div>
    """
    return card_html

# ---------------------
# Data Retrieval & Processing Functions
# ---------------------
def fix_yf_data_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    # Rename columns to standard names
    rename_dict = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'open' in col_lower:
            rename_dict[col] = 'Open'
        elif 'high' in col_lower:
            rename_dict[col] = 'High'
        elif 'low' in col_lower:
            rename_dict[col] = 'Low'
        elif 'close' in col_lower and 'adj' not in col_lower:
            rename_dict[col] = 'Close'
        elif 'volume' in col_lower:
            rename_dict[col] = 'Volume'
    df.rename(columns=rename_dict, inplace=True)
    # Ensure each column is one-dimensional
    for col in df.columns:
        if len(df[col].shape) > 1 and df[col].shape[1] == 1:
            df[col] = df[col].iloc[:, 0]
    return df

@st.cache_data(ttl=300)
def get_historical_data(symbol, period="1y", interval="1d", angelone_api_key="JQvcKLzK"):
    intraday_intervals = ["5m", "15m", "30m", "1H"]
    if interval in intraday_intervals:
        period = "60d"
    try:
        data = yf.download(symbol, period=period, interval=interval)
    except Exception as e:
        st.error(f"yfinance error: {e}")
        data = pd.DataFrame()
    if data.empty and interval in ["5m", "15m", "30m"]:
        st.warning(f"yfinance returned no data for {symbol} at {interval}. Falling back to AngelOne.")
        base_symbol = symbol.split('.')[0]
        data = fetch_intraday_angelone(base_symbol, interval=interval, token=angelone_api_key, days=2)
    if data.empty:
        return data
    data = fix_yf_data_columns(data)
    data = add_technical_indicators(data)
    return data

def fetch_intraday_angelone(symbol: str, interval: str = "5m", token: str = "JQvcKLzK", days: int = 4) -> pd.DataFrame:
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)
    interval_map = {"5m": 5, "15m": 15}
    interval_minutes = interval_map.get(interval, 5)
    base_url = "https://apim.angelbroking.com/historical/NSE"
    endpoint = f"{base_url}/{symbol}?from={from_date.strftime('%Y-%m-%d')}&to={to_date.strftime('%Y-%m-%d')}&interval={interval_minutes}"
    headers = {
        "Authorization": token,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    try:
        resp = requests.get(endpoint, headers=headers, timeout=10)
        resp.raise_for_status()
        data_json = resp.json()
        candles = data_json.get("data", [])
        if not candles:
            st.warning(f"No intraday data returned from AngelOne for {symbol} ({interval}).")
            return pd.DataFrame()
        records = []
        for c in candles:
            ts = c.get("timestamp")
            dt = datetime.fromtimestamp(ts) if ts else None
            records.append({
                "Date": dt,
                "Open": c.get("open"),
                "High": c.get("high"),
                "Low":  c.get("low"),
                "Close": c.get("close"),
                "Volume": c.get("volume", 0)
            })
        df = pd.DataFrame(records)
        if not df.empty:
            df.set_index("Date", inplace=True)
        return df
    except Exception as e:
        st.error(f"AngelOne API error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_indices_and_sectors():
    results = []
    for idx_name, ticker in INDICES.items():
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[0]
                change = ((current - prev) / prev) * 100
                results.append({
                    "Name": idx_name,
                    "Price": current,
                    "Change (%)": change
                })
        except Exception as e:
            st.error(f"Error fetching {idx_name}: {str(e)}")
    for sector_name, ticker in SECTOR_SYMBOLS.items():
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[0]
                change = ((current - prev) / prev) * 100
                results.append({
                    "Name": sector_name,
                    "Price": current,
                    "Change (%)": change
                })
        except Exception as e:
            st.error(f"Error fetching {sector_name}: {str(e)}")
    return pd.DataFrame(results)

def add_technical_indicators(data):
    try:
        data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], 20).ema_indicator()
        data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], 50).ema_indicator()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        bollinger = ta.volatility.BollingerBands(data['Close'])
        data['bb_high'] = bollinger.bollinger_hband()
        data['bb_low'] = bollinger.bollinger_lband()
        data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
        data['VWAP'] = ta.volume.VolumeWeightedAveragePrice(data['High'], data['Low'], data['Close'], data['Volume']).volume_weighted_average_price()
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
        data['rsi'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        macd = ta.trend.MACD(data['Close'])
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        return data
    except Exception as e:
        st.error(f"Indicator error: {str(e)}")
        return data

def detect_candlestick_patterns(data: pd.DataFrame):
    df = data.copy()
    df['body'] = abs(df['Close'] - df['Open'])
    avg_body = df['body'].rolling(14).mean().iloc[-1] if len(df) >= 14 else df['body'].mean()
    df['doji'] = (df['body'] < (0.1 * (df['High'] - df['Low'])))
    df['hammer'] = (((df['Low'] - df[['Open', 'Close']].min(axis=1)) > (2 * df['body'])) & (df['body'] > avg_body))
    df['shooting_star'] = (((df[['Open', 'Close']].max(axis=1) - df['High']) > (2 * df['body'])) & (df['body'] > avg_body))
    df['bullish_engulfing'] = ((df['Close'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1)))
    df['bearish_engulfing'] = ((df['Close'] < df['Open'].shift(1)) & (df['Open'] > df['Close'].shift(1)))
    return df

def generate_technical_insights(df: pd.DataFrame):
    insights = []
    if df.empty:
        return insights
    latest = df.iloc[-1]
    if latest.get('doji', False):
        insights.append(('Doji', 'Potential reversal signal', '1-3 days', 'yellow'))
    if latest.get('hammer', False):
        insights.append(('Hammer', 'Bullish reversal signal', '3-5 days', 'green'))
    if latest.get('shooting_star', False):
        insights.append(('Shooting Star', 'Bearish reversal signal', '3-5 days', 'red'))
    if latest.get('bullish_engulfing', False):
        insights.append(('Bullish Engulfing', 'Strong bullish momentum', '1-2 weeks', 'darkgreen'))
    if latest.get('bearish_engulfing', False):
        insights.append(('Bearish Engulfing', 'Strong bearish momentum', '1-2 weeks', 'darkred'))
    rsi_val = latest.get('rsi', None)
    if rsi_val is not None:
        if rsi_val > 70:
            insights.append(('Overbought', f'RSI {rsi_val:.1f} - Potential pullback', '1-5 days', 'orange'))
        elif rsi_val < 30:
            insights.append(('Oversold', f'RSI {rsi_val:.1f} - Potential rebound', '1-5 days', 'blue'))
    return insights

def plot_technical_analysis(data, symbol):
    if data.empty:
        st.warning(f"No data available for {symbol} at the selected timeframe.")
        return
    if data.index.name is None:
        data.index.name = "Date"
    df = data.reset_index()
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='green',
        decreasing_line_color='red'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_20'], mode='lines', name='EMA 20', line=dict(color='blue', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_50'], mode='lines', name='EMA 50', line=dict(color='orange', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='purple', dash='dot', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='brown', dash='dot', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['bb_high'], mode='lines', name='BB High', line=dict(color='grey', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['bb_low'], mode='lines', name='BB Low', line=dict(color='grey', width=1)), row=1, col=1)
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='lightgrey'), row=2, col=1)
    fig.update_layout(
        title=f"{symbol} Technical Analysis",
        xaxis=dict(rangeslider=dict(visible=False)),
        template='plotly_dark' if st.session_state.dark_mode else 'plotly_white',
        height=700,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    if 'rsi' in df.columns and not df['rsi'].empty:
        rsi_val = df['rsi'].iloc[-1]
        if rsi_val > 70:
            conclusion = "Overbought conditions. Potential pullback."
        elif rsi_val < 30:
            conclusion = "Oversold conditions. Potential rebound."
        else:
            conclusion = "RSI in neutral range."
    else:
        conclusion = "RSI data unavailable."
    st.markdown(f"**Technical Conclusion:** {conclusion}")

def create_sequences(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i+look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

def show_prediction_Prophet(symbol, forecast_days=30):
    # Create folder for saving model/scaler if not exists
    os.makedirs(f"saved_models/{symbol}", exist_ok=True)
    model_path = f"saved_models/{symbol}/prophet_model_{forecast_days}d.json"
    scaler_path = f"saved_models/{symbol}/prophet_scaler_{forecast_days}d.pkl"

    # Read CSV and perform initial cleaning
    df = pd.read_csv(f"{symbol}.CSV", parse_dates=['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
    
    # Compute technical indicators
    df['EMA_50'] = df['Price'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Price'].ewm(span=200, adjust=False).mean()
    window = 14
    delta = df['Price'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    df['RSI'] = 100 - (100 / (1 + (pd.Series(gain).rolling(window).mean() / pd.Series(loss).rolling(window).mean())))
    df['MACD'] = df['Price'].ewm(span=12, adjust=False).mean() - df['Price'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Define columns to normalize
    cols_to_norm = ['Price', 'EMA_50', 'EMA_200', 'RSI', 'MACD', 'MACD_Signal']
    
    retrain = False
    # Check if saved scaler and model exist
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        # If the scaler has feature names, ensure they match our current features
        if hasattr(scaler, "feature_names_in_"):
            if not np.array_equal(scaler.feature_names_in_, np.array(cols_to_norm)):
                retrain = True
        else:
            # If no feature_names_in_ attribute exists, force retrain to be safe
            retrain = True
        
        if not retrain:
            with open(model_path, 'r') as fin:
                prophet_model = model_from_json(json.load(fin))
            # Transform the data using the loaded scaler
            df[cols_to_norm] = scaler.transform(df[cols_to_norm])
            price_mean = scaler.mean_[0]
            price_std = scaler.scale_[0]
            df_prophet = pd.DataFrame({
                'ds': df.index, 
                'y': (df['Price'] * price_std) + price_mean
            }).sort_values(by='ds')
        else:
            # Refit scaler and model if feature names mismatch
            scaler = StandardScaler()
            df[cols_to_norm] = scaler.fit_transform(df[cols_to_norm])
            price_mean = scaler.mean_[0]
            price_std = scaler.scale_[0]
            df_prophet = pd.DataFrame({
                'ds': df.index, 
                'y': (df['Price'] * price_std) + price_mean
            }).sort_values(by='ds')
            prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            prophet_model.fit(df_prophet)
            with open(model_path, 'w') as fout:
                json.dump(model_to_json(prophet_model), fout)
            joblib.dump(scaler, scaler_path)
    else:
        # Train new model and scaler if not found
        scaler = StandardScaler()
        df[cols_to_norm] = scaler.fit_transform(df[cols_to_norm])
        price_mean = scaler.mean_[0]
        price_std = scaler.scale_[0]
        df_prophet = pd.DataFrame({
            'ds': df.index, 
            'y': (df['Price'] * price_std) + price_mean
        }).sort_values(by='ds')
        prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        prophet_model.fit(df_prophet)
        with open(model_path, 'w') as fout:
            json.dump(model_to_json(prophet_model), fout)
        joblib.dump(scaler, scaler_path)
    
    # Get the current price (from the reconstructed series)
    current_price = df_prophet['y'].iloc[-1]
    
    # Forecast exactly forecast_days into the future
    future = prophet_model.make_future_dataframe(periods=forecast_days, freq='D')
    forecast = prophet_model.predict(future)
    # Extract only the forecast period (after the last actual date)
    last_date = df_prophet['ds'].iloc[-1]
    forecast_period = forecast[forecast['ds'] > last_date].copy().tail(forecast_days)
    predicted_price = forecast_period['yhat'].iloc[-1]
    
    # Optional: Compute training metrics
    train_forecast = prophet_model.predict(df_prophet)
    mse = mean_squared_error(df_prophet['y'], train_forecast['yhat'])
    rmse = math.sqrt(mse)
    r2 = r2_score(df_prophet['y'], train_forecast['yhat'])

    # Compute percentage change and direction
    price_diff = predicted_price - current_price
    pct_diff = (price_diff / current_price) * 100
    direction = "increase" if pct_diff > 0 else "decrease"
    
    # Display summary info (like your reference image)
    st.markdown(f"### {symbol} - {forecast_days}-day Forecast")
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric(label="Current Price", value=f"{current_price:.2f}")
    with colB:
        st.metric(label="Predicted Price", value=f"{predicted_price:.2f}")
    with colC:
        sign_str = "+" if pct_diff > 0 else ""
        st.metric(label="Movement (%)", value=f"{sign_str}{pct_diff:.2f}%")
    
    # Provide a short text statement summarizing the forecast
    movement_type = "upside" if direction == "increase" else "downside"
    st.write(
        f"Our AI model predicts a possible **{movement_type}** movement of "
        f"**{abs(pct_diff):.2f}%** over the next **{forecast_days} days**. "
        "This forecast is based on historical data, market trends, and sentiment analysis."
    )
    
    # Full plot: actual price, forecasted price, forecasted high/low, EMA 50, and EMA 200
    forecast_period.rename(columns={
            'yhat': 'Forecasted_Price',
            'yhat_lower': 'Forecasted_Low',
            'yhat_upper': 'Forecasted_High'
        }, inplace=True)
    forecast_period.set_index('ds', inplace=True)
    df_plot = pd.DataFrame({
        'Actual_Price': (df['Price'] * price_std) + price_mean,
        'EMA_50': (df['EMA_50'] * price_std) + price_mean,
        'EMA_200': (df['EMA_200'] * price_std) + price_mean,
    }, index=df.index)
    combined_df = pd.concat([
        df_plot,
        forecast_period[['Forecasted_Price', 'Forecasted_Low', 'Forecasted_High']]
    ], axis=1)
    
    # ---------------------------
    # Detect Golden & Death Cross
    # ---------------------------
    # Golden Cross: EMA_50 crosses above EMA_200
    # Death Cross:  EMA_50 crosses below EMA_200
    df_plot['GoldenCross'] = (
        (df_plot['EMA_50'] > df_plot['EMA_200']) &
        (df_plot['EMA_50'].shift(1) <= df_plot['EMA_200'].shift(1))
    )
    df_plot['DeathCross'] = (
        (df_plot['EMA_50'] < df_plot['EMA_200']) &
        (df_plot['EMA_50'].shift(1) >= df_plot['EMA_200'].shift(1))
    )
    golden_points = df_plot[df_plot['GoldenCross'] == True]
    death_points  = df_plot[df_plot['DeathCross'] == True]
    
    fig = go.Figure()
    # Plot actual price
    fig.add_trace(go.Scatter(
        x=combined_df.index,
        y=combined_df['Actual_Price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='blue')
    ))
    # Plot forecast
    fig.add_trace(go.Scatter(
        x=combined_df.index,
        y=combined_df['Forecasted_Price'],
        mode='lines',
        name='Forecasted Price',
        line=dict(color='red', dash='dot')
    ))
    # Forecasted high/low
    fig.add_trace(go.Scatter(
        x=combined_df.index,
        y=combined_df['Forecasted_High'],
        fill=None,
        mode='lines',
        line_color='lightgray',
        name='Forecasted High'
    ))
    fig.add_trace(go.Scatter(
        x=combined_df.index,
        y=combined_df['Forecasted_Low'],
        fill='tonexty',
        mode='lines',
        line_color='lightgray',
        name='Forecasted Low'
    ))
    # EMA lines
    fig.add_trace(go.Scatter(
        x=combined_df.index,
        y=combined_df['EMA_50'],
        mode='lines',
        name='EMA 50',
        line=dict(color='orange')
    ))
    fig.add_trace(go.Scatter(
        x=combined_df.index,
        y=combined_df['EMA_200'],
        mode='lines',
        name='EMA 200',
        line=dict(color='lightgray')
    ))
    # Markers for Golden & Death Cross
    fig.add_trace(go.Scatter(
        x=golden_points.index,
        y=golden_points['Actual_Price'],
        mode='markers',
        name='Golden Cross',
        marker=dict(color='green', size=10, symbol='triangle-up')
    ))
    fig.add_trace(go.Scatter(
        x=death_points.index,
        y=death_points['Actual_Price'],
        mode='markers',
        name='Death Cross',
        marker=dict(color='red', size=10, symbol='triangle-down')
    ))
    
    # Annotation for final forecasted price
    forecast_valid = combined_df[['Forecasted_Price', 'Forecasted_Low', 'Forecasted_High']].dropna()
    if not forecast_valid.empty:
        last_forecast_date = forecast_valid.index[-1]
        last_forecast_price = forecast_valid['Forecasted_Price'].iloc[-1]
        annotation_text = f"Predicted: {last_forecast_price:.2f}"
        fig.add_annotation(
            x=last_forecast_date,
            y=last_forecast_price,
            text=annotation_text,
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            font=dict(color="red", size=12),
            bgcolor="white",
            bordercolor="red"
        )
    
    fig.update_layout(
        title=f"{symbol} Stock Price: Prophet Forecast for {forecast_days} Days with EMA & Crossovers",
        xaxis_title='Date',
        yaxis_title='Stock Price (INR)',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    return mse, rmse, r2, forecast_period

def show_prediction_LSTM(symbol, forecast_days=30):
    # Hyperparameters
    lookback = 60      # Number of past days to use as input sequence
    max_epochs = 100   # Maximum number of epochs; early stopping will pick the best one

    # Ensure model folder exists
    os.makedirs(f"saved_models/{symbol}", exist_ok=True)
    model_path = f"saved_models/{symbol}/lstm_model_{forecast_days}d.h5"
    
    # Load and preprocess the CSV data
    df = pd.read_csv(f"{symbol}.CSV", parse_dates=['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # Clean Price and store original price
    df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
    df['Price_orig'] = df['Price'].copy()

    # Compute EMA overlays on the original price
    df['EMA_50'] = df['Price_orig'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Price_orig'].ewm(span=200, adjust=False).mean()

    # Scale the Price column for training the model
    scaler = StandardScaler()
    df[['Price']] = scaler.fit_transform(df[['Price']])
    price_mean = scaler.mean_[0]
    price_std = scaler.scale_[0]

    # Prepare data for the neural network using create_sequences
    data = df['Price'].values
    X, y = create_sequences(data, lookback)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split into training and validation sets (80/20 split)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Callbacks: EarlyStopping picks best epoch, ModelCheckpoint saves best model,
    # and ReduceLROnPlateau lowers the learning rate when progress stalls.
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    ]

    # Load or train the LSTM model
    if os.path.exists(model_path):
        lstm_model = tf.keras.models.load_model(model_path)
    else:
        lstm_model = Sequential()
        lstm_model.add(LSTM(50, activation='tanh', return_sequences=True, input_shape=(lookback, 1)))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(LSTM(50, activation='tanh'))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mse')
        
        history = lstm_model.fit(
            X_train, y_train,
            epochs=max_epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        lstm_model.save(model_path)

    # Evaluate model on validation set
    val_predictions = lstm_model.predict(X_val)
    val_predictions_denorm = val_predictions * price_std + price_mean
    y_val_denorm = y_val * price_std + price_mean
    rmse = np.sqrt(np.mean((val_predictions_denorm - y_val_denorm) ** 2))
    st.write(f"Validation RMSE: {rmse:.2f}")

    # Forecast iteratively using the last lookback values
    current_price = df['Price_orig'].iloc[-1]
    last_sequence = data[-lookback:]
    forecast_lstm = []
    current_seq = last_sequence.copy()
    for i in range(forecast_days):
        current_seq_reshaped = current_seq.reshape(1, lookback, 1)
        pred = lstm_model.predict(current_seq_reshaped)
        forecast_lstm.append(pred[0, 0])
        current_seq = np.append(current_seq[1:], pred[0, 0])
    forecast_lstm_denorm = np.array(forecast_lstm) * price_std + price_mean
    predicted_price = forecast_lstm_denorm[-1]

    # Compute percentage change and determine direction
    price_diff = predicted_price - current_price
    pct_diff = (price_diff / current_price) * 100
    direction = "increase" if pct_diff > 0 else "decrease"

    # --- Display summary panel ---
    st.markdown(f"### {symbol} - {forecast_days}-day LSTM Forecast")
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric(label="Current Price", value=f"{current_price:.2f}")
    with colB:
        st.metric(label="Predicted Price", value=f"{predicted_price:.2f}")
    with colC:
        sign_str = "+" if pct_diff > 0 else ""
        st.metric(label="Movement (%)", value=f"{sign_str}{pct_diff:.2f}%")
    movement_type = "upside" if direction == "increase" else "downside"
    st.write(
        f"Our AI LSTM model predicts a potential **{movement_type}** movement of "
        f"**{abs(pct_diff):.2f}%** over the next **{forecast_days} days**."
    )

    # Build forecast DataFrame for plotting
    forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_days+1, freq='D')[1:]
    forecast_df = pd.DataFrame({'Forecasted_Price': forecast_lstm_denorm}, index=forecast_dates)

    # Historical plot data with EMA overlays
    df_plot = pd.DataFrame({
        'Actual_Price': df['Price_orig'],
        'EMA_50': df['EMA_50'],
        'EMA_200': df['EMA_200']
    }, index=df.index)

    # Detect Golden Cross & Death Cross from historical data
    df_plot['GoldenCross'] = (
        (df_plot['EMA_50'] > df_plot['EMA_200']) &
        (df_plot['EMA_50'].shift(1) <= df_plot['EMA_200'].shift(1))
    )
    df_plot['DeathCross'] = (
        (df_plot['EMA_50'] < df_plot['EMA_200']) &
        (df_plot['EMA_50'].shift(1) >= df_plot['EMA_200'].shift(1))
    )
    golden_points = df_plot[df_plot['GoldenCross']]
    death_points  = df_plot[df_plot['DeathCross']]

    # Create full plot
    fig = go.Figure()
    # Plot Actual Price
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot['Actual_Price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='blue')
    ))
    # Plot Forecasted Price
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['Forecasted_Price'],
        mode='lines',
        name='Forecasted Price',
        line=dict(color='red', dash='dot')
    ))
    # Plot EMA overlays
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot['EMA_50'],
        mode='lines',
        name='EMA 50',
        line=dict(color='orange')
    ))
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot['EMA_200'],
        mode='lines',
        name='EMA 200',
        line=dict(color='lightgray')
    ))
    # Plot Golden Cross markers
    fig.add_trace(go.Scatter(
        x=golden_points.index,
        y=golden_points['Actual_Price'],
        mode='markers',
        name='Golden Cross',
        marker=dict(color='green', size=10, symbol='triangle-up')
    ))
    # Plot Death Cross markers
    fig.add_trace(go.Scatter(
        x=death_points.index,
        y=death_points['Actual_Price'],
        mode='markers',
        name='Death Cross',
        marker=dict(color='red', size=10, symbol='triangle-down')
    ))
    # Annotate final forecasted price
    if not forecast_df.empty:
        fig.add_annotation(
            x=forecast_df.index[-1],
            y=forecast_df['Forecasted_Price'].iloc[-1],
            text=f"Predicted: {forecast_df['Forecasted_Price'].iloc[-1]:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            font=dict(color="red", size=12),
            bgcolor="white",
            bordercolor="red"
        )
    fig.update_layout(
        title=f"{symbol} LSTM Forecast for {forecast_days} Days with EMA & Crossovers",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    return rmse, forecast_df

def show_prediction_GRU(symbol, forecast_days=30, lookback=60, epochs=50):

    # Ensure model folder exists
    os.makedirs(f"saved_models/{symbol}", exist_ok=True)
    model_path = f"saved_models/{symbol}/gru_model_{forecast_days}d.keras"
    
    # Load and preprocess CSV
    df = pd.read_csv(f"{symbol}.CSV", parse_dates=['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
    df['Price_orig'] = df['Price'].copy()

    # Compute EMA overlays on original prices
    df['EMA_50'] = df['Price_orig'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Price_orig'].ewm(span=200, adjust=False).mean()

    # Scale the Price column for model training
    scaler = StandardScaler()
    df[['Price']] = scaler.fit_transform(df[['Price']])
    price_mean = scaler.mean_[0]
    price_std = scaler.scale_[0]

    # Prepare data for the neural network
    data = df['Price'].values
    X, y = create_sequences(data, look_back=lookback)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Load or train the GRU model
    if os.path.exists(model_path):
        gru_model = tf.keras.models.load_model(model_path)
    else:
        gru_model = Sequential()
        gru_model.add(GRU(50, activation='tanh', return_sequences=True, input_shape=(lookback, 1)))
        gru_model.add(Dropout(0.2))
        gru_model.add(GRU(50, activation='tanh'))
        gru_model.add(Dropout(0.2))
        gru_model.add(Dense(1))
        gru_model.compile(optimizer='adam', loss='mse')
        gru_model.fit(X, y, epochs=epochs, batch_size=32, verbose=1)
        gru_model.save(model_path)

    # Current (last) actual price (in original scale)
    current_price = df['Price_orig'].iloc[-1]

    # Forecast iteratively using the last lookback values
    last_sequence = data[-lookback:]
    forecast_gru = []
    current_seq = last_sequence.copy()
    for i in range(forecast_days):
        current_seq_reshaped = current_seq.reshape(1, lookback, 1)
        pred = gru_model.predict(current_seq_reshaped)
        forecast_gru.append(pred[0, 0])
        current_seq = np.append(current_seq[1:], pred[0, 0])
    forecast_gru_denorm = np.array(forecast_gru) * price_std + price_mean
    predicted_price = forecast_gru_denorm[-1]

    # Compute percentage change and direction
    price_diff = predicted_price - current_price
    pct_diff = (price_diff / current_price) * 100
    direction = "increase" if pct_diff > 0 else "decrease"

    # --- Display summary panel ---
    st.markdown(f"### {symbol} - {forecast_days}-day GRU Forecast")
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric(label="Current Price", value=f"{current_price:.2f}")
    with colB:
        st.metric(label="Predicted Price", value=f"{predicted_price:.2f}")
    with colC:
        sign_str = "+" if pct_diff > 0 else ""
        st.metric(label="Movement (%)", value=f"{sign_str}{pct_diff:.2f}%")
    movement_type = "upside" if direction == "increase" else "downside"
    st.write(
        f"Our AI GRU model predicts a potential **{movement_type}** movement of "
        f"**{abs(pct_diff):.2f}%** over the next **{forecast_days} days**."
    )

    # Build forecast DataFrame for plotting
    forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_days+1, freq='D')[1:]
    forecast_df = pd.DataFrame({'Forecasted_Price': forecast_gru_denorm}, index=forecast_dates)

    # Historical plot data with EMA overlays
    df_plot = pd.DataFrame({
        'Actual_Price': df['Price_orig'],
        'EMA_50': df['EMA_50'],
        'EMA_200': df['EMA_200']
    }, index=df.index)

    # Detect Golden & Death Cross in historical data
    df_plot['GoldenCross'] = (
        (df_plot['EMA_50'] > df_plot['EMA_200']) &
        (df_plot['EMA_50'].shift(1) <= df_plot['EMA_200'].shift(1))
    )
    df_plot['DeathCross'] = (
        (df_plot['EMA_50'] < df_plot['EMA_200']) &
        (df_plot['EMA_50'].shift(1) >= df_plot['EMA_200'].shift(1))
    )
    golden_points = df_plot[df_plot['GoldenCross'] == True]
    death_points  = df_plot[df_plot['DeathCross'] == True]

    # Combine historical and forecasted data
    combined_df = pd.concat([df_plot, forecast_df])
    X, y = create_sequences(data, look_back=lookback)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    split = int(0.8 * len(X))
    X_val = X[split:]
    y_val = y[split:]
    val_predictions = gru_model.predict(X_val)
    val_predictions_denorm = val_predictions * price_std + price_mean
    y_val_denorm = y_val * price_std + price_mean
    rmse = np.sqrt(np.mean((val_predictions_denorm - y_val_denorm) ** 2))
    # Create full plot
    fig = go.Figure()
    # Plot Actual Price
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot['Actual_Price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='blue')
    ))
    # Plot Forecasted Price
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['Forecasted_Price'],
        mode='lines',
        name='Forecasted Price',
        line=dict(color='red', dash='dot')
    ))
    # Plot EMA overlays
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot['EMA_50'],
        mode='lines',
        name='EMA 50',
        line=dict(color='orange')
    ))
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot['EMA_200'],
        mode='lines',
        name='EMA 200',
        line=dict(color='lightgray')
    ))
    # Golden Cross markers
    fig.add_trace(go.Scatter(
        x=golden_points.index,
        y=golden_points['Actual_Price'],
        mode='markers',
        name='Golden Cross',
        marker=dict(color='green', size=10, symbol='triangle-up')
    ))
    # Death Cross markers
    fig.add_trace(go.Scatter(
        x=death_points.index,
        y=death_points['Actual_Price'],
        mode='markers',
        name='Death Cross',
        marker=dict(color='red', size=10, symbol='triangle-down')
    ))
    # Annotate final forecasted price
    if not forecast_df.empty:
        fig.add_annotation(
            x=forecast_df.index[-1],
            y=forecast_df['Forecasted_Price'].iloc[-1],
            text=f"Predicted: {forecast_df['Forecasted_Price'].iloc[-1]:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            font=dict(color="red", size=12),
            bgcolor="white",
            bordercolor="red"
        )
    fig.update_layout(
        title=f"{symbol} GRU Forecast for {forecast_days} Days with EMA & Crossovers",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    return rmse, forecast_df

# ---------------------
# Other Modules (Fundamental, Sentiment, Risk, etc.)
# ---------------------
def show_fundamental_analysis(symbol):
    st.subheader("Advanced Fundamental & Financial Analysis")
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
    except Exception as e:
        st.error(f"Error fetching fundamentals: {e}")
        return
    col1, col2, col3, col4 = st.columns(4)
    market_cap = info.get('marketCap')
    pe_ratio = info.get('trailingPE')
    eps = info.get('trailingEps')
    dividend_yield = info.get('dividendYield')
    col1.metric("Market Cap", f"₹{market_cap:,}" if market_cap else "N/A")
    col2.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
    col3.metric("EPS", f"{eps:.2f}" if eps else "N/A")
    col4.metric("Dividend Yield", f"{dividend_yield*100:.2f}%" if dividend_yield else "N/A")
    profit_growth = info.get("earningsQuarterlyGrowth")
    revenue_growth = info.get("revenueGrowth")
    col5, col6 = st.columns(2)
    col5.metric("Profit Growth", f"{profit_growth*100:.2f}%" if profit_growth else "N/A")
    col6.metric("Revenue Growth", f"{revenue_growth*100:.2f}%" if revenue_growth else "N/A")
    st.markdown("### Fundamental Screener (CAGR Analysis)")
    screener_symbol = symbol.split('.')[0]
    url = f"https://www.screener.in/company/{screener_symbol}/consolidated/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            st.warning(f"Unable to fetch Screener.in page. Status code: {resp.status_code}")
        else:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            cagr_periods = ["10 Years", "5 Years", "3 Years", "1 Year"]
            growth_data = {
                "Compounded Sales Growth": {p: None for p in cagr_periods},
                "Compounded Profit Growth": {p: None for p in cagr_periods}
            }
            table_rows = soup.find_all("tr")
            for row in table_rows:
                th = row.find("th")
                if not th:
                    continue
                label = th.get_text(strip=True)
                if label in growth_data:
                    tds = row.find_all("td")
                    if len(tds) >= 4:
                        growth_data[label]["10 Years"] = tds[0].get_text(strip=True)
                        growth_data[label]["5 Years"] = tds[1].get_text(strip=True)
                        growth_data[label]["3 Years"] = tds[2].get_text(strip=True)
                        growth_data[label]["1 Year"]  = tds[3].get_text(strip=True)
            df_rows = []
            for metric, values in growth_data.items():
                df_rows.append({
                    "Metric": metric,
                    "10Y": values["10 Years"] or "N/A",
                    "5Y": values["5 Years"] or "N/A",
                    "3Y": values["3 Years"] or "N/A",
                    "1Y": values["1 Year"] or "N/A"
                })
            growth_df = pd.DataFrame(df_rows)
            st.dataframe(growth_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error scraping Screener.in: {e}")
    st.write("### Detailed Financials")
    with st.expander("Balance Sheet"):
        try:
            bs = ticker.balance_sheet
            st.dataframe(bs)
        except Exception as e:
            st.error(f"Error fetching balance sheet: {e}")
    with st.expander("Cash Flow"):
        try:
            cf = ticker.cashflow
            st.dataframe(cf)
        except Exception as e:
            st.error(f"Error fetching cash flow: {e}")

def fetch_google_news(query, num_results=10):
    try:
        googlenews = GoogleNews(lang='en', region='IN')
        googlenews.search(query)
        results = googlenews.results()
        return results[:num_results]
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def show_sentiment_analysis(symbol):
    st.subheader(f"Alternative Data & Sentiment Analytics for {symbol}")
    query = f"{symbol.split('.')[0]} stock news"
    articles = fetch_google_news(query, num_results=10)
    if not articles:
        st.info("No recent news available.")
        return
    analyzer = SentimentIntensityAnalyzer()
    news_data = []
    st.markdown("### Latest News Headlines")
    for article in articles:
        title = article.get('title', 'No Title')
        link = article.get('link', '')
        snippet = article.get('desc', 'No description available.')
        sentiment_score = analyzer.polarity_scores(title)['compound']
        st.markdown(f"- [{title}]({link})\n> {snippet}\n**Sentiment Score:** {sentiment_score:.2f}")
        news_data.append({
            "Title": title,
            "Link": link,
            "Snippet": snippet,
            "Sentiment": sentiment_score
        })
    df_news = pd.DataFrame(news_data)
    st.dataframe(df_news)

def show_risk_management(symbol):
    st.subheader("Risk Management & Quantitative Tools")
    data = get_historical_data(symbol, period="1y")
    if data.empty:
        st.error("No historical data available.")
        return
    data['Returns'] = data['Close'].pct_change()
    avg_return = data['Returns'].mean()
    std_return = data['Returns'].std()
    sharpe_ratio = avg_return / std_return if std_return != 0 else np.nan
    var_95 = np.percentile(data['Returns'].dropna(), 5)
    data['Cumulative'] = (1 + data['Returns']).cumprod()
    cumulative_max = data['Cumulative'].cummax()
    drawdown = (data['Cumulative'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Daily Return", f"{avg_return:.2%}")
    col2.metric("Daily Volatility", f"{std_return:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    col4, col5 = st.columns(2)
    col4.metric("VaR (5%)", f"{var_95:.2%}")
    col5.metric("Max Drawdown", f"{max_drawdown:.2%}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Cumulative'], mode='lines', name='Cumulative Returns'))
    fig.update_layout(title="Cumulative Returns Over Time", xaxis_title="Date", yaxis_title="Cumulative Return")
    st.plotly_chart(fig, use_container_width=True)

def predict_market_movement(data):
    latest = data.iloc[-1]
    prediction = "Sideways"
    explanation = "Market appears to be moving in a sideways trend."
    if latest['rsi'] > 70 and latest['macd'] < latest['macd_signal']:
         prediction = "Bearish / Crash"
         explanation = "Overbought conditions with a bearish MACD crossover indicate potential for a crash."
    elif latest['rsi'] < 30 and latest['macd'] > latest['macd_signal']:
         prediction = "Bullish / Bull Run"
         explanation = "Oversold conditions with a bullish MACD crossover suggest potential for a bull run."
    else:
         if latest['Close'] > latest['EMA_50'] and latest['Close'] > latest['EMA_20']:
              prediction = "Bullish"
              explanation = "Strong upward trend indicated by price above moving averages."
         elif latest['Close'] < latest['EMA_50'] and latest['Close'] < latest['EMA_20']:
              prediction = "Bearish"
              explanation = "Downtrend indicated by price below moving averages."
         else:
              prediction = "Sideways"
              explanation = "No strong directional signal from technical indicators."
    return prediction, explanation

def show_market_movement_prediction(symbol):
    st.subheader("Market Movement Prediction")
    data = get_historical_data(symbol, period="6mo")
    if data.empty:
        st.error("No historical data available for market movement prediction.")
        return
    prediction, explanation = predict_market_movement(data)
    st.markdown(f"**Market Movement Prediction:** {prediction}")
    st.write(explanation)

def show_model_improvement_tips():
    st.subheader("Improving Prediction Performance & Model Estimation")
    st.markdown("""
    **Suggestions for improving model performance:**
    - **Feature Engineering:** Incorporate additional technical indicators and alternative data sources.
    - **Hyperparameter Tuning:** Experiment with different lookback periods, learning rates, and architectures.
    - **Cross-Validation:** Use time series cross-validation to evaluate model performance.
    - **Ensemble Methods:** Combine predictions from multiple models (e.g., Prophet, LSTM, GRU).
    - **Data Quality:** Ensure data cleaning and normalization are optimized.
    - **Regularization:** Use dropout and other techniques to prevent overfitting.
    """)

@st.cache_data(ttl=15)
def get_live_stock_data():
    data = []
    for symbol in STOCK_SYMBOLS:
        try:
            stock = yf.Ticker(symbol)
            info = stock.history(period="1d")
            info = fix_yf_data_columns(info)
            yahoo_stock = Ticker(symbol)
            prev_close = yahoo_stock.summary_detail[symbol].get("previousClose", None)
            if not info.empty and prev_close:
                current_price = info.iloc[-1]["Close"]
                change = current_price - prev_close
                percent_change = (change / prev_close) * 100
                data.append({
                    "Stock": symbol,
                    "Price": f"{current_price:.2f}",
                    "Prev Close": f"{prev_close:.2f}",
                    "Change": f"{change:.2f}",
                    "% Change": f"{percent_change:.2f}"
                })
        except:
            pass
    return pd.DataFrame(data)

# ---------------------
# Main App Layout
# ---------------------
def get_period_for_interval(interval: str) -> str:
    intraday_intervals = ["5m", "15m", "30m", "1H"]
    return "60d" if interval in intraday_intervals else "1y"

def apply_custom_css():
     st.markdown(
         """
         <style>
             body {
                 font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                 transition: background-color 0.5s ease;
             }
             .login-box {
                 background: #f9f9f9;
                 border-radius: 10px;
                 padding: 20px;
                 box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                 max-width: 400px;
                 margin: auto;
             }
             .lottie-container {
                 text-align: center;
             }
         </style>
         """, unsafe_allow_html=True)
 
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def main():
    try:
        if 'user' not in st.session_state:
            # Use the CookieManager stored in session_state
            refresh_token = st.session_state.cookie_manager.get('supabase_refresh_token')
            if refresh_token:
                current_session = supabase.auth.refresh_session(refresh_token)
                if current_session and current_session.user:
                    st.session_state.user = current_session.user
                    st.session_state.cookie_manager.set(
                        'supabase_refresh_token',
                        current_session.refresh_token,
                        expires_at=int(time.time() + 604800)
                    )
    except Exception as e:
        st.error(f"Session restoration error: {e}")

    if "user" not in st.session_state:
        st.session_state.user = None

    if st.session_state.user is None:
        st.sidebar.title("Welcome to MarketMinds")

        with st.sidebar: 
            try:
                lottie_animation = load_lottiefile("lottie_animation.json")
                st_lottie(lottie_animation, height=200, key="login_anim")
            except Exception as e:
                st.info("Lottie animation file not found.")

        # Show login or signup based on state
        if st.session_state.get('show_login', False):
            login_form()
            if st.button("← Back to Signup"):
                st.session_state.show_login = False
                st.rerun()
        else:
            signup_form()
            if st.button("Already have an account? Login"):
                st.session_state.show_login = True
                st.rerun()
        st.stop()

    with st.sidebar:
        st.write(f"Logged in as: {st.session_state.user.email}")
        logout()
        st.title("Stock Analytics")
        if st.toggle("Dark Mode", value=st.session_state.dark_mode, on_change=apply_theme):
            st.session_state.dark_mode = not st.session_state.dark_mode
        selected_mode = st.radio(
            "Navigation",
            ["📊 Dashboard", "📈 Live Data", "🔮 Predictions", "🧠 Assistant", "💼 Portfolio", "🛠️ Advanced Tools"],
            index=0
        )
        st.session_state.active_mode = selected_mode.split()[1].lower()
        if st.session_state.active_mode == "portfolio":
            with st.expander("💰 Portfolio Manager", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    p_symbol = st.selectbox("Stock", STOCK_SYMBOLS, key="portfolio_symbol")
                with col2:
                    p_qty = st.number_input("Shares", min_value=1, key="portfolio_qty")
                if st.button("➕ Add to Portfolio"):
                    add_to_portfolio(p_symbol, p_qty)
                    st.success(f"Added {p_qty} shares of {p_symbol}")
    
    if st.session_state.active_mode == "dashboard":
        st.header("Market Dashboard")
        st.subheader("📊 Indices & Sector Performance")
        perf_data = get_indices_and_sectors()
        if not perf_data.empty:
            for i in range(0, len(perf_data), 3):
                row_data = perf_data.iloc[i:i+3]
                cols = st.columns(len(row_data))
                for j, (idx, row) in enumerate(row_data.iterrows()):
                    with cols[j]:
                        st.markdown(create_metric_card(
                            row['Name'],
                            f"₹{row['Price']:,.2f}",
                            f"{row['Change (%)']:.3f}"
                        ), unsafe_allow_html=True)
        else:
            st.warning("⚠️ Could not fetch combined indices/sector data.")
        timeframe = st.selectbox("Select Time Frame For Stock Analysis", ["5m", "15m", "30m", "1H", "1D"], index=4)
        st.subheader("🔍 Advanced Stock Analysis")
        selected_stock = st.selectbox("Select Stock", STOCK_SYMBOLS)
        period_for_data = get_period_for_interval(timeframe)
        data = get_historical_data(selected_stock, period=period_for_data, interval=timeframe, angelone_api_key="JQvcKLzK")
        if not data.empty:
            data_patterns = detect_candlestick_patterns(data)
            insights = generate_technical_insights(data_patterns)
            plot_technical_analysis(data, selected_stock)
            st.subheader("📊 Technical Insights")
            if insights:
                cols = st.columns(2)
                for idx, (title, message, timeframe_insight, color) in enumerate(insights):
                    with cols[idx % 2]:
                        st.markdown(f"""
                        <div style="border-left: 4px solid {color}; padding: 8px; margin: 4px;">
                            <h4 style="color:{color}; margin: 0;">{title}</h4>
                            <small>{timeframe_insight}</small>
                            <p style="margin: 4px 0 0 0;">{message}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No significant candlestick patterns detected.")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current RSI", f"{data['rsi'].iloc[-1]:.1f}")
            with col2:
                st.metric("MACD Signal", f"{data['macd_signal'].iloc[-1]:.2f}")
            with col3:
                st.metric("Volatility (ATR)", f"{data['ATR'].iloc[-1]:.2f}")
        else:
            st.warning("No data available for the selected stock/timeframe.")
    
    elif st.session_state.active_mode == "live":
        st.header("Live Market Data")
        live_data = get_live_stock_data()
        if not live_data.empty:
            st.dataframe(live_data.style.highlight_max(axis=0, color='#000000'),height=840, use_container_width=True)
        else:   
            st.info("No live data available.")
        st_autorefresh(interval=15000, key="live_refresh")
    
    elif st.session_state.active_mode == "predictions":
        st.header("Predictive Analytics")
        selected_stock = st.selectbox("Select Stock for Prediction", STOCK_SYMBOLS)
        forecast_horizon = st.selectbox(
            "Select Forecast Horizon",
            ["1M", "3M", "6M", "1Y", "2Y", "5Y"],
            index=0
        )
        horizon_map = {
            "1M": 30,
            "3M": 90,
            "6M": 180,
            "1Y": 365,
            "2Y": 730,
            "5Y": 1825
        }
        forecast_days = horizon_map[forecast_horizon]
        with st.expander("⚙️ Model Configuration", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                model_type = st.selectbox("Select Model", ["Prophet", "LSTM", "GRU"])
               
        if st.button("🚀 Generate Predictions"):
            with st.spinner("Training model..."):
                try:
                    if model_type == "Prophet":
                        show_prediction_Prophet(selected_stock, forecast_days)
                    elif model_type == "LSTM":
                        show_prediction_LSTM(selected_stock, forecast_days)
                    elif model_type == "GRU":
                        show_prediction_GRU(selected_stock, forecast_days)
                    st.success("Prediction completed successfully!")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    elif st.session_state.active_mode == "assistant":
        st.header("MarketMinds AI Assistant")
        
        # Ensure conversation history is a list
        if "messages" not in st.session_state or not isinstance(st.session_state.messages, list):
            st.session_state.messages = []
        
        # Display the conversation history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        placeholder_text = "Ask me about stocks, market trends, or how to use MarketMinds tools..."
        if prompt := st.chat_input(placeholder=placeholder_text):
            # Append user message to conversation history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(f"**You:** {prompt}")
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if "what can you do" in prompt.lower() or "what can u do" in prompt.lower():
                        response = (
                            "At MarketMinds, we offer comprehensive stock analysis, real-time market data, "
                            "advanced technical charts, predictive analytics, portfolio management, and more. "
                            "You can ask about technical indicators, market predictions, risk management, and actionable investment insights."
                        )
                    else:
                        try:
                            # Prepend concise financial adviser context
                            financial_context = (
                                "You are a seasoned financial adviser at MarketMinds. "
                                "Please provide a concise, clear, and actionable answer point wise and not much long . "
                                "Use your deep knowledge of the stock market and financial analysis to answer the following query:\n\n"
                            )
                            full_prompt = financial_context + prompt
                            response = gemini_model.generate_content(full_prompt).text
                        except Exception as e:
                            response = f"Error generating response: {str(e)}"
                    st.markdown(f"**MarketMinds Assistant:** {response}")
            st.session_state.messages.append({"role": "assistant", "content": response})

    elif st.session_state.active_mode == "portfolio":
        st.header("Portfolio Management")
        
        # Helper function to load portfolio data including buying_price
        def load_full_portfolio():
            user_id = st.session_state.user.id
            response = supabase.table("portfolio").select("*").eq("user_id", user_id).execute()
            return response.data if response.data else []
        
        # Helper function to load transaction history (assumes a portfolio_transactions table exists)
        
        with st.spinner("Loading portfolio data..."):
            portfolio_data = load_full_portfolio()
        
        if portfolio_data:
            st.subheader("Your Portfolio Summary")
            total_invested = 0
            total_current = 0
            detailed_holdings = []
            
            # Loop through each holding and calculate metrics
            for row in portfolio_data:
                symbol = row["stock_symbol"]
                shares = row["shares"]
                buying_price = row.get("buying_price", 0)
                
                # Get current price with error handling
                try:
                    current_price = yf.Ticker(symbol).history(period="1d")['Close'].iloc[-1]
                except Exception as e:
                    current_price = 0
                    st.error(f"Error fetching data for {symbol}: {e}")
                
                invested_value = buying_price * shares
                current_value = current_price * shares
                profit_loss = current_value - invested_value
                profit_loss_pct = ((current_price - buying_price) / buying_price * 100) if buying_price else 0
                
                total_invested += invested_value
                total_current += current_value
                
                # Get historical data for sparkline chart (last 1 month)
                try:
                    hist = yf.Ticker(symbol).history(period="1mo")['Close']
                except Exception:
                    hist = pd.Series()
                
                detailed_holdings.append({
                    "Stock": symbol,
                    "Quantity": shares,
                    "Buying Price": f"₹{buying_price:,.2f}",
                    "Current Price": f"₹{current_price:,.2f}",
                    "Profit/Loss": f"₹{profit_loss:,.2f}",
                    "% Change": f"{profit_loss_pct:+.2f}%",
                    "Sparkline": hist
                })
            
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Invested Amount", f"₹{total_invested:,.2f}")
            with col2:
                st.metric("Total Current Value", f"₹{total_current:,.2f}")
            overall_change = ((total_current - total_invested) / total_invested * 100) if total_invested else 0
            with col3:
                st.metric("Overall Change", f"{overall_change:+.2f}%")
            
            # Sorting option for detailed holdings
            st.subheader("Detailed Holdings")
            sort_option = st.selectbox("Sort By", ["Stock", "Quantity", "Buying Price", "Current Price", "Profit/Loss", "% Change"])
            # Sorting based on numeric values (removing currency symbols) or alphabetically for Stock
            if sort_option == "Stock":
                detailed_holdings = sorted(detailed_holdings, key=lambda x: x["Stock"])
            else:
                def parse_value(val):
                    return float(val.replace("₹", "").replace(",", "").replace("%", ""))
                detailed_holdings = sorted(detailed_holdings, key=lambda x: parse_value(x[sort_option]), reverse=True)
            
            df_holdings = pd.DataFrame(detailed_holdings)
            # Display the table without the sparkline data
            st.dataframe(df_holdings.drop(columns=["Sparkline"]))
            
            # Display sparklines for each holding
            st.markdown("### Price Trends (Last 1 Month)")
            for index, row in df_holdings.iterrows():
                st.write(f"**{row['Stock']}**:")
                if not row["Sparkline"].empty:
                    st.line_chart(row["Sparkline"])
                else:
                    st.write("No historical data available.")
  
            
            # Auto-refresh the portfolio every 10 seconds to reflect any changes
            st_autorefresh(interval=100000, key="portfolio_refresh")
        else:
            st.info("💼 Your portfolio is empty. Add stocks from the sidebar!")

    elif st.session_state.active_mode == "advanced":
        st.header("Advanced Tools")
        adv_stock = st.selectbox("Select Stock", STOCK_SYMBOLS, key="adv_stock")
        adv_option = st.selectbox("Select Analysis Type", ["Fundamental Analysis", "Sentiment Analytics", "Risk Management", "Market Movement Prediction", "Model Improvement Tips"])
        if adv_option == "Fundamental Analysis":
            show_fundamental_analysis(adv_stock)
        elif adv_option == "Sentiment Analytics":
            show_sentiment_analysis(adv_stock)
        elif adv_option == "Risk Management":
            show_risk_management(adv_stock)
        elif adv_option == "Market Movement Prediction":
            show_market_movement_prediction(adv_stock)
        elif adv_option == "Model Improvement Tips":
            show_model_improvement_tips()

if __name__ == "__main__":
    main()