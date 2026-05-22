import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Page configurations & Themes
st.set_page_config(page_title="Smart Invest AI Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Premium UI & Animations
st.markdown("""
    <style>
    /* Global Background and Text Smoothness */
    .main { background-color: #0e1117; color: #ffffff; }
    
    /* Fade-in Animation for Title and Cards */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .animated-header {
        animation: fadeIn 1.2s ease-out;
        text-align: center;
        background: linear-gradient(45deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 5px;
    }
    
    /* Modern Card styling with hover effect */
    .metric-card {
        background-color: #1a1c23;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid #2d3139;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 1.5s ease-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 114, 255, 0.2);
        border-color: #0072ff;
    }
    </style>
""", unsafe_allow_html=True)

# Animated Super Title
st.markdown("<h1 class='animated-header'>🚀 SMART INVEST: ADVANCED LIVE STOCK FORECASTING ENGINE</h1>", unsafe_allow_html=True)

# Institutional Sidebar Info
st.sidebar.markdown("## 🧑‍💻 ACADEMIC PROJECT")
st.sidebar.info("""
    **SUBMITTED BY:** 🎓 Aniket Hedau  
    
    **SUBMITTED TO:** 👨‍🏫 Rajesh Prasad Sir  
    
    **INSTITUTION:** 🏫 Prestige Institute of Management and Research (PIMR)
""")

# PyTorch Model Struct
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :])

@st.cache_resource
def load_engine():
    model = StockLSTM()
    model.load_state_dict(torch.load("reliance_lstm_pytorch.pth", map_dict={'cpu': torch.device('cpu')}))
    model.eval()
    return model

model = load_engine()

# Sidebar Controls & Expanded Company List
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ LIVE CONTROLS")

stock_dict = {
    "Tata Motors (NSE)": "TATAMOTORS.NS",
    "Reliance Industries (NSE)": "RELIANCE.NS",
    "State Bank of India (NSE)": "SBIN.NS",
    "MRF Tyres (NSE)": "MRF.NS",
    "Infosys Ltd (NSE)": "INFY.NS",
    "Apple Inc. (NASDAQ)": "AAPL",
    "Tesla Inc. (NASDAQ)": "TSLA",
    "Google LLC (NASDAQ)": "GOOGL"
}

selected_stock = st.sidebar.selectbox("🎯 Choose Asset to Analyze:", list(stock_dict.keys()))
ticker = stock_dict[selected_stock]

# Past Day Window - Dynamic Slider Fixed!
window = st.sidebar.slider("⏰ Past Window (Days for Pattern Matching):", min_value=30, max_value=90, value=60)

# Fetching Data Live
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')

@st.cache_data(ttl=600)  # Auto refresh data every 10 mins
def load_live_market_data(stock_ticker, start, end):
    raw_data = yf.download(stock_ticker, start=start, end=end)
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)
    return raw_data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

try:
    df_live = load_live_market_data(ticker, start_date, end_date)
    currency = "₹" if ".NS" in ticker else "$"
    
    # UI Layout: Top Metrics Cards
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    
    # Extract prices and volume
    close_prices = df_live['Close'].values
    current_price = close_prices[-1][0] if isinstance(close_prices[-1], np.ndarray) else close_prices[-1]
    
    # Fixed SLIDER LOGIC for real-time model interaction
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(df_live[['Close']].values)
    
    # Dynamic window extraction based on slider
    last_window_data = scaled_close[-window:]
    
    # Shape matching to ensure LSTM accepts it correctly [1, window_size, 1]
    input_tensor = torch.tensor(last_window_data, dtype=torch.float32).unsqueeze(0)
    if input_tensor.shape[1] != window:
        # Fallback padding if data length mismatch occurs
        input_tensor = torch.nn.functional.pad(input_tensor, (0,0, window - input_tensor.shape[1], 0))

    with torch.no_grad():
        pred_scaled = model(input_tensor)
    pred_price = scaler.inverse_transform(pred_scaled.numpy())[0][0]
    price_diff = pred_price - current_price

    with c1:
        st.markdown(f"""
            <div class='metric-card'>
                <p style='color:#8a90a6; margin:0;'>LIVE CLOSING PRICE</p>
                <h2 style='color:#ffffff; margin:5px 0;'>{currency}{current_price:.2f}</h2>
                <span style='color:#00ff88;'>● Live Market Connected</span>
            </div>
        """, unsafe_allow_html=True)
        
    with c2:
        color_trend = "#00ff88" if price_diff >= 0 else "#ff4b4b"
        arrow = "▲" if price_diff >= 0 else "▼"
        st.markdown(f"""
            <div class='metric-card'>
                <p style='color:#8a90a6; margin:0;'>AI NEXT-DAY FORECAST</p>
                <h2 style='color:#ffffff; margin:5px 0;'>{currency}{pred_price:.2f}</h2>
                <span style='color:{color_trend};'>{arrow} {price_diff:.2f} ({window} Days Trend Engine)</span>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        latest_vol = int(df_live['Volume'].iloc[-1])
        st.markdown(f"""
            <div class='metric-card'>
                <p style='color:#8a90a6; margin:0;'>TODAY TRADING VOLUME</p>
                <h2 style='color:#ffffff; margin:5px 0;'>{latest_vol:,}</h2>
                <span style='color:#00c6ff;'>🔄 Auto-Refreshed Live</span>
            </div>
        """, unsafe_allow_html=True)

    # UI Layout: Multiple Charts Section
    st.markdown("### 📊 MULTI-DIMENSIONAL REAL TIME CHARTS")
    
    tab1, tab2, tab3 = st.tabs(["📈 Area Chart (Market Flow)", "📉 Line Chart (Historical Trend)", "📊 Bar Chart (Volume Analytics)"])
    
    recent_df = df_live.tail(120)  # Extract last 120 days for sharp visibility
    
    with tab1:
        st.write("#### Modern Area Fill-Effect Chart")
        fig1, ax1 = plt.subplots(figsize=(12, 4.5))
        fig1.patch.set_facecolor('#0e1117')
        ax1.set_facecolor('#1a1c23')
        
        ax1.fill_between(recent_df.index, recent_df['Close'], color='#0072ff', alpha=0.3)
        ax1.plot(recent_df.index, recent_df['Close'], color='#00c6ff', linewidth=2.5, label="Closing Price")
        
        ax1.title.set_color('white')
        ax1.tick_params(colors='white', labelsize=9)
        ax1.grid(True, color='#2d3139', linestyle='--', alpha=0.5)
        plt.xticks(rotation=25)
        st.pyplot(fig1)

    with tab2:
        st.write("#### Technical High-Low Boundary Line Chart")
        fig2, ax2 = plt.subplots(figsize=(12, 4.5))
        fig2.patch.set_facecolor('#0e1117')
        ax2.set_facecolor('#1a1c23')
        
        ax2.plot(recent_df.index, recent_df['High'], color='#00ff88', linestyle=':', label="Day High", alpha=0.7)
        ax2.plot(recent_df.index, recent_df['Close'], color='#ffffff', linewidth=2, label="Day Close")
        ax2.plot(recent_df.index, recent_df['Low'], color='#ff4b4b', linestyle=':', label="Day Low", alpha=0.7)
        
        ax2.tick_params(colors='white', labelsize=9)
        ax2.grid(True, color='#2d3139', linestyle='--', alpha=0.5)
        ax2.legend(facecolor='#1a1c23', edgecolor='#2d3139', labelcolor='white')
        plt.xticks(rotation=25)
        st.pyplot(fig2)

    with tab3:
        st.write("#### Market Liquid Liquidity (Volume Bar Representation)")
        fig3, ax3 = plt.subplots(figsize=(12, 4.5))
        fig3.patch.set_facecolor('#0e1117')
        ax3.set_facecolor('#1a1c23')
        
        # Color matching bars based on up/down days
        colors = ['#00ff88' if recent_df['Close'].iloc[i] >= recent_df['Open'].iloc[i] else '#ff4b4b' for i in range(len(recent_df))]
        ax3.bar(recent_df.index, recent_df['Volume'], color=colors, alpha=0.8, width=0.8)
        
        ax3.tick_params(colors='white', labelsize=9)
        ax3.grid(True, color='#2d3139', linestyle='--', alpha=0.3)
        plt.xticks(rotation=25)
        st.pyplot(fig3)

except Exception as e:
    st.error(f"⚠️ Live Analysis Syncing Error: {e}")