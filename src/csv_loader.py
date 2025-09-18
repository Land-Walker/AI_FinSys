import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import os

# --- Configuration ---
TICKER = "^GSPC"  # S&P 500 Index
MACRO_TICKER = "^TNX" # 10-Year Treasury Yield as an interest rate proxy
START_DATE = "2010-01-01"
END_DATE = "2023-12-31"

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)

# ==============================
#     YFINANCE DATA LOADER
# ==============================
# --- 1. Download Stock Data ---
print(f"Downloading stock data for {TICKER}...")
stock_data = yf.download(TICKER, start=START_DATE, end=END_DATE)
stock_df = stock_data[['Close']].rename(columns={'Close': 'close'})
stock_df.to_csv('data/raw/stock_data.csv')
print("✅ Saved stock_data.csv")

# --- 2. Download Macro Proxy Data ---
print(f"Downloading macro proxy data for {MACRO_TICKER}...")
macro_data = yf.download(MACRO_TICKER, start=START_DATE, end=END_DATE)
macro_df = macro_data[['Close']].rename(columns={'Close': 'interest_rate'})
# We will add more macro data in the next step
# For now, let's start with this one file
macro_df.to_csv('data/raw/macro_data.csv') 
print("✅ Saved macro_data.csv (partial)")


# ==============================
#     FRED DATA LOADER
# ==============================
# FRED series codes
# You can find more codes on the FRED website
fred_series = {
    "CPIAUCSL": "cpi",          # Consumer Price Index
    "UNRATE": "unemployment",   # Unemployment Rate
    "DGS10": "interest_rate"    # 10-Year Treasury (alternative to yfinance)
}

# --- Download Data from FRED ---
print("Downloading data from FRED...")
fred_df = web.DataReader(list(fred_series.keys()), 'fred', START_DATE, END_DATE)
fred_df = fred_df.rename(columns=fred_series)

# FRED data can be monthly, so we forward-fill to make it daily
fred_df = fred_df.ffill() 

# --- Save to CSV ---
fred_df.to_csv('data/raw/macro_data.csv')
print("✅ Saved macro_data.csv (complete) from FRED")


# ==============================
#     LABEL DATA LOADER
# ==============================
# --- Configuration ---
# Use the stock data we downloaded earlier as the basis for our labels
STOCK_DATA_PATH = 'data/raw/stock_data.csv'
MOVING_AVERAGE_WINDOW = 200
# Define the threshold: e.g., 5% above the MA is "bull", 5% below is "bear"
THRESHOLD = 0.05 

# --- Load S&P 500 Data ---
df = pd.read_csv(STOCK_DATA_PATH, parse_dates=['date'], index_col='date')

# --- 1. Calculate the Moving Average ---
df['ma'] = df['close'].rolling(window=MOVING_AVERAGE_WINDOW).mean()
df = df.dropna() # Remove initial rows where MA is not available

# --- 2. Define the Rule and Apply it ---
conditions = [
    (df['close'] > df['ma'] * (1 + THRESHOLD)),  # Bull condition
    (df['close'] < df['ma'] * (1 - THRESHOLD)),  # Bear condition
]
choices = ['bull', 'bear']

# numpy.select is a great way to apply conditional logic
df['label'] = np.select(conditions, choices, default='neutral')

# --- 3. Save the Labels File ---
labels_df = df[['label']].reset_index()
labels_df.to_csv('data/raw/scenario_labels.csv', index=False)

print("✅ Saved scenario_labels.csv")
print("\nLabel distribution:")
print(labels_df['label'].value_counts())