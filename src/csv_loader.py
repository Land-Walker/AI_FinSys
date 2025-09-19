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
#     1. YFINANCE DATA LOADER
# ============================== 
# --- 1. Download Stock Data ---
print(f"Downloading stock data for {TICKER}...")
stock_data = yf.download(TICKER, start=START_DATE, end=END_DATE)
stock_df = stock_data[['Open']].rename(columns={'Open': 'open'})
stock_df = stock_data[['High']].rename(columns={'High': 'high'})
stock_df = stock_data[['Low']].rename(columns={'Low': 'low'})
stock_df = stock_data[['Close']].rename(columns={'Close': 'close'})
stock_df = stock_data[['Volume']].rename(columns={'Volume': 'volume'})
stock_df = stock_df.reset_index() # <-- FIX: Convert the date index into a column
stock_df.to_csv('data/raw/stock_data.csv')
print("✅ Saved stock_data.csv")

# --- 2. Download Macro Proxy Data ---
print(f"Downloading macro proxy data for {MACRO_TICKER}...")
macro_data = yf.download(MACRO_TICKER, start=START_DATE, end=END_DATE)
macro_df = macro_data[['Close']].rename(columns={'Close': 'interest_rate'})
macro_df = macro_df.reset_index()
# We will add more macro data in the next step
# For now, let's start with this one file
macro_df.to_csv('data/raw/macro_data.csv') 
print("✅ Saved macro_data.csv (partial)")


# ==============================
#     2. FRED DATA LOADER
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
fred_df = fred_df.reset_index().rename(columns={'index': 'date'}) # <-- FIX: Convert index and rename
fred_df.to_csv('data/raw/macro_data.csv')
print("✅ Saved macro_data.csv (complete) from FRED")


# ==============================
#      3. SCENARIO LABELS
# ==============================
print("Generating scenario labels...")

# 💡 NEW LOGIC: Check if the main ticker is the S&P 500
if TICKER != "^GSPC":
    print(f"Ticker '{TICKER}' is not S&P 500. Downloading ^GSPC separately for market labels...")
    sp500_data = yf.download("^GSPC", start=START_DATE, end=END_DATE)
    df_for_labels = sp500_data[['Close']].rename(columns={'Close': 'close'})
    df_for_labels = df_for_labels.reset_index().rename(columns={'Date': 'date'})
else:
    print("Ticker is S&P 500. Using its own data for market labels...")
    # Use the DataFrame we already loaded
    df_for_labels = stock_df.copy()


# --- Define Rules and Apply (This part remains the same) ---
df_for_labels['date'] = pd.to_datetime(df_for_labels['Date'])
df_for_labels = df_for_labels.set_index('date')

# Check if columns have a MultiIndex and flatten if necessary
if isinstance(df_for_labels.columns, pd.MultiIndex):
    print("Detected MultiIndex in columns. Flattening...")
    df_for_labels.columns = [col[0] if col[1] == '^GSPC' else col[0] for col in df_for_labels.columns]
else:
    print("No MultiIndex detected in columns.")

MOVING_AVERAGE_WINDOW = 200
THRESHOLD = 0.05

# Calculate moving average
df_for_labels['ma'] = df_for_labels['close'].rolling(window=MOVING_AVERAGE_WINDOW).mean()

# No need for align, as 'close' and 'ma' should have the same index
conditions = [
    df_for_labels['close'] > df_for_labels['ma'] * (1 + THRESHOLD),  # Bull
    df_for_labels['close'] < df_for_labels['ma'] * (1 - THRESHOLD),  # Bear
]

choices = ['bull', 'bear']
df_for_labels['label'] = np.select(conditions, choices, default='neutral')

# --- Save the Labels File ---
labels_df = df_for_labels[['label']].reset_index()
labels_df.to_csv('data/raw/macro_data.csv')

print("✅ Saved scenario_labels.csv")
print("\nLabel distribution:")
print(labels_df['label'].value_counts())