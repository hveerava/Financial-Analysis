import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Create output directory if it doesn't exist
output_dir = "graph_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read and preprocess Microsoft's CSV data
def read_and_preprocess(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    
    # Handling missing data
    df.fillna(method='ffill', inplace=True)
    
    df['Daily Return'] = df['Adj Close'].pct_change()
    df['5-Day Moving Average'] = df['Adj Close'].rolling(window=5).mean()
    df['20-Day Moving Average'] = df['Adj Close'].rolling(window=20).mean()
    df['50-Day Moving Average'] = df['Adj Close'].rolling(window=50).mean()
    df['200-Day Moving Average'] = df['Adj Close'].rolling(window=200).mean()
    df['Volatility'] = df['Daily Return'].rolling(window=20).std()
    df['RSI'] = calculate_rsi(df)
    df['EMA_12'] = df['Adj Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Adj Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Cumulative Return'] = (1 + df['Daily Return']).cumprod()
    df['Upper Band'], df['Lower Band'] = calculate_bollinger_bands(df)
    df['VWAP'] = calculate_vwap(df)
    df['ATR'] = calculate_atr(df)
    df['Stochastic Oscillator'] = calculate_stochastic_oscillator(df)
    df['ADX'] = calculate_adx(df)
    return df

# Calculate Relative Strength Index (RSI)
def calculate_rsi(df, window=14):
    delta = df['Adj Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate Bollinger Bands
def calculate_bollinger_bands(df, window=20, std_multiplier=2):
    sma = df['Adj Close'].rolling(window=window).mean()
    std = df['Adj Close'].rolling(window=window).std()
    upper_band = sma + (std_multiplier * std)
    lower_band = sma - (std_multiplier * std)
    return upper_band, lower_band

# Calculate Volume Weighted Average Price (VWAP)
def calculate_vwap(df):
    vwap = (df['Volume'] * (df['High'] + df['Low'] + df['Adj Close']) / 3).cumsum() / df['Volume'].cumsum()
    return vwap

# Calculate Average True Range (ATR)
def calculate_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Adj Close'].shift())
    low_close = np.abs(df['Low'] - df['Adj Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

# Calculate Stochastic Oscillator
def calculate_stochastic_oscillator(df, window=14):
    low_min = df['Low'].rolling(window=window).min()
    high_max = df['High'].rolling(window=window).max()
    stochastic_oscillator = 100 * ((df['Adj Close'] - low_min) / (high_max - low_min))
    return stochastic_oscillator

# Calculate Average Directional Index (ADX)
def calculate_adx(df, window=14):
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = df[['High', 'Low', 'Adj Close']].max(axis=1) - df[['High', 'Low', 'Adj Close']].min(axis=1)
    tr_sma = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).sum() / tr_sma)
    minus_di = 100 * (minus_dm.rolling(window=window).sum() / tr_sma)
    dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=window).mean()
    return adx

# Plotting functions
def plot_closing_price(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Adj Close'], label='Adjusted Close Price')
    plt.title('Microsoft Adjusted Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/microsoft_adjusted_close_price.png")
    plt.close()

def plot_rsi(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['RSI'], label='RSI')
    plt.title('Microsoft Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/microsoft_rsi.png")
    plt.close()

def plot_macd(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['MACD'], label='MACD', color='blue')
    plt.plot(df.index, df['Signal Line'], label='Signal Line', color='red')
    plt.title('Microsoft Moving Average Convergence Divergence (MACD)')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/microsoft_macd.png")
    plt.close()

def plot_cumulative_returns(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Cumulative Return'], label='Cumulative Return', color='green')
    plt.title('Microsoft Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/microsoft_cumulative_returns.png")
    plt.close()

def plot_bollinger_bands(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Adj Close'], label='Adjusted Close Price')
    plt.plot(df.index, df['Upper Band'], label='Upper Band', color='red', linestyle='--')
    plt.plot(df.index, df['Lower Band'], label='Lower Band', color='green', linestyle='--')
    plt.fill_between(df.index, df['Upper Band'], df['Lower Band'], color='gray', alpha=0.2)
    plt.title('Microsoft Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/microsoft_bollinger_bands.png")
    plt.close()

def plot_vwap(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Adj Close'], label='Adjusted Close Price')
    plt.plot(df.index, df['VWAP'], label='VWAP', color='purple', linestyle='--')
    plt.title('Microsoft Volume Weighted Average Price (VWAP)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/microsoft_vwap.png")
    plt.close()

def plot_moving_averages(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Adj Close'], label='Adjusted Close Price')
    plt.plot(df.index, df['5-Day Moving Average'], label='5-Day MA')
    plt.plot(df.index, df['20-Day Moving Average'], label='20-Day MA')
    plt.plot(df.index, df['50-Day Moving Average'], label='50-Day MA')
    plt.plot(df.index, df['200-Day Moving Average'], label='200-Day MA')
    plt.title('Microsoft - Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/microsoft_moving_averages.png")
    plt.close()

def plot_daily_return(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Daily Return'], label='Daily Return')
    plt.title('Microsoft Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/microsoft_daily_returns.png")
    plt.close()

def plot_volatility(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Volatility'], label='Volatility')
    plt.title('Microsoft Volatility (20-Day Rolling Std Dev of Daily Returns)')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/microsoft_volatility.png")
    plt.close()

def plot_stochastic_oscillator(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Stochastic Oscillator'], label='Stochastic Oscillator')
    plt.title('Microsoft Stochastic Oscillator')
    plt.xlabel('Date')
    plt.ylabel('Stochastic Oscillator')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/microsoft_stochastic_oscillator.png")
    plt.close()

def plot_adx(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['ADX'], label='ADX')
    plt.title('Microsoft Average Directional Index (ADX)')
    plt.xlabel('Date')
    plt.ylabel('ADX')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/microsoft_adx.png")
    plt.close()

# Main function to call all other functions
def main():
    file_path = "data/microsoft.csv"
    df = read_and_preprocess(file_path)
    
    plot_closing_price(df)
    plot_rsi(df)
    plot_macd(df)
    plot_cumulative_returns(df)
    plot_bollinger_bands(df)
    plot_vwap(df)
    plot_moving_averages(df)
    plot_daily_return(df)
    plot_volatility(df)
    plot_stochastic_oscillator(df)
    plot_adx(df)

if __name__ == "__main__":
    main()