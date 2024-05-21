import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create output directory if it doesn't exist
output_dir = "graph_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read and preprocess Microsoft's CSV data
def read_and_preprocess(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    df['Daily Return'] = df['Adj Close'].pct_change()
    df['5-Day Moving Average'] = df['Adj Close'].rolling(window=5).mean()
    df['20-Day Moving Average'] = df['Adj Close'].rolling(window=20).mean()
    df['50-Day Moving Average'] = df['Adj Close'].rolling(window=50).mean()
    df['200-Day Moving Average'] = df['Adj Close'].rolling(window=200).mean()
    df['Volatility'] = df['Daily Return'].rolling(window=20).std()
    return df

microsoft = read_and_preprocess("data/microsoft.csv")

# Save the augmented CSV
microsoft.to_csv(f'data/microsoft_augmented.csv')

# Plotting functions
def plot_closing_price(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Adj Close'], label='Adjusted Close Price')
    plt.title('Microsoft Adjusted Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/microsoft_adjusted_close_price.png")
    plt.close()

def plot_moving_averages(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Adj Close'], label='Adjusted Close Price')
    plt.plot(df['5-Day Moving Average'], label='5-Day MA')
    plt.plot(df['20-Day Moving Average'], label='20-Day MA')
    plt.plot(df['50-Day Moving Average'], label='50-Day MA')
    plt.plot(df['200-Day Moving Average'], label='200-Day MA')
    plt.title('Microsoft - Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/microsoft_moving_averages.png")
    plt.close()

def plot_daily_return(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Daily Return'], label='Daily Return')
    plt.title('Microsoft Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/microsoft_daily_returns.png")
    plt.close()

def plot_volatility(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Volatility'], label='Volatility')
    plt.title('Microsoft Volatility Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/microsoft_volatility.png")
    plt.close()

def plot_volume(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Volume'], label='Volume')
    plt.title('Microsoft Trading Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/microsoft_volume.png")
    plt.close()

# Plotting all the graphs
plot_closing_price(microsoft)
plot_moving_averages(microsoft)
plot_daily_return(microsoft)
plot_volatility(microsoft)
plot_volume(microsoft)

# Additional Analysis Functions
def calculate_sharpe_ratio(df, risk_free_rate=0.01):
    sharpe_ratio = (df['Daily Return'].mean() - risk_free_rate/252) / df['Daily Return'].std() * np.sqrt(252)
    return sharpe_ratio

def calculate_cagr(df):
    start_value = df['Adj Close'].iloc[0]
    end_value = df['Adj Close'].iloc[-1]
    n_years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (end_value / start_value) ** (1/n_years) - 1
    return cagr

def calculate_drawdown(df):
    df['Cumulative Return'] = (1 + df['Daily Return']).cumprod()
    df['Cumulative Peak'] = df['Cumulative Return'].cummax()
    df['Drawdown'] = (df['Cumulative Return'] - df['Cumulative Peak']) / df['Cumulative Peak']
    return df['Drawdown']

# Calculate and print additional metrics
sharpe_ratio = calculate_sharpe_ratio(microsoft)
print("Microsoft Sharpe Ratio:", sharpe_ratio)

cagr = calculate_cagr(microsoft)
print("Microsoft CAGR:", cagr)

drawdown = calculate_drawdown(microsoft).min()
print("Microsoft Max Drawdown:", drawdown)

# Correlation Analysis
def correlation_analysis(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()

correlation_analysis(microsoft)

# Rolling Beta Calculation against Market Index
def calculate_rolling_beta(df, market_index, window=252):
    covariance = df['Daily Return'].rolling(window=window).cov(market_index['Daily Return'])
    market_variance = market_index['Daily Return'].rolling(window=window).var()
    rolling_beta = covariance / market_variance
    return rolling_beta

# Let's assume we have a market index data, for example S&P 500, which we read and preprocess similar to Microsoft data

# Read and preprocess Market Index's CSV data
def read_and_preprocess_market_index(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    df['Daily Return'] = df['Adj Close'].pct_change()
    return df

# Assuming S&P 500 index data is stored in 'data/sp500.csv'
sp500 = read_and_preprocess_market_index("data/sp500.csv")

# Calculate Rolling Beta
rolling_beta = calculate_rolling_beta(microsoft, sp500)

# Plot Rolling Beta
def plot_rolling_beta(rolling_beta):
    plt.figure(figsize=(14, 7))
    plt.plot(rolling_beta, label='Rolling Beta', color='blue')
    plt.title('Microsoft Rolling Beta against S&P 500')
    plt.xlabel('Date')
    plt.ylabel('Rolling Beta')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/microsoft_rolling_beta.png")
    plt.close()

plot_rolling_beta(rolling_beta)

# Portfolio Optimization (Dummy Example)
# For demonstration purposes, we'll create a simple equally-weighted portfolio using Microsoft and S&P 500
portfolio = pd.concat([microsoft['Adj Close'], sp500['Adj Close']], axis=1)
portfolio.columns = ['Microsoft', 'S&P 500']

# Calculate Portfolio Returns
portfolio['Portfolio Return'] = portfolio.mean(axis=1).pct_change()

# Plot Portfolio Returns
def plot_portfolio_returns(portfolio):
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio['Portfolio Return'], label='Portfolio Return', color='green')
    plt.title('Portfolio Returns')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Return')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/portfolio_returns.png")
    plt.close()

plot_portfolio_returns(portfolio)

print("Extended analysis and plotting completed. Check the graph_output folder for the graphs.")

