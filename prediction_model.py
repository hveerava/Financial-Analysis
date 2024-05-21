import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Read and preprocess Microsoft's CSV data
def read_and_preprocess(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    
    # Handling missing data
    df.fillna(method='ffill', inplace=True)
    
    return df

output_dir = "predicted_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Add additional features including technical indicators
def add_additional_features(df):
    df['5-Day Moving Average'] = df['Adj Close'].rolling(window=5).mean()
    df['20-Day Moving Average'] = df['Adj Close'].rolling(window=20).mean()
    df['50-Day Moving Average'] = df['Adj Close'].rolling(window=50).mean()
    df['200-Day Moving Average'] = df['Adj Close'].rolling(window=200).mean()
    df['Volatility'] = df['Adj Close'].pct_change().rolling(window=20).std()
    df['RSI'] = calculate_rsi(df)
    df['MACD'] = calculate_macd(df)
    df['ATR'] = calculate_atr(df)
    return df

# Calculate Relative Strength Index (RSI)
def calculate_rsi(df, window=14):
    delta = df['Adj Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate Moving Average Convergence Divergence (MACD)
def calculate_macd(df):
    ema_12 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Adj Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd = macd_line - signal_line
    return macd

# Calculate Average True Range (ATR)
def calculate_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Adj Close'].shift())
    low_close = np.abs(df['Low'] - df['Adj Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

# Split data into features and target variable
def split_data(df):
    X = df.drop(columns=['Adj Close'])  # Features
    y = df['Adj Close']  # Target variable
    return X, y

# Train a random forest regressor model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, r2

# Plot the predictions
def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test.values, label='Actual Closing Price', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted Closing Price', color='red')
    plt.title('Actual vs Predicted Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(f"{output_dir}/microsoft_adjusted_close_price.png")
    plt.close()

# Function to handle missing values
def handle_missing_values(X_train, X_test):
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed, X_test_imputed

# Main function
def main():
    file_path = "data/microsoft.csv"
    df = read_and_preprocess(file_path)
    df = add_additional_features(df)
    
    # Split data into train and test sets
    X, y = split_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Handle missing values
    X_train_imputed, X_test_imputed = handle_missing_values(X_train, X_test)
    
    # Train the model
    model = train_model(X_train_imputed, y_train)
    
    # Evaluate the model
    mse, rmse, r2 = evaluate_model(model, X_test_imputed, y_test)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R-squared Score:", r2)
    # File writing
    with open("model_eval.txt", "w") as file:
        file.write("Mean Squared Error: " + str(mse) + "\n")
        file.write("Root Mean Squared Error: " + str(rmse) + "\n")
        file.write("R-squared Score: " + str(r2) + "\n")
    
    # Make predictions
    y_pred = model.predict(X_test_imputed)
    
    # Plot predictions
    plot_predictions(y_test, y_pred)

if __name__ == "__main__":
    main()