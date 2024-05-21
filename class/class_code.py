import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import dates
import numpy as np
import datetime
import os

directory = 'graph_output'
filename_1 = 'df_plot.png'
filepath = os.path.join(directory, filename_1)

# Correcting the method to read CSV files
fb = pd.read_csv('../data/facebook.csv', index_col='Date', parse_dates=True)
ms = pd.read_csv('../data/microsoft.csv', index_col='Date', parse_dates=True)

# Collect next day data
fb['Tomorrow Price'] = fb['Close'].shift(-1)
# Price diff between today and tomorrow
fb['Price Difference'] = fb['Tomorrow Price'] - fb['Close']

# Return
fb['Return'] = fb['Price Difference'] /fb['Close']

# Checking Direction
'''
conditions for direction:
Price Difference > 0 => UP
                <= 0 => DOWN
'''
# init Direction col
fb['Direction'] = fb['Close'] # Choose random column - doesn't matter

for i in range(fb.shape[0]):
    if fb['Price Difference'][i] > 0:
        (fb['Direction']) = 1
    else:
        fb['Direction'] = -1

# Moving Average
fb['Moving Avg'] = (fb['Close'] + fb['Close'].shift(1) + fb['Close'].shift(2))/3
fb['MA_40'] = fb['Close'].rolling(40).mean() # Fast Signal
fb['MA_200'] = fb['Close'].rolling(200).mean() # Slow Signal

if not os.path.exists(directory):
    os.makedirs(directory)

df = fb[['Close','MA_40', 'MA_200']].plot(title ="Moving Average Analysis",figsize=(15,5),legend=True)
df.plot()
plt.savefig(filepath)
plt.close()