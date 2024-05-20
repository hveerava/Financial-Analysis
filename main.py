import pandas as pd
import matplotlib.pyplot as plt
import os

directory = 'data'
filename_1 = 'df_plot.png'
filepath = os.path.join(directory, filename_1)

# Correcting the method to read CSV files
fb = pd.read_csv('data/facebook.csv')
ms = pd.read_csv('data/microsoft.csv')

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
fb['Direction'] = fb['Close']

for i in range(fb.shape[0]):
    if fb.loc[i, 'Price Difference'] > 0:
        (fb['Direction'][i]) = 1
    else:
        fb['Direction'][i] = -1

# Moving Average
fb['Moving Avg'] = (fb['Close'] + fb['Close'].shift(1) + fb['Close'].shift(2))/3
fb['MA_40'] = fb['Close'].rolling(40).mean()
fb['MA_200'] = fb['Close'].rolling(200).mean()

fb['Close'].plot()
fb['MA_40'].plot()
fb['MA_200'].plot()

if not os.path.exists(directory):
    os.makedirs(directory)

# Save the plot

plt.show()
plt.savefig(filepath)
plt.close()

#print(fb.tail())