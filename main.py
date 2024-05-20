import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

directory = 'graph_output'
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

'''fb['Close'].plot()
fb['MA_40'].plot()
fb['MA_200'].plot()'''

if not os.path.exists(directory):
    os.makedirs(directory)

df = fb[['Close','MA_40', 'MA_200']].plot(title ="Moving Average Analysis",figsize=(15,5),legend=True)
df.set_xticklabels(fb['Date'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
df.plot()
plt.gcf().autofmt_xdate()
plt.show()


# Save the plot
#plt.xticks(list(fb['Date']), rotation=0)
#plt.savefig(filepath)
#plt.close()

#print(fb.tail())