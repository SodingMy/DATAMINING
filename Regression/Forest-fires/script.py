# imports
import pandas as pd
import matplotlib.pyplot as plt

# read data into a DataFrame
data = pd.read_csv('forestfires.csv')

data.columns # This will show all the column names
print data.head()

# print the shape of the DataFrame 
data.shape

# visualize the relationship between the features and the response using scatterplots
fig, axs = plt.subplots(1, 10, sharey=True)
data.plot(kind='scatter', x='Xs', y='areas', ax=axs[0], figsize=(16, 8))
data.plot(kind='scatter', x='Ys', y='areas', ax=axs[1])
data.plot(kind='scatter', x='FFMCs', y='areas', ax=axs[2])
data.plot(kind='scatter', x='DMCs', y='areas', ax=axs[3])
data.plot(kind='scatter', x='DCs', y='areas', ax=axs[4])
data.plot(kind='scatter', x='ISIs', y='areas', ax=axs[5])
data.plot(kind='scatter', x='temps', y='areas', ax=axs[6])
data.plot(kind='scatter', x='RHs', y='areas', ax=axs[7])
data.plot(kind='scatter', x='winds', y='areas', ax=axs[8])
data.plot(kind='scatter', x='rains', y='areas', ax=axs[9])

plt.show()