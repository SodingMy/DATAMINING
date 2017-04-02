# imports
import pandas as pd
import matplotlib.pyplot as plt
# this is the standard import if you're using "formula notation" (similar to R)
import statsmodels.formula.api as smf

# read data into a DataFrame
data = pd.read_csv('Airfoil-Self-Noise.csv')

data.columns # This will show all the column names
data.head()

# print the shape of the DataFrame 
data.shape

# visualize the relationship between the features and the response using scatterplots
fig, axs = plt.subplots(1, 5, sharey=True)
data.plot(kind='scatter', x='A', y='F', ax=axs[0], figsize=(16, 8))
data.plot(kind='scatter', x='B', y='F', ax=axs[1])
data.plot(kind='scatter', x='C', y='F', ax=axs[2])
data.plot(kind='scatter', x='D', y='F', ax=axs[3])
data.plot(kind='scatter', x='E', y='F', ax=axs[4])

# create a fitted model in one line
lm = smf.ols(formula='F ~ B', data=data).fit()

# print the coefficients
lm.params

#print 126.309388 + (0.008927 * 70)

# you have to create a DataFrame since the Statsmodels formula interface expects it
X_new = pd.DataFrame({'B': [70]})
X_new.head()

# use the model to make predictions on a new value
lm.predict(X_new)

# create a DataFrame with the minimum and maximum values of TV
X_new = pd.DataFrame({'B': [data.B.min(), data.B.max()]})
X_new.head()

# make predictions for those x values and store them
preds = lm.predict(X_new)
preds

# first, plot the observed data
data.plot(kind='scatter', x='B', y='F')

# then, plot the least squares line
plt.plot(X_new, preds, c='red', linewidth=2)

# print the confidence intervals for the model coefficients
lm.conf_int()

# print the p-values for the model coefficients
print lm.pvalues

#plt.show()