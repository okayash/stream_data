#%%
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def get_rmse(line, actual):
    error = [(x-y) for x, y in zip(line, actual)]
    errorsq = [v**2 for v in error]
    return(f'RMSE Error Measurement: {np.sqrt(np.mean(errorsq))}')

data=pd.read_csv('stream_data.csv')

inverts = ['ephem1', 'ephem2', 'tricho1', 'tricho2', 'diptera','coleo', 'oligo', 'amphi','mollusca']

# find proportions
proportions = data[inverts].div(data['total inverts'], axis=0)
print(f'\n{proportions}')

# find natural log
ln = np.log(proportions.where(proportions > 0))
print(ln)

# proportion * ln
ln_proportion = proportions * ln

# summation
print('\nShannon Diversity: ')
sums = -1 * ln_proportion.sum(axis=1)
print(sums)

# correlation
print("\nCorrelation Coefficient: ")
print(sums.corr(data['fine sediment']))

# linear regression
x = data['fine sediment'].values.reshape(-1,1)
y = sums.values.reshape(-1,1)

regressor = LinearRegression()
regressor.fit(x, y)

coef = regressor.coef_[0][0]
intercept = regressor.intercept_[0]

print(f'Coefficent: {coef}\n')
print(f'Intercept: {intercept}\n')
      
# graph
ys = coef * data['fine sediment'] + intercept
fig, ax = plt.subplots(figsize=(9,10))
ax.scatter(x = data['fine sediment'],y = sums)
regressionline = (coef * i + intercept for i in data['fine sediment'])
plt.plot(data['fine sediment'], [coef * i + intercept for i in data['fine sediment']], 'r-', label='Regression Line')
plt.show()

# RMSE ERROR
sumlist = sums.tolist()
print(get_rmse(regressionline, sumlist))

print("\nAs fine sediment in the streambed increases, Shannon diversity is decreased.  \
      This could be due to sedimentation reducing hiding spots of species. \
      Additionally, phosphorus is important for the primary production of a system, and they are tied to sediments. \
      They can be lost through erosion, and more sediment can cause more phosphorus to become tied with sediment.")

# future forecasting
extended = np.append(data['fine sediment'], np.arange(12, 90))
extended = extended.reshape(-1,1)
extended_prediction = regressor.predict(extended)

ax.scatter(x = data['fine sediment'],y = sums)
fig, ax = plt.subplots(figsize=(9,10))
plt.plot(extended, extended_prediction, 'r--')
plt.title("extrapolated plot")
plt.xlabel("sediment %")
plt.ylabel('shannon diversity')
plt.show()
#%%