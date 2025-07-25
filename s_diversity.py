#%%
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

data=pd.read_csv('stream_data.csv')

inverts = ['ephem1', 'ephem2', 'tricho1', 'tricho2', 'diptera','coleo', 'oligo', 'amphi','mollusca']

# find proportions
proportions = data[inverts].div(data['total inverts'], axis=0)
print(f'{proportions}')

# find natural log
ln = np.log(proportions.where(proportions > 0))
print(ln)

# proportion * ln
ln_proportion = proportions * ln

# summation
print('Shannon Diversity: ')
sums = -1 * ln_proportion.sum(axis=1)
print(sums)

# graph
fig, ax = plt.subplots(figsize=(9,10))
ax.scatter(x = data['fine sediment'],y = sums)

# correlation
print("Correlation Coefficient: ")
print(sums.corr(data['fine sediment']))
print("As fine sediment in the streambed increases, Shannon diversity is decreased. This could be due to sedimentation reducing hiding spots of species. Additionally, phosphorus is important for the primary production of a system, and they are tied to sediments. They can be lost through erosion, and more sediment can cause more phosphorus to become tied with sediment.")
#%%