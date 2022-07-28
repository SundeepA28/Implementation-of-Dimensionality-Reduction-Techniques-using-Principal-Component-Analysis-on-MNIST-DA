#here implementation of Dimensionality Reduction is done using
#Principal Component Analysis(PCA)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import StandardScaler
import warnings # Current version of Seaborn generates a bunch of warnings that will be ignored.
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# integrating dataset to python notebook
d0 = pd.read_csv('train.csv')
print(d0.head(5)) # checking the data
# separating the labels from the dataset
l = d0['label'] 
d = d0.drop('label',axis = 1)
# confriming with the shapes
print(l.shape)
print(d.shape)
# ploting a sample number visually
plt.figure(figsize=(5,5))
idx = 150
grid_data=d.iloc[idx].values.reshape(28,28) # reshaping from 1d to 2d
plt.imshow(grid_data,interpolation='none',cmap='gray')
plt.show()
print('The above values is',l[idx])
#2D visualization of MNIST using PCA
# using SKlearn importing PCA
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
label = l.head(15000)
data = d.head(15000)
print('The shape of data is ',data.shape)
standard_data = StandardScaler().fit_transform(data)
sample_data = standard_data
pca = decomposition.PCA()
#PCA for dimensionality redcution (not for visualization)
pca.n_components = 784
pca_data = pca.fit_transform(sample_data)
percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);
cum_var_explained = np.cumsum(percentage_var_explained)
# Plot the PCA spectrum
plt.figure(1, figsize=(6, 4))
plt.clf()
plt.plot(cum_var_explained, linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()
# directly entering parameters 
pca.n_components = 2
pca_data = pca.fit_transform(sample_data)
print('shape of pca_reduced data = ',pca_data.shape)
# Data massaging - adding label colomn to the reduced matrix
pca_data = np.vstack((pca_data.T,label)).T
# dataframing and plotting the pca data
pca_df = pd.DataFrame(data=pca_data,columns=('1st_principal','2nd_principal','labels'))
sn.FacetGrid(pca_df,hue='labels',size=6).map(plt.scatter,'1st_principal','2nd_principal').add_legend()
plt.show()