# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Reading the data set
data = pd.read_csv('cancer_dataset.csv')
X = data.iloc[:, 1:31].values
Y = data.iloc[:, 31].values

data.head()

print("Cancer data set dimensions : {}".format(data.shape))

# data.groupby('diagnosis').size()

data.isnull().sum()
data.isna().sum()

dataframe = pd.DataFrame(Y)
# using Label Encoder to convert categorical data into number so the
# model can understand better
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print(Y)

