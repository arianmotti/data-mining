import copy
import pandas as pd
from sklearn import preprocessing
import statistics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('./iris.data',
                 names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
print(df)
Null_count = [0, 0, 0, 0, 0]
missing_data = df.isna()
for i in range(len(missing_data)):
    for j in range(5):
        if missing_data.iloc[i][j]:
            Null_count[j] += 1
print("missing data in each column:")
print(Null_count)
df = df.dropna(axis=0, how='any')
print(df)
encoder = preprocessing.LabelEncoder()
df['target'] = (encoder.fit_transform(df['target']))
df['target'].unique()

variance = []
mean = []

for i in range(4):
    variance.append(statistics.variance(df.iloc[i]))
    mean.append(statistics.mean(df.iloc[i]))
df2 = copy.deepcopy(df)
del df2['target']
scaler = preprocessing.StandardScaler()
standard_df = scaler.fit_transform(df2)
standard_df = pd.DataFrame(standard_df, columns=[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])

normalized_variance = []
normalized_mean = []
for i in range(4):
    normalized_variance.append(statistics.variance(standard_df.iloc[i]))
    normalized_mean.append(statistics.mean(standard_df.iloc[i]))
for i in range(4):
    print("column %2d" % (i + 1))
    print("variance : before  %5.2f  after %5.2f" % (variance[i], normalized_variance[i]))
    print("mean     : before  %5.2f  after %5.2f" % (mean[i], normalized_mean[i]))
