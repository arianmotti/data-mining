import pandas as pd
from sklearn import preprocessing

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
