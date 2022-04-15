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
print('sepal_lenght : ', Null_count[0])
print('sepal_width : ', Null_count[1])
print('petal_length : ', Null_count[2])
print('petal_width : ', Null_count[3])
print('target : ', Null_count[4])

df = df.dropna(axis=0, how='any')
print(df)
encoder = preprocessing.LabelEncoder()
df['target'] = (encoder.fit_transform(df['target']))
df['target'].unique()

variance = []
mean = []

for i in range(4):
    variance.append(statistics.variance(df.iloc[:, i]))
    mean.append(statistics.mean(df.iloc[:, i]))
df2 = copy.deepcopy(df)
del df2['target']
scaler = preprocessing.StandardScaler()
standard_df = scaler.fit_transform(df2)
standard_df = pd.DataFrame(standard_df, columns=[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])

normalized_variance = []
normalized_mean = []
for i in range(4):
    normalized_variance.append(statistics.variance(standard_df.iloc[:, i]))
    normalized_mean.append(statistics.mean(standard_df.iloc[:, i]))
for i in range(4):
    print("column %2d" % (i + 1))
    print("variance : before  %6.2f  after %6.2f" % (variance[i], normalized_variance[i]))
    print("mean     : before  %6.2f  after %6.2f" % (mean[i], normalized_mean[i]))

principal = PCA(n_components=2)
principal.fit(standard_df)
data = principal.transform(standard_df)

setosa = []
versicolor = []
virginica = []
for i in range(len(data)):
    if df.iloc[i]['target'] == 0:
        setosa.append(data[i])
    elif df.iloc[i]['target'] == 1:
        versicolor.append(data[i])
    else:
        virginica.append(data[i])


def plot(arr1, arr2, arr3,tmp):
    plt.scatter([i[0] for i in setosa], [i[1] for i in arr1])
    plt.scatter([i[0] for i in versicolor], [i[1] for i in arr2])
    plt.scatter([i[0] for i in virginica], [i[1] for i in arr3])
    plt.legend(["setosa", "versicolor", "virginica"])
    plt.show()
    visual_data = copy.deepcopy(tmp)
    del visual_data['target']
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.boxplot(visual_data)
    plt.show()


plot(setosa, versicolor, virginica,df)
