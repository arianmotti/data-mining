🌸 Iris Dataset — PCA & Data Normalization

This project demonstrates Principal Component Analysis (PCA) on the classic Iris dataset, including data cleaning, normalization, variance analysis, and visualization of results in both 2D and boxplot form.

🧠 Overview

The Iris dataset consists of 150 samples from three species of Iris flowers:

Iris setosa

Iris versicolor

Iris virginica

Each sample includes four features:

Sepal length

Sepal width

Petal length

Petal width

This project:

Loads and preprocesses the dataset

Handles missing values

Encodes categorical labels

Standardizes numeric features

Applies PCA to reduce dimensions from 4 → 2

Visualizes the transformed data

⚙️ Steps Performed
1️⃣ Data Loading

The dataset (iris.data) is read into a Pandas DataFrame with column names:

['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']

2️⃣ Missing Data Handling

The script counts and removes any rows with missing (NaN) values:

df = df.dropna(axis=0, how='any')

3️⃣ Label Encoding

Species names are converted into numeric labels using LabelEncoder:

Species	Encoded Value
setosa	0
versicolor	1
virginica	2
4️⃣ Normalization

Each feature is standardized (zero mean, unit variance) using:

StandardScaler().fit_transform(df)


The script prints before/after statistics for each column:

variance : before X.XX  after 1.00
mean     : before X.XX  after 0.00

5️⃣ PCA Transformation

The first two principal components are extracted:

pca = PCA(n_components=2)
data = pca.fit_transform(standard_df)

6️⃣ Visualization

Scatter Plot (2D PCA Space)
Displays the three species in a reduced 2D space with distinct colors.

Box Plot
Shows value distribution across normalized features.

📊 Example Output
PCA Scatter Plot
<p align="center"> <img src="docs/pca_scatter.png" width="500"> </p>
Box Plot
<p align="center"> <img src="docs/box_plot.png" width="500"> </p>
🧩 Technologies Used
Category	Library
Data Handling	pandas, copy
Preprocessing	sklearn.preprocessing
Analysis	sklearn.decomposition (PCA)
Statistics	statistics
Visualization	matplotlib
🚀 How to Run
# Clone the repository
git clone https://github.com/arianmotti/Iris-PCA.git
cd Iris-PCA

# Run the script
python iris_pca.py


Make sure the dataset file iris.data is in the same directory.

📈 Results
Step	Description	Output
Preprocessing	Normalized & encoded data	Clean DataFrame
PCA	Reduced to 2D	150×2 matrix
Visualization	PCA & Box plots	PNG / Displayed figures
📄 License

Developed by Mohammad Mottaghi
as part of an AI & Data Science practice project.
© 2024 Mohammad Mottaghi
