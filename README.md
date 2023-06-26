# HanzeMachineLearning
this repo is about 7 assignments of Hanze university for "Data Science for Life Science" major. The Repository just involves codes and most code has following steps:
-)loading data and exploratory analysis
-)prepare and preprocess data
-)clustering or classification data

# Week01:
Regarding this assignmet, the preprocessing and exploring of data is defiened.
In the project of this assignment which uses breast_cancer dataset, The first step involves preprocessing and exploring the dataset. 
The dataset consists of various features related to a specific problem. For this step, the following tasks were performed:
- Histogram: Histograms were created for each column in the dataset to visualize the distribution of the data.
- The next step involved clustering the data using K-Means and Agglomerative clustering algorithms. 
### K-Means Clustering
- K-Means Model Fitting: K-Means models were fit with cluster values ranging from 1 to 20, and the number of clusters and inertia values were stored for each model.
- Cluster vs. Inertia Plot: A plot was created to visualize the relationship between the number of clusters and the inertia value. This plot helps determine the ideal number of clusters.

### Agglomerative Clustering
- Dendrogram: A dendrogram was created from the agglomerative clustering results to visualize the hierarchical clustering structure.

## Task 3: Classification

The final task involved classification using Random Forest Classifiers. The following tasks were performed:

- Binary Target Variable Creation: A binary target variable, denoting whether the quality is greater than 7 or not, was created.
- Feature Engineering: Two datasets, namely X_with_kmeans and X_without_kmeans, were created by dropping specific columns from the original dataset.
- Random Forest Classification: StratifiedSh


