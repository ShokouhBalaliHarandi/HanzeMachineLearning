# HanzeMachineLearning
this repo is about 7 assignments of Hanze university for "Data Science for Life Science" major. The Repository just involves codes and most code has following steps:
-)loading data and exploratory analysis
-)prepare and preprocess data
-)clustering or classification data

## Week01:
Regarding this assignmet, the preprocessing and exploring of data is defiened.
In the project of this assignment which uses breast_cancer dataset, The first step involves preprocessing and exploring the dataset. 
The dataset consists of various features related to a specific problem. For this step, the following tasks were performed:
- Histogram: Histograms were created for each column in the dataset to visualize the distribution of the data.
- The next step involved clustering the data using K-Means and Agglomerative clustering algorithms. 
- The final task involved classification using Random Forest Classifiers.

## Week02:
### Week02-KmeanSpect for categorical data.ipynb
This assignment focuses on comparing the performance of k-means and spectral clustering algorithms and evaluating the visualization techniques of PCA and t-SNE after loading, exploring data and preprocessing data. As independent data are categorical so they reformated and plots of the data based on formatted data. This assignment is done because I wanted to test how we should used categorical data in unsupervised methods.
the source of data is from: https://ec.europa.eu/
### Week02-Fitting for BreastCancer.ipynb
This assignment focuses on comparing the performance of k-means and spectral clustering algorithms and evaluating the visualization techniques of PCA and t-SNE. The goal is to identify the best combination of clustering method and visualization technique for a given dataset. The assignment involves preprocessing the dataset, defining a pipeline with preprocessing and clustering, performing hyperparameter tuning using GridSearchCV, evaluating the performance of models, and selecting the best combination.
### Week02-Text Detection.ipynb
This assignment involves applying text clustering techniques to a given dataset. The goal is to cluster similar pieces of text together and gain insights from the clusters.
The data involves 25 top news in recent years. Date: The date of the news headlines. Top1 to Top25: The news headlines for each date.
the source of data is: https://www.refinitiv.com/

## Week03:
This assignment focuses on improving the data quality for anomaly detection in time series data. The goal is to remove outliers due to sensor reading errors while preserving anomalies and perform resampling or aggregation to reduce noise and highlight higher-level patterns relevant for anomaly detection.
Steps to Improve Data Quality: 
Handling Missing Values, Outlier Removal, Resampling or Aggregation, Anomaly Detection Algorithms.
The improved dataset will be used to evaluate the performance of various anomaly detection algorithms. Common algorithms such as Local Outlier Factor, Isolation Forest, One-Class Support Vector Machine, and Robust Covariance will be applied to detect anomalies in the time series data.

## Week04:
