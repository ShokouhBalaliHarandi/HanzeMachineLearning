# HanzeMachineLearning
this repo includes about 7 assignments of Datascience3 course related to supervised and unsupervised machine learning methods for "Data Science for Life Science" major in the Hanze university. The Repository just involves codes and most code has following steps:
-) loading data and exploratory analysis
-) prepare and preprocess data
-) clustering or classification data

All assignments are developed in jupyter notebooks with Python 3.9.13. and most libraries used in them are:
- scikit-learn
- numpy
- pandas
- matplotlib
- sklearn

The codes of this repository are under MIT License.

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
This assignment focuses on implementing the gradient descent algorithm to find the optimal parameters for a linear regression model. The task involves loading housing data, performing gradient descent, and analyzing the convergence of the cost function. The steps of this assignment are:
Load the data, Scatter plot to visualize the relationship between the size of the houses and their prices. The scatter plot should display the sizes on the horizontal axis and the prices on the vertical axis, Compute the cost, Gradient descent, Plot cost values: Use the `compute_cost` function to calculate the cost at each iteration of gradient descent. Plot the cost values against the iteration numbers to analyze the convergence behavior, Analyze the plot, Comparison with sklearn

## Week05:
### Week05-Cancer.ipynb
This assignment involves implementing and analyzing Support Vector Machines (SVM) using different kernels to classify the moon dataset. The goal is to explore the behavior of SVM with various kernel configurations and understand their impact on the separation of the moon dataset. After preprocessing and modeling data, evaluation is done with several metrics like Classification Report, AUC ROC and Precision-Recall Curve.
### Week05-Moon.ipynb
This assignment involves implementing and analyzing Support Vector Machines (SVM) using different kernels to classify the moon dataset. The goal is to explore the behavior of SVM with various kernel configurations and understand their impact on the separation of the moon dataset.

## Week06:
This week assignments contains the code for two assignments related to the breast cancer and moon dataset. The assignments involve working with scikit-learn, performing data analysis, preprocessing, modeling, and evaluation using different algorithms and parameters. 
### Week06-BreastCancer.ipynb:
The goal is to add several new models with different parameters to the classifiers variable provided in the notebook. The scikit-learn library is used to work with decision trees and naive Bayes classifiers. 
### Week06-Corona.ipynb
This assignment focuses on preprocessing the moon dataset by addressing skewness and normality of the features.

## Week07:
This assignment focuses on building and evaluating different classifiers for breast cancer classification using the breast cancer dataset. The classifiers implemented include Bagging, Boosting, and a Dummy Classifier. The scikit-learn library is utilized for model creation and evaluation.
