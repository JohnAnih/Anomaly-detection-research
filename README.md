# Detection Of Anamolies in a Time-series data with InfluxDB and Python

## Table of Contents

- [About](#about)
- [Frameworks used for this project](#Libraries_used)
- [Prerequisitest](#Prerequisites)
- [Installation](#installation)

## About <a name = "about"></a>

Analysis of water and environmental data is an important aspect of many intelligent water and environmental system applications where inference from such analysis plays a significant role in decision making. Quite often these data that are collected through sensible sensors can be anomalous due to different reasons such as systems breakdown, malfunctioning of sensor detectors, and more. Regardless of their root causes, such data severely affect the results of the subsequent analysis. 

This repo reports the methods and approaches to data cleaning and preparation for time-series data and further proposes cost-sensitive machine learning algorithms as a solution to detect anomalous data points in a time-series data. 

The following models: Logistic Regression, Random Forest, Support Vector Machines have been modified to support the cost-sensitive learning which penalizes misclassified samples thereby minimizing the total misclassification cost. Our results showed that Random Forest outperformed the rest of the models at predicting the positive class (i.e anomalies). Other methods like data oversampling seem to provide little or no improvement to the Random Forest model. Interestingly, with recursive feature elimination we achieved a better model performance thereby reducing the model complexity. 

Finally, with Influxdb and Kapacitor the data was ingested and streamed to generate new data points to further evaluate the model performance on unseen data, this will allow for early recognition of undesirable changes in the drinking water quality and will enable the water supply companies to rectify on a timely basis whatever undesirable changes abound.

## Frameworks used for this project <a name = "Libraries_used"></a>

The following libraries have been used in Python
* pandas and Numpy for data analysis and numeric analysis.
* influxdb for ingesting and writing data points.
* Kapacitor for streaming and loading new data points
* scikit-learn for classical machine learning.
* matplotlib for plotting and visualization
* statsmodels for adfuller test, autocorrelation and partial autocorrelation
* imbalanced-learn for investing oversampling methods with the best performing model.



### Prerequisites <a name = "Prerequisites"></a>

You need to have influxDB and Kapacitor installed to properly duplicate the environment. Learn more here: https://portal.influxdata.com/downloads/



### Installing <a name = "installation"></a>

To duplicate the enviornment used:

```
pip install -r requirements.txt
```
