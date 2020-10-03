from pickle import dump
from pickle import load

import urllib.request

import time

from sklearn.metrics import f1_score

from general_funcs import *

from general_funcs import (get_dataset, 
                           check_class_dist, 
                           interpolate_df, 
                           visualize_time_series_vars, 
                           get_holdout_set, 
                           get_stationary_df, 
                           perform_adfuller_test, 
                           feature_selection_mutual_info)

from eval_cost_sensitive_ML import (run_models, 
                                    RF_feature_importance,
                                    run_rfe_feature_importance)

from eval_oversampling_methods import run_oversampling_models

from influxdb_operations import (write_points_to_influxdb, 
                                 transform_to_dataframe)

# matplotlib style
plt.style.use("fivethirtyeight") 

# filename
filename = "water_data.xlsx"

# target variable
target = "EVENT"

# load the data
water_data = get_dataset(filename=filename)


# visualize the missing values 
visualize_time_series_vars(df=water_data, 
                           savefigure=True, 
                           file_name='./Images/time-series.png')


# divide set into training and testing. The test here would be the holdout set.
X_train, X_test, y_train, y_test = get_holdout_set(df=water_data, target=target)


# fill missing values by linear interpolation
training_inputs = interpolate_df(df=X_train)

# make the time series stationary
training_inputs = get_stationary_df(df= training_inputs)


# visualize stationary time series
visualize_time_series_vars(df=training_inputs, 
                           savefigure=True, 
                           file_name='./Images/stationary-time-series.png')


# perform adfuller test
adfuller_test = perform_adfuller_test(df=training_inputs)

# understand the test results
print(adfuller_test)


# perform feature importance with mutual information
X_train_fs, X_test_fs, fs = feature_selection_mutual_info(training_inputs=training_inputs, 
                                                          training_output=y_train, 
                                                          savefigure=True, 
                                                          figure_name='./Images/feature-selection-mutual-info.png')


# run cost sensitive ml models
f1_scores, f0_5_scores, sensitivity, specificity= run_models(X=training_inputs, 
                                                             y=y_train, 
                                                             savefigure=True,
                                                             figure_name='./Images/model-evualtion.png')




# Evaulate oversampling methods -- caution, this takes a lot of time to run
f1, f0_5, sensitivity_o, specificity_o = run_oversampling_models(X=training_inputs,
                                                                 y=y_train)



# Perform Recursive feature elimination  -- caution, this takes a lot of time to run
run_rfe_feature_importance(X=training_inputs, y=y_train)


# Random Forest feature importance
RF_feature_importance(inputs=training_inputs, outputs=y_train)


# stage 2
# Selected model
model = RandomForestClassifier(n_estimators=1000, 
                               class_weight='balanced',
                               random_state=1,
                               n_jobs=-1)


# train the model
model.fit(training_inputs, y_train)

# apply the same data preprosing for the test input datasets

# fill missing values by linear interpolation
test_inputs = interpolate_df(df=X_test)

# make the time series stationary
test_inputs = get_stationary_df(df= test_inputs)

# make predictions
predictions = model.predict(test_inputs)

# check F1 scire
f1_score(y_test, predictions)


# save model
filename = 'finalized_model.sav'
dump(model, open(filename, 'wb'))




# load saved model
loaded_model = load(open(filename, 'rb'))


while True:
    
    url = write_points_to_influxdb(X_test=test_inputs, y_test=y_test)
    recent_1 = urllib.request.urlopen(url).read()
    recent_2 = urllib.request.urlopen(url).read()
    
    while recent_1 == recent_2:
        print("No New Data")
        time.sleep(5)
        recent_2 = urllib.request.urlopen(url).read()
        
        data = transform_to_dataframe(recent_2)
        
        x= data.drop(target, axis=1)
        
        y_true= data[target]
        
        # fill missing values by linear interpolation
        x = interpolate_df(df=x)
        
        # make the time series stationary
        x = get_stationary_df(df= x)
        
        yhat = loaded_model.predict(x)
        
        f1_score(y_true, yhat)