
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from general_funcs import (get_dataset, 
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

from influxdb_operations import automate_data_process

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

# Evaluate the model on unseen data
print(classification_report(test_inputs, predictions))


# deploy and test on streamed data
automate_data_process(model=model, inputs=test_inputs, output=y_test)
