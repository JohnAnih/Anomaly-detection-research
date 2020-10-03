# data and numeric analysis
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import (SelectKBest, 
                                       mutual_info_classif, 
                                       f_classif)


# statistical analysis
from statsmodels.graphics import tsaplots

# Import augmented dicky-fuller test function
from statsmodels.tsa.stattools import adfuller


def get_dataset(filename):
    """
    get_dataset loads the dataset into Python

    ----------
    filename : str
        filename
    """    ""
    df = pd.read_excel(filename, 
                       index_col ="Time", 
                       parse_dates=True)
    
    return df



def check_class_dist(df, target):
    """
    check_class_dist checks the distribution of the target variable.

    Parameters
    ----------
    target : str
        target variable

    Returns
    -------
    value_counts
        value counts of the target variable
    """    ""
    class_dist = df[target].value_counts(normalize=True)*100
    return class_dist




def visualize_time_series_vars(df, savefigure=False, file_name=None):
    """
    visualize_time_series_vars visualizes the time series of variables

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame
    """    ""
    # plot the varaibles
    df.plot(subplots=True, 
            layout=(3,3), 
            sharex=False, 
            sharey=False,  
            fontsize=4, 
            legend=True, 
            figsize=(16,10),
            linewidth=0.2)

    # tighten the layout
    plt.tight_layout()

    if (savefigure==True) and (file_name==None):
        raise TypeError('You must pass a name to save the fig')
    
    elif (savefigure==True) and (file_name != None):
        plt.savefig(file_name, transparent=True, dpi=300)



def interpolate_df(df, interpolation_type = 'zero'):
    """
    interpolate_df fills up missing values by linear interpolation

    Parameters
    ----------
    df : DataFrame
        Pandas df
    interpolation : interpolation type
        Linear interpolation

    Returns
    -------
    df
        DataFrame with no missing values
    """    ""

    # Interpolate the missing values
    df_interp = df.interpolate(interpolation_type)
    
    return df_interp



def get_holdout_set(df, target):
    """
    get_holdout_set splits the data to save an holdout set for streaming

    Parameters
    ----------
    df : pandas dataframe
    target: str
         column used as the target variable
    """    ""
    X = df.drop(target, axis=1)
    y = LabelEncoder().fit_transform(df[target])

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, 
                                                        stratify=y, 
                                                        test_size=0.2)
    

    return X_train, X_test, y_train, y_test



def get_stationary_df(df, numb_of_diff=1, fill_na_method='bfill'):
    """
    get_stationary_df makes the time series data stationary

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame
    numb_of_diff : int, optional
        number of differencing, by default 1
    fill_na_method : str, optional
        How to fill in the missing values, by default 'bfill'

    Returns
    -------
    DataFrame
        Diferrenced dataframe by the number of differencing specified, default, 1
    """    ""
    return df.diff(numb_of_diff).fillna(method=fill_na_method)



def perform_adfuller_test(df, verbose=True):
    """
    perform_adfuller_test performs adfuller test.
    
    Hypothesis test:
    H0 - The data is non-stationary
    Ha - The data is stationary

    For more info read here: 
    https://stackoverflow.com/questions/47349422/how-to-interpret-adfuller-test-results

    Parameters
    ----------
    df : DataFrame
        DataFrame to perform the test on
    verbose : bool, optional
        If you want the print messages, by default True

    Returns
    -------
    DataFrame
        ADF Statistic and p-value.
    """    ""
    columns = df.columns
    
    adfuller_test = dict()

    for col_name in columns:
        res = adfuller(df[col_name])
        adfuller_test[col_name] = {'ADF Statistic:': res[0], 'p-value:': res[1]}
        
        if verbose:
            print(f'finished computing for the column: {col_name}')
    
    return pd.DataFrame(adfuller_test)
    


def feature_selection_mutual_info(training_inputs, 
                                  training_output, 
                                  SEED=1, 
                                  show_plots=True, 
                                  savefigure=False, 
                                  figure_name=None):
    """
    feature_selection_mutual_info performs feature importance by learning 
    what inputs are most important to the prediction of the target variable

    Parameters
    ----------
    training_inputs : DataFrame or Numpy arrays
        input variables
    training_output : DataFrame or Numpy arrays
        output variables
    SEED : int, optional
        set the random state for reproduciblity, by default 1
    show_plots : bool, optional
        output the plot, by default True
    savefigure : bool, optional
        save the figure, by default False
    figure_name : str, optional
        file name or filepath if you have a path to save figure, by default None

    Returns
    -------
    Numpy arrays
        results from feature selection with mutual info

    Raises
    ------
    TypeError
        if you want to save figure without passing a figure name to save the figure
    """    ""
    
    X_train, X_test, y_train, y_test = train_test_split(training_inputs, 
                                                        training_output, 
                                                        test_size=0.33, 
                                                        random_state=SEED,
                                                        stratify=training_output)
    
    # configure to select all features
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    
    # learn relationship from training data
    fs.fit(X_train, y_train)
    
    # transform train input data
    X_train_fs = fs.transform(X_train)
    
    # transform test input data
    X_test_fs = fs.transform(X_test)
    
    # put results in a pandas series
    scores_mutual = pd.Series(fs.scores_, index=X_train.columns)
    
    # plot results
    fig, ax = plt.subplots()
    
    scores_mutual.nlargest(10).plot(kind='bar', figsize=(10,8), ax=ax)
    plt.title("Mutual Information Feature Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if show_plots:        
        plt.show()
        
    if (savefigure==True) and (figure_name==None):
        raise TypeError('You must pass a name to save the figure')
    
    elif (savefigure==True) and (figure_name != None):
        fig.savefig(figure_name, transparent=True, dpi=300)
        
    
    return X_train_fs, X_test_fs, fs