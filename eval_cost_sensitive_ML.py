# data visualization
import matplotlib.pyplot as plt

# save and load ml models
from pickle import dump
from pickle import load

# data and numeric analysis
import pandas as pd
import numpy as np

# ML
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (cross_validate,                   
                                     cross_val_score,
                                     RepeatedStratifiedKFold)
                                     
from sklearn.feature_selection import RFE

from sklearn.metrics import (fbeta_score,
                             f1_score,
                             make_scorer,
                             recall_score)




def get_cost_sensitive_models():
    """
    get_cost_sensitive_models returns three cost sensitive models and their names
    - Logistic Regression
    - Random Forest
    - Support Vector Machines

    """    ""
    models, names = list(), list()
    
    # Logistic Regression
    models.append(LogisticRegression(solver='lbfgs', 
                                     class_weight='balanced', 
                                     random_state=1,
                                     n_jobs=-1))
    
    names.append('LR')
    
    # Random Forest
    models.append(RandomForestClassifier(n_estimators=1000, 
                                         class_weight='balanced',
                                         random_state=1,
                                         n_jobs=-1))
    
    names.append('RF')
    
    # Support Vector Machine SVM
    models.append(SVC(gamma='scale', 
                      class_weight='balanced', 
                      max_iter=10000))
    
    names.append('SVM')
    
    return models, names




def f05_measure(y_true, y_pred):
    """
    f05_measure calculates F0.5 score

    Parameters
    ----------
    y_true : numpy arrays
        actual observations
    y_pred : numpy arrays
        predicted values
    """    ""
    return fbeta_score(y_true, y_pred, beta=0.5)




def evaluate_model(X, y, model):
    """
    evaluate_model evaluates the ML models

    Parameters
    ----------
    X : DataFrame or Numpy arrays
        input variables
    y : DataFrame or Numpy arrays
        output variable
    model : model
        model

    Returns
    -------
    Numpy arrays
        scores after each runs
    """    ""
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # define the model evaluation metric
    metric = {
              'f1_score': make_scorer(f1_score),
              'f05_measure': make_scorer(f05_measure),
              'sensitivity': make_scorer(recall_score),
              'specificity': make_scorer(recall_score,pos_label=0)
              }
    
    # evaluate model
    scores = cross_validate(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
    
    return scores
 



def get_best_models_RFE():
    """
    get_best_models_RFE gets contribution of each feature towards the target
    via recursive feature elimintation

    Returns
    -------
    models
        model for each feature

    """""

    models = {}
    
    for i in range(2,10):
        model = RandomForestClassifier()
        
        rfe = RFE(estimator=model, n_features_to_select=i)
                                       
        models[str(i)] = Pipeline(steps=[('s', rfe), ('m', model)])
        
        return models





def evaluate_model_RFE(model, X, y):
    """
    evaluate_model_RFE evaluates the contribution of each feature

    [extended_summary]

    Parameters
    ----------
    model : model
        model for each feature
    X : DataFrame or Numpy arrays
        inputs
    y : DataFrame or Numpy arrays
        outputs

    Returns
    -------
    Numpy arrays
        scores

    """""
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    scores = cross_val_score(model, 
                             X, y, 
                             scoring=make_scorer(f1_score), 
                             cv=cv,  
                             n_jobs=-1)
    
    return scores





def run_rfe_feature_importance(X, y, 
                               verbose=True, 
                               showplots=True, 
                               savefigure=False,
                               figure_name=None):
    
    """
     Performs Recursive Feature Elimination 

    Raises
    ------
    TypeError
        When user specified to save figure without providing a figure name
    """    ""
                               
    # get the models to evaluate
    models = get_best_models_RFE()
    
    # evaluate the models and store results
    results, names = list(), list()
    
    for name, model in models.items():
    	scores = evaluate_model_RFE(model, X, y)
    	results.append(scores)
    	names.append(name)
    	
    	if verbose:
    	   print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
    	                              
    
    # create the figure
    fig, ax = plt.subplots(figsize=(12,10))
    
    # plot model performance for comparison
    ax.boxplot(results, labels=names, showmeans=True)
    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Number Of Features')
    
    plt.tight_layout()
    
    if showplots:
        plt.show()
        
    if (savefigure==True) and (figure_name==None):
        raise TypeError('You must pass a name to save the figure')
    
    elif (savefigure==True) and (figure_name != None):
        fig.savefig(figure_name, transparent=True, dpi=300)
        



def RF_feature_importance(inputs, 
                          outputs, 
                          saveplots=True, 
                          figure_name ='./Images/rf_feature_imp.png'):
    """
    RF_feature_importance shows feature importance of each input variable

    Parameters
    ----------
    inputs : DataFrame
        Input data
    outputs : Numpy array
        Output data
    saveplots : bool, optional
        option to save plots, by default True
    figure_name : str, optional
        name of the plot to save, by default 'rf_feature_imp.png'
    """    ""
    
    model = RandomForestClassifier()
    
    trained_model= model.fit(inputs, outputs)
    
    feature_importance = pd.Series(trained_model.feature_importances_ ,
                                   index= inputs.columns)
    
    feature_importance.nlargest(20).plot(kind='bar', figsize=(10,8))
    
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    plt.show()
    
    if saveplots:
        plt.savefig(figure_name, transparent=True, dpi=300)
        


def verbosity_for_summary_stats(i, scores, names):
    """
    verbosity_for_summary_stats ouputs the mean and std of the models.
    This function is used in the loop to print out messages of the summary stats.

    Parameters
    ----------
    i : int
        position in the loop
    scores : Numpy arrays
        position in the loop
    names : str
        position in the loop
    """    ""
    # summarize and store
    print('> F1 Score %s %.3f (%.3f)' % (names[i], 
                                         np.mean(scores['test_f1_score']), 
                                         np.std(scores['test_f1_score'])))
    
    print('> F0.5 Score %s %.3f (%.3f)' % (names[i],
                                           np.mean(scores['test_f05_measure']),
                                           np.std(scores['test_f05_measure'])))
    
    print('> Sensitivity Score %s %.3f (%.3f)' % (names[i], 
                                                  np.mean(scores['test_sensitivity']), 
                                                  np.std(scores['test_sensitivity'])))
    
    print('> Specificity Score %s %.3f (%.3f)' % (names[i], 
                                                  np.mean(scores['test_specificity']),
                                                  np.std(scores['test_specificity'])))
    



def get_scores(results):
    """
    get_scores calculates the F1 Score, F0.5 Scores, Sensitivity and Specificity.

    Parameters
    ----------
    results : Numpy arrays
        the results from evaluation of the models

    Returns
    -------
    Numpy arrays
        F1 Scores, F0.5 Scores, Sensitivity and Specificity
    """    ""
    
    scores = list()
    score_names = ['test_f1_score', 
                   'test_f05_measure', 
                   'test_sensitivity', 
                   'test_specificity']
    
    for test_scores in score_names:
        values = [item[test_scores] for item in results if test_scores in item.keys()]
        scores.append(values)
        
    return scores[0], scores[1], scores[2], scores[3]
        



def run_models(X, 
               y, 
               verbose=True, 
               show_plots=True, 
               savefigure=False, 
               figure_name=None):
    """
    run_models runs the cost sensitive models

    Parameters
    ----------
    X : DataFrame or Numpy arrays
        inputs
    y : DataFrame or Numpy arrays
        output
    verbose : bool, optional
        option to see or remove print messages, by default True
    show_plots : bool, optional
        Option to show plots, by default True
    savefigure : bool, optional
        Option to save the plot, by default False
    figure_name : str, optional
        name of the figure, by default None
    """    ""

    # define models
    models, names = get_cost_sensitive_models()
    results = list()
    
    # evaluate each model
    for i in range(len(models)):
        # evaluate the pipeline and store results
        scores = evaluate_model(X, y, models[i])
        results.append(scores)
        
        if verbose:
            verbosity_for_summary_stats(i=i, scores=scores, names=names)

    
    f1_scores, f0_5_scores, sensitivity, specificity = get_scores(results=results)
    
    # create the figure
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,10))
    
    ax[0,0].boxplot(f1_scores, labels=names, showmeans=True)
    ax[0,0].set_ylabel('F1 Scores')
    
    ax[0,1].boxplot(f0_5_scores, labels=names, showmeans=True)
    ax[0,1].set_ylabel('F0.5 Scores')
    
    ax[1,0].boxplot(sensitivity, labels=names, showmeans=True)
    ax[1,0].set_ylabel('Sensitivity scores')
    
    ax[1,1].boxplot(specificity, labels=names, showmeans=True)
    ax[1,1].set_ylabel('Specificity scores')
    
    plt.tight_layout()
    plt.suptitle("Evaluation of cost sensitive models", y=1.08)
    
    if show_plots:
        
        plt.show()
        
    if (savefigure==True) and (figure_name==None):
        raise TypeError('You must pass a name to save the figure')
    
    elif (savefigure==True) and (figure_name != None):
        fig.savefig(figure_name, transparent=True, dpi=300)
        
    return f1_scores, f0_5_scores, sensitivity, specificity
        
        
        

def save_model(model, filename = 'finalized_model.sav'):
    """
    save_model saves the finalized ML model for later use.

    Parameters
    ----------
    model : model
        The model to save
    filename : str, optional
        name to save the model, by default 'finalized_model.sav'
    """    ""
    dump(model, open(filename, 'wb'))
    
    
    

def load_model(filename = 'finalized_model.sav'):
    """
    load_model loads the already saved model

    Parameters
    ----------
    filename : str, optional
        name of the model you want to load, by default 'finalized_model.sav'

    Returns
    -------
    model
        Loads the model.
    """    ""
    return load(open(filename, 'rb'))