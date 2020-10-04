# data visualization
import matplotlib.pyplot as plt

# numeric analysis
import pandas as pd
import numpy as np

# Random forest classifier
from sklearn.ensemble import RandomForestClassifier

# oversampling methods
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline

from eval_cost_sensitive_ML import (evaluate_model, 
                                    verbosity_for_summary_stats, 
                                    get_scores)


def RF_model():
	"""
	Random Forest classifier with cost sensitive settings

	Returns
	-------
	model
		Random Forest model
	"""	""
	return RandomForestClassifier(n_estimators=1000, 
	                              class_weight='balanced',
	                              random_state=1,
	                              n_jobs=-1)
	                              


def get_oversampling_models(SEED=1):
	"""
	get_oversampling_models gets oversampling models

	Parameters
	----------
	SEED : int, optional
		set seed for reproduciblity, by default 1

	Returns
	-------
	models
		models

	"""""
	
	models, names = list(), list()
	
	# RandomOverSampler
	models.append(RandomOverSampler(random_state=SEED))
	names.append('ROS')
	
	# SMOTE
	models.append(SMOTE(random_state=SEED))
	names.append('SMOTE')
	
	# BorderlineSMOTE
	models.append(BorderlineSMOTE())
	names.append('BLSMOTE')
	
	# SVMSMOTE
	models.append(SVMSMOTE())
	names.append('SVMSMOTE')
	
	# ADASYN
	models.append(ADASYN())
	names.append('ADASYN')
	
	return models, names
	



def run_oversampling_models(X,y, verbose=True, show_plots=True,  savefigure=False,  figure_name=None):
	"""
	run_oversampling_models evaluates oversampling methods with Random Forest

	Parameters
	----------
	X : DataFrame/Numpy arrays
		inputs
	y : DataFrame/Numpy arrays
		output
	verbose : bool, optional
		option to see print messages, by default True
	show_plots : bool, optional
		option to show plots as results, by default True
	savefigure : bool, optional
		option to save figure, by default False
	figure_name : str, optional
		name of the figure, by default None

	Returns
	-------
	Numpy arrays
		F1 Score, F0.5 Score, sensitivity and specificity

	Raises
	------
	TypeError
		When the user chooses to save but failed to specify a figure name
	"""    ""
    
	models, names = get_oversampling_models()
	results = list()
    
	# evaluate each model
	for i in range(len(models)):

	    model = RF_model()
	
	    # define the pipeline steps
	    steps = [('o', models[i]), ('m', model)]
	
	    # define the pipeline
	    pipeline = Pipeline(steps=steps)
	
	    # evaluate the model and store results
	    scores = evaluate_model(X, y, pipeline)
	
	    results.append(scores)
	
	    if verbose:
	        # summarize and store
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
	plt.suptitle('Evaluation of Oversampling methods with Random Forest model', y=1.08)

	if show_plots:
	    plt.show()
	
	if (savefigure==True) and (figure_name==None):
	    raise TypeError('You must pass a name to save the figure')

	elif (savefigure==True) and (figure_name != None):
	    fig.savefig(figure_name, transparent=True, dpi=300)
	
	return f1_scores, f0_5_scores, sensitivity, specificity
            