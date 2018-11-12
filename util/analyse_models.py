import os
import argparse

import numpy as np
import pandas as pd

from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor	# stochastic gradient descent
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

# The machine learning models to predict the CONSULTATION TIME
ml_models = [LinearRegression(), GradientBoostingRegressor(), SGDRegressor(max_iter=1000), 
		KNeighborsRegressor(), GaussianProcessRegressor(), DecisionTreeRegressor(), 
		MLPRegressor(max_iter=1000), SVR(gamma='scale'), RandomForestRegressor(n_estimators=100)]

# The nomes of machine learning (or not) approaches to model the prediction estimates
ml_models_names = ['baseline', 'baseline_plus', 'LinearRegression', 'GradientBoostingRegressor', 
		'SGDRegressor', 'KNeighborsRegressor', 'GaussianProcessRegressor', 'DecisionTreeRegressor', 
		'MLPRegressor', 'SVR', 'RandomForestRegressor', 'LinearRegression w/ extra features', 
		'GradientBoostingRegressor w/ extra features', 'SGDRegressor w/ extra features', 
		'KNeighborsRegressor w/ extra features', 'GaussianProcessRegressor w/ extra features', 
		'DecisionTreeRegressor w/ extra features', 'MLPRegressor w/ extra features', 
		'SVR w/ extra features', 'RandomForestRegressor w/ extra features']


def get_args():
	parser = argparse.ArgumentParser(description='Analysis of several prediction models, \
		selecting the best to generate the one to be integrated with the final solution. ')
	parser.add_argument('folds_folder', type=str)
	parser.add_argument('summary_data_file', type=str)
	args = parser.parse_args()
	return args


def readFold(folder, fold_ID): 
	file_name = os.path.join(folder, "fold_"+str(fold_ID)+".txt")
	f = open(file_name, "r")
	train   = np.fromstring(f.readline().split('=')[1], dtype=int, sep=',')
	test    = np.fromstring(f.readline().split('=')[1], dtype=int, sep=',')
	f.close()
	return train, test


def generate_approach_v0(train, test):
	"""
	The baseline : return the RMSE considering the prediction as the average 
	consultation time of the training data

	Arguments:
	- train (DataFrame) : the training data
	- test (DataFrame) : the testing data

	"""
	print("	Evaluating approach [baseline]")
	consultation_duration_mean = train.duration.mean()
	y_pred = np.array( [consultation_duration_mean for i in range(len(test.duration))] )
	return [np.sqrt(metrics.mean_squared_error(test.duration, y_pred))]


def generate_approach_v1(train, test):
	"""
	The baseline PLUS : return the RMSE considering the prediction as the average 
	consultation time of the training data, depending on the patient priority 
	(normal or urgent)

	Arguments:
	- train (DataFrame) : the training data
	- test (DataFrame) : the testing data

	"""
	print("	Evaluating approach [baseline_plus]")
	consultation_duration_ifNormal_mean = train[train.priority == 0].duration.mean() 	
	consultation_duration_ifUrgent_mean = train[train.priority == 1].duration.mean() 
	y_pred = np.array([ consultation_duration_ifNormal_mean if i == 0 else 
		consultation_duration_ifUrgent_mean for i in test.priority ])
	return [np.sqrt(metrics.mean_squared_error(test.duration.values, y_pred))]


def evaluate_model(model, model_name, X_train, Y_train, X_test, ground_truth):
	"""
	Given a Machine Learning Regression model (ML), and the training/testing data, 
	evaluate it and return the desired metric (RMSE). 

	Arguments: 
	- model (a sklearn class) : the ML model under evaluation 
	- model_name (str): the name of the ML model under evalution
	- X_train (DataFrame) : the training data
	- Y_train (DataFrame) : the original values of the training data
	- X_test (DataFrame): the testing data
	- ground_truth (DataFrame) : the original values of the training data
	"""
	print("		Model [" + model_name + "]")
	model.fit(X_train, Y_train)
	Y_pred = model.predict(X_test).astype(int)
	regression = np.sqrt(metrics.mean_squared_error(ground_truth, Y_pred))
	return regression
	

def generate_approach_v2(train, test):
	"""
	The Machine Learning : return the RMSE considering the best regression model 
	for the problem among several options (linear regressions, SVM, neural networks, 
	decision trees, ensembles), using just the original features

	Arguments:
	- train (DataFrame) : the training data
	- test (DataFrame) : the testing data

	"""
	print("	Evaluating approaches [Machine Learning]")

	X_train = train.drop(['assessment_duration', 'waiting_4_assessment_duration', 
		'assessment_end_time', 'assessment_start_time', 'consultation_end_time', 
		'consultation_start_time', 'day', 'patient', 'duration'], axis=1)
	Y_train = train['duration']
	X_test  = test.drop(['assessment_duration', 'waiting_4_assessment_duration', 
		'assessment_end_time', 'assessment_start_time', 'consultation_end_time', 
		'consultation_start_time', 'day', 'patient', 'duration'], axis=1)

	results = []
	index = 2
	for model in ml_models: 
		results.append(evaluate_model(model, ml_models_names[index], X_train, 
			Y_train, X_test, test.duration.values))
		index += 1

	return results


def generate_approach_v3(train, test):
	"""
	The Machine Learning PLUS : return the RMSE considering the best regression model 
	for the problem among several options (linear regressions, SVM, neural networks, 
	decision trees, ensembles), using the original and extended features

	Arguments:
	- train (DataFrame) : the training data
	- test (DataFrame) : the testing data

	"""
	print("	Evaluating approaches [Machine Learning], [using extra information]")
	
	X_train = train.drop(['assessment_end_time', 'assessment_start_time', 
		'consultation_end_time', 'consultation_start_time', 'day', 'patient', 
		'duration'], axis=1)
	Y_train = train['duration']
	X_test  = test.drop(['assessment_end_time', 'assessment_start_time', 
		'consultation_end_time', 'consultation_start_time', 'day', 'patient', 
		'duration'], axis=1)

	results = []
	index = 11
	for model in ml_models: 
		results.append(evaluate_model(model, ml_models_names[index], X_train, 
			Y_train, X_test, test.duration.values))
		index += 1

	return results
	

def print_results(folds): 
	print("Results: " + '='*50 + '\n')
	print(' '*16 + "Results in [Root Mean Squared Error] (seconds)\n")
	pd.options.display.float_format = '{:,.2f}'.format
	results = pd.DataFrame.from_dict(folds, orient='index', columns=ml_models_names)
	sorted_results = results.mean().sort_values(ascending=False)
	print(sorted_results)
	print("\n\n===> Concluding, the best model is [{}]".format(sorted_results.tail(1)))


def main():
	args = get_args()
	
	# Open the summary.csv file, expands the features and clears the non-numeric data.
	data = pd.read_csv(args.summary_data_file, index_col=0)
	time = data.assessment_end_time - data.assessment_start_time
	data['assessment_duration'] = time
	time = data.assessment_start_time - data.arrival_time
	data['waiting_4_assessment_duration'] = time
	pain_level = {'no pain': 0, 'moderate pain': 1, 'severe pain': 2}
	data.pain.replace(pain_level, inplace=True)
	priority_level = {'normal': 0, 'urgent': 1}
	data.priority.replace(priority_level, inplace=True)
	

	# Get the metrics (RMSE) for each fold
	folds = {}
	for i in range(1,6):
		
		print("Fold ({}): {}".format(i, '-'*50))

		# Select TRAIN and TEST sets
		train_IDs, test_IDs = readFold(args.folds_folder, i)
		train = data[data['day'].isin(train_IDs)]
		test = data[data['day'].isin(test_IDs)]

		# Generate and stores the predictions for each approach
		folds[i] = []
		folds[i] += generate_approach_v0(train, test)
		folds[i] += generate_approach_v1(train, test)
		folds[i] += generate_approach_v2(train, test)
		folds[i] += generate_approach_v3(train, test)

		print()
		
		
	# Consolidate (calculates the average of the folds) and show the results by approach
	print_results(folds)


if __name__ == '__main__':
	main()