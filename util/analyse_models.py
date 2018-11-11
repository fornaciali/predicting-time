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


from sklearn.feature_selection import VarianceThreshold



ml_models = [LinearRegression(), GradientBoostingRegressor(), SGDRegressor(max_iter=1000), 
		KNeighborsRegressor(), GaussianProcessRegressor(), DecisionTreeRegressor(), 
		MLPRegressor(max_iter=1000), SVR(gamma='scale'), RandomForestRegressor(n_estimators=100)]

ml_models_names = ['baseline', 'baseline_plus', 'LinearRegression', 'GradientBoostingRegressor', 
		'SGDRegressor', 'KNeighborsRegressor', 'GaussianProcessRegressor', 'DecisionTreeRegressor', 
		'MLPRegressor', 'SVR', 'RandomForestRegressor', 'LinearRegression w/ extra features', 
		'GradientBoostingRegressor w/ extra features', 'SGDRegressor w/ extra features', 
		'KNeighborsRegressor w/ extra features', 'GaussianProcessRegressor w/ extra features', 
		'DecisionTreeRegressor w/ extra features', 'MLPRegressor w/ extra features', 
		'SVR w/ extra features', 'RandomForestRegressor w/ extra features']

		



# TODO: MOVER PARA OUTRO LUGAR OU APAGAR! 
def feature_selection(data, test_IDs):

	test = data[data['day'].isin(test_IDs)]
	print(test.columns)

	sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
	sel.fit_transform(test)
	new_data = sel.transform(test)
	print(new_data)
	#print(new_data.columns)



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


# TODO: AVALIAR SE USAR ESSA METRICA DE AVALIACAO OU NAO... EH A MESMA DO ANDRE.
#def evaluate(consultation_end_time, assessment_end_time, consultation_duration_time):
#	log_consul = np.array([log(x) for x in consultation_end_time])
#	log_assess = np.array([log(x) for x in assessment_end_time])
#	log_durati = np.array([log(x) for x in consultation_duration_time])
#	return np.sqrt(np.mean( ((log_consul - log_assess) - (log_durati))**2 ))
	

# baseline
def generate_approach_v0(train, test):

	print("	Evaluating approach [baseline]")

	consultation_duration_mean = train.duration.mean()
	
	y_pred = np.array( [consultation_duration_mean for i in range(len(test.duration))] )
	
	return [np.sqrt(metrics.mean_squared_error(test.duration, y_pred))]


# baseline plus
def generate_approach_v1(train, test):

	print("	Evaluating approach [baseline_plus]")

	consultation_duration_ifNormal_mean = train[train.priority == 0].duration.mean() 	
	consultation_duration_ifUrgent_mean = train[train.priority == 1].duration.mean() 
		
	y_pred = np.array([ consultation_duration_ifNormal_mean if i == 0 else 
		consultation_duration_ifUrgent_mean for i in test.priority ])
		
	return [np.sqrt(metrics.mean_squared_error(test.duration.values, y_pred))]


def evaluate_model(model, model_name, X_train, Y_train, X_test, ground_truth):
	print("		Model [" + model_name + "]")
	model.fit(X_train, Y_train)
	Y_pred = model.predict(X_test).astype(int)
	regression = np.sqrt(metrics.mean_squared_error(ground_truth, Y_pred))
	return regression
	

# Machine Learning Tunning 
def generate_approach_v2(train, test):
	
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


# Machine Learning Tunning, com os dados extras 
def generate_approach_v3(train, test):

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
	
	# abre o arquivo, amplia as features e limpa os dados nao numericos
	data = pd.read_csv(args.summary_data_file, index_col=0)
	time = data.assessment_end_time - data.assessment_start_time
	data['assessment_duration'] = time
	time = data.assessment_start_time - data.arrival_time
	data['waiting_4_assessment_duration'] = time
	pain_level = {'no pain': 0, 'moderate pain': 1, 'severe pain': 2}
	data.pain.replace(pain_level, inplace=True)
	priority_level = {'normal': 0, 'urgent': 1}
	data.priority.replace(priority_level, inplace=True)
	

	# obtem as metricas (RMSE) para cada fold
	folds = {}
	for i in range(1,6):
		
		print("Fold ({}): {}".format(i, '-'*50))

		# seleciona os dias de TREINO e TESTE, separando os dados nas 2 particoes
		train_IDs, test_IDs = readFold(args.folds_folder, i)
		train = data[data['day'].isin(train_IDs)]
		test = data[data['day'].isin(test_IDs)]

		# gera e salva as predicoes para cada abordagem
		folds[i] = []
		folds[i] += generate_approach_v0(train, test)
		folds[i] += generate_approach_v1(train, test)
		folds[i] += generate_approach_v2(train, test)
		folds[i] += generate_approach_v3(train, test)

		print()
		
		
	# mostra e consolida (calcula a média dos folds) os resultados por abordagem
	print_results(folds)


if __name__ == '__main__':
	main()