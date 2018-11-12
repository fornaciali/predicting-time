import os
import argparse

import numpy as np
import pandas as pd
import pickle 

from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

def get_args():
	parser = argparse.ArgumentParser(description='Generates the official model for \
		final evaluation, based on the model analysis done internally to identify \
		the best approach to the problem.')
	parser.add_argument('summary_data_file', type=str)
	args = parser.parse_args()
	return args


def main(): 

	args = get_args()
	
	# abre o arquivo, amplia as features e limpa os dados nao numericos
	print("Reading data...")
	data = pd.read_csv(args.summary_data_file, index_col=0)
	time = data.assessment_end_time - data.assessment_start_time
	data['assessment_duration'] = time
	time = data.assessment_start_time - data.arrival_time
	data['waiting_4_assessment_duration'] = time
	time = data.consultation_start_time - data.assessment_end_time
	data['waiting_4_consultation_duration'] = time
	pain_level = {'no pain': 0, 'moderate pain': 1, 'severe pain': 2}
	data.pain.replace(pain_level, inplace=True)
	priority_level = {'normal': 0, 'urgent': 1}
	data.priority.replace(priority_level, inplace=True)
	
	
	# remove as features nao necessarias para gerar o modelo
	X_train = data.drop(['assessment_end_time', 'assessment_start_time', 
		'consultation_end_time', 'consultation_start_time', 'day', 'patient', 
		'duration', 'waiting_4_consultation_duration'], axis=1)

	
	# gera e persiste o modelo para TEMPO DE FILA. 
	print("Generating QUEUE model...")
	Y_train = data['waiting_4_consultation_duration']
	model = LinearRegression()
	model.fit(X_train, Y_train)
	pickle.dump(model, open('queue.model', 'wb'))
	print("Done! The QUEUE model file is [queue.model].")


	#queue_duration_ifNormal_mean = data[data.priority == 0].waiting_4_consultation_duration.mean() 	
	#queue_duration_ifUrgent_mean = data[data.priority == 1].waiting_4_consultation_duration.mean()
	#f = open('best_model_queue.model', 'w')
	#f.write(str(queue_duration_ifNormal_mean) + ";" + str(queue_duration_ifUrgent_mean))
	#f.close()

	# remove as features nao necessarias para gerar o modelo
	#X_train = data.drop(['assessment_end_time', 'assessment_start_time', 
	#	'consultation_end_time', 'consultation_start_time', 'day', 'patient', 
	#	'duration', 'waiting_4_consultation_duration'], axis=1)

	
	# gera e persiste o modelo para DURACAO DA CONSULTA
	print("Generating the CONSULTATION model...")
	Y_train = data['duration']
	model = GradientBoostingRegressor()
	model.fit(X_train, Y_train)
	pickle.dump(model, open('consultation.model', 'wb'))
	print("Done! The CONSULTATION model file is [consultation.model].")

	print(X_train.columns)


if __name__ == "__main__":
	main()