import os
import argparse

import numpy as np
import pandas as pd
import pickle 

from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor

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
	pain_level = {'no pain': 0, 'moderate pain': 1, 'severe pain': 2}
	data.pain.replace(pain_level, inplace=True)
	priority_level = {'normal': 0, 'urgent': 1}
	data.priority.replace(priority_level, inplace=True)
	
	# remove as features nao necessarias para gerar o modelo
	X_train = data.drop(['assessment_end_time', 'assessment_start_time', 
		'consultation_end_time', 'consultation_start_time', 'day', 'patient', 
		'duration'], axis=1)
	Y_train = data['duration']
	
	# gera e persiste o modelo oficial
	print("Generating the model...")
	model = GradientBoostingRegressor()
	model.fit(X_train, Y_train)
	pickle.dump(model, open('best_model.model', 'wb'))
	print("Done! The model file is [best_model.model].")


if __name__ == "__main__":
	main()