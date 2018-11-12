import os
import argparse

import numpy as np
import pandas as pd
import pickle 

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


def get_args():
	parser = argparse.ArgumentParser(description='Generates the models for \
		alternative evaluation, based on the model analysis done internally to identify \
		the best approach for waiting time on queue and consultation duration.')
	parser.add_argument('summary_data_file', type=str)
	args = parser.parse_args()
	return args


def main(): 

	args = get_args()
	
	# Opens the summary.csv file, expands the features and clears the non-numeric data.
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
	
	# Remove the unnecessary features to generate the model.
	X_train = data.drop(['assessment_end_time', 'assessment_start_time', 
		'consultation_end_time', 'consultation_start_time', 'day', 'patient', 
		'duration', 'waiting_4_consultation_duration'], axis=1)

	# Generates and persists the model for QUEUEING TIME.
	print("Generating QUEUE model...")
	Y_train = data['waiting_4_consultation_duration']
	model = LinearRegression()
	model.fit(X_train, Y_train)
	pickle.dump(model, open('queue.model', 'wb'))
	print("Done! The QUEUE model file is [queue.model].")
	
	# Generates and persists the model for CONSULTATION DURATION.
	print("Generating the CONSULTATION model...")
	Y_train = data['duration']
	model = GradientBoostingRegressor()
	model.fit(X_train, Y_train)
	pickle.dump(model, open('consultation.model', 'wb'))
	print("Done! The CONSULTATION model file is [consultation.model].")


if __name__ == "__main__":
	main()