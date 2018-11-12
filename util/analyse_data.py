import os
import argparse

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


def get_args():
    parser = argparse.ArgumentParser(description='Generating graphics.')
    parser.add_argument('summary_data_file', type=str)
    args = parser.parse_args()
    return args


def save_info(file_path_and_name, info):
    f = open(file_path_and_name, 'w')
    f.write(info)
    f.close()


# Load data
print("Loading...")
args = get_args()
data = pd.read_csv(args.summary_data_file)
print("Done!")


# Extend the data with extra features
data = pd.read_csv(args.summary_data_file, index_col=0)
time = data.assessment_end_time - data.assessment_start_time
data['assessment_duration'] = time
time = data.assessment_start_time - data.arrival_time
data['waiting_4_assessment_duration'] = time
pain_level = {'no pain': 0, 'moderate pain': 1, 'severe pain': 2}
data.pain.replace(pain_level, inplace=True)
priority_level = {'normal': 0, 'urgent': 1}
data.priority.replace(priority_level, inplace=True)

# View and save consolidated statistics
#print(data.describe())
#save_info("stats.txt", data.describe().to_string())

print("Arrival time correlation:")
print(np.corrcoef(data.arrival_time, data.duration))
plt.scatter(data.arrival_time, data.duration)
plt.title("ARRIVAL TIME x CONSULTATION DURATION")
plt.show(block=True)

print("Time for assessment correlation:")
print(np.corrcoef(data.assessment_start_time - data.arrival_time, data.duration))
plt.scatter(data.assessment_start_time - data.arrival_time, data.duration)
plt.title("TIME FOR ASSESSMENT x CONSULTATION DURATION")
plt.show(block=True)

print("Temperature correlation:")
print(np.corrcoef(data.temperature, data.duration))
plt.scatter(data.temperature, data.duration)
plt.title("TEMPERATURE x CONSULTATION DURATION")
plt.show(block=True)

print("Pain level correlation:")
print(np.corrcoef(data.pain, data.duration))
plt.scatter(data.pain, data.duration)
plt.title("PAIN LEVEL x CONSULTATION DURATION")
plt.show(block=True)

print("Priority correlation:")
print(np.corrcoef(data.priority, data.duration))
plt.scatter(data.priority, data.duration)
plt.title("PRIORITY x CONSULTATION DURATION")
plt.show(block=True)

print("Assessment duration correlation:")
print(np.corrcoef(data.assessment_end_time - data.assessment_start_time, data.duration))
plt.scatter(data.assessment_end_time - data.assessment_start_time, data.duration)
plt.title("ASSESSMENT DURATION x CONSULTATION DURATION")
plt.show(block=True)

print("Time for consultation correlation:")
print(np.corrcoef(data.consultation_start_time - data.assessment_end_time, data.duration))
plt.scatter(data.consultation_start_time - data.assessment_end_time, data.duration)
plt.title("TIME FOR CONSULTATION x CONSULTATION DURATION")
plt.show(block=True)

x = list(data['arrival_time'])
n, bins, patches = plt.hist(x, 40, density=False, color='b')
plt.xlabel('Arrival Time')
plt.ylabel('Quantity')
plt.title('HISTOGRAM OF ARRIVAL TIME')
plt.axis([-500, 15000, 0, 200])
plt.grid(True)
plt.show()
