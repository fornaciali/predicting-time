# This is where the interfaces to your model should be. Please edit the functions below and add whatever is needed
# for your model to work


import pickle
import pandas as pd


EVENT_PATIENT_ARRIVED = 'arrived'
EVENT_ASSESSMENT_STARTED = 'assessment initiated'
EVENT_ASSESSMENT_FINISHED = 'assessment concluded'
EVENT_CONSULTATION_STARTED = 'consultation_initiated'
EVENT_CONSULTATION_FINISHED = 'consultation_finished'
	

def get_model():
    model_queue = pickle.load(open('queue.model', 'rb'))
    model_consultation = pickle.load(open('consultation.model', 'rb'))
    return model_queue, model_consultation


def get_state_machine():
    return {}	


def get_features(state_machine, patient_id):
    patient_info = state_machine[patient_id]
    assessment_end_time = patient_info['assessment_end_time']
    X_test = pd.DataFrame(patient_info, index=[0])
    X_test = X_test.drop(['assessment_end_time', 'assessment_start_time'], axis=1)        
    pain_level = {'no pain': 0, 'moderate pain': 1, 'severe pain': 2}
    X_test.pain_level.replace(pain_level, inplace=True)
    priority_level = {'normal': 0, 'urgent': 1}
    X_test.priority.replace(priority_level, inplace=True)
    return X_test, assessment_end_time


def get_estimate(model, features):
    """
    Return the estimation time for a patient leave the clinic, that is: the time 
    the assessment ended + a prediction of the waiting time on queue + a prediction 
    of the consultation duration. 

    Arguments: 
    - model : a tuple of the model to predict the waiting time and the model to predict the consultation time
    - features : a tuple of the features used to estimate the times on each model and the assessment end time
    """

    model_queue, model_consultation = model
    f, assessment_end_time = features
    return int(assessment_end_time) + model_queue.predict(f).astype(int) + model_consultation.predict(f).astype(int)  


def update_state(state_machine, event):
    if event.event == EVENT_PATIENT_ARRIVED:
        state_machine[event.patient] = {}
        state_machine[event.patient]['arrival_time'] = event.time

    elif event.event == EVENT_ASSESSMENT_STARTED:
        state_machine[event.patient]['assessment_start_time'] = event.time

    elif event.event == EVENT_ASSESSMENT_FINISHED:
        urgency, temperature, pain_level = event.assessment.split('|')
        state_machine[event.patient]['pain_level'] = pain_level
        state_machine[event.patient]['priority'] = urgency
        state_machine[event.patient]['temperature'] = float(temperature)
        state_machine[event.patient]['assessment_end_time'] = event.time

        time = state_machine[event.patient]['assessment_end_time'] - state_machine[event.patient]['assessment_start_time']
        state_machine[event.patient]['assessment_duration'] = time

        time = state_machine[event.patient]['assessment_start_time'] - state_machine[event.patient]['arrival_time']
        state_machine[event.patient]['waiting_4_assessment_duration'] = time
       
    elif event.event == EVENT_CONSULTATION_STARTED:
        state_machine[event.patient]['consultation_start_time'] = event.time

    elif event.event == EVENT_CONSULTATION_FINISHED:
        state_machine[event.patient]['consultation_end_time'] = event.time

