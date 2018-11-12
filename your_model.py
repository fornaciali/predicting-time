# This is where the interfaces to your model should be. Please edit the functions below and add whatever is needed
# for your model to work


import pickle
import pandas as pd


EVENT_PATIENT_ARRIVED = 'arrived'
EVENT_ASSESSMENT_STARTED = 'assessment initiated'
EVENT_ASSESSMENT_FINISHED = 'assessment concluded'
EVENT_CONSULTATION_STARTED = 'consultation_initiated'
EVENT_CONSULTATION_FINISHED = 'consultation_finished'


def predict_consultation_duration(patient_info):
    """
    Returns the estimation of the consultation time regarding the 
    patient features (original and extended). 
    """
    X_test = pd.DataFrame(patient_info, index=[0])
    X_test = X_test.drop(['assessment_end_time', 'assessment_start_time'], axis=1)		
    pain_level = {'no pain': 0, 'moderate pain': 1, 'severe pain': 2}
    X_test.pain_level.replace(pain_level, inplace=True)
    priority_level = {'normal': 0, 'urgent': 1}
    X_test.priority.replace(priority_level, inplace=True)
    filename = 'best_model.model'
    model = pickle.load(open(filename, 'rb'))
    Y_pred = model.predict(X_test).astype(int)
    return Y_pred	
		

def get_model():
    return None


def get_state_machine():
    return { 'queue': [], 'urgent_queue': [] }	


def get_features(state_machine, patient_id):
    """
    For a patient N, returns a list of estimated consultation duration time for each patient until N-1.  

    Arguments:
    - state_machine (dict) : the data about patients and queues. 
    - patient_id (int) : the ID of the patient in question. 
    """

    features = []
    features.append(state_machine[patient_id]['assessment_end_time'])

    patient_priority = state_machine[patient_id]['priority']

    # Includes the time of consultation of all patients that precede the patient in question
    if patient_priority == 'normal':

        # If you are a NORMAL patient, all the URGENT ones go first.
        for patientID, consultationTime in state_machine['urgent_queue']:
            features.append(consultationTime)

        for patientID, consultationTime in state_machine['queue']:
            features.append(consultationTime)
            if patientID == patient_id:
                break

    elif patient_priority == 'urgent':

        # If you are an URGENT patient, only other URGENT who have arrived earlier goes first
        for patientID, consultationTime in state_machine['urgent_queue']:
            features.append(consultationTime)
            if patientID == patient_id:
                break

    return features


def get_estimate(model, features):
    """
    The estimation time is the sum of the consultation duration time of each patient that goes 
    first of the patient in question. In this function we can ignore the model because it was 
    used in other part of the problem. 
    """
    return sum([int(x) for x in features])


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

        consultation_duration = predict_consultation_duration(state_machine[event.patient])

        # Put the patient in the appropriate queue
        patient4queue = (event.patient, consultation_duration)
        if urgency == 'normal': 
            state_machine['queue'].append(patient4queue)
        elif urgency == 'urgent':
            state_machine['urgent_queue'].append(patient4queue)

    elif event.event == EVENT_CONSULTATION_STARTED:
        state_machine[event.patient]['consultation_start_time'] = event.time

        # Take the first patient in the queue, according to the priority
        patient_priority = state_machine[event.patient]['priority']
        if patient_priority == 'normal':
            state_machine['queue'].pop(0)
        elif patient_priority == 'urgent':
            state_machine['urgent_queue'].pop(0)

    elif event.event == EVENT_CONSULTATION_FINISHED:
        state_machine[event.patient]['consultation_end_time'] = event.time
