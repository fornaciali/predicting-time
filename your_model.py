# This is where the interfaces to your model should be. Please edit the functions below and add whatever is needed
# for your model to work


import pickle
import pandas as pd


EVENT_PATIENT_ARRIVED = 'arrived'
EVENT_ASSESSMENT_STARTED = 'assessment initiated'
EVENT_ASSESSMENT_FINISHED = 'assessment concluded'
EVENT_CONSULTATION_STARTED = 'consultation_initiated'
EVENT_CONSULTATION_FINISHED = 'consultation_finished'


def predict_consultation_duration(patient_info, patient_priority=''):

    patient_info['assessment_duration'] = patient_info['assessment_end_time'] - patient_info['assessment_start_time']
    patient_info['waiting_4_assessment_duration'] = patient_info['assessment_start_time'] - patient_info['arrival_time']

    X_test = pd.DataFrame(patient_info, index=[0])
    X_test = X_test.drop(['assessment_end_time', 'assessment_start_time'], axis=1)
		
    pain_level = {'no pain': 0, 'moderate pain': 1, 'severe pain': 2}
    X_test.pain_level.replace(pain_level, inplace=True)
	
    priority_level = {'normal': 0, 'urgent': 1}
    X_test.priority.replace(priority_level, inplace=True)
	
    filename = 'best_model.model'
    model = pickle.load(open(filename, 'rb'))
    Y_pred = model.predict(X_test).astype(int)
    
    #if patient_priority == 'normal':
    #    return 474
    #elif patient_priority == 'urgent':
    #    return 1014
    #else:
    #    return 629

    return Y_pred	
		

def get_model():
    return None


def get_state_machine():
    #return {'time': 0}

    # provisoriamente, serah uma maquina vazia
    return {
            #'time': 0,					# TODO: remover essa linha (apenas para debug)
    		'queue': [],
    		'urgent_queue': []
            }	


def get_features(state_machine, patient_id):
    #return state_machine['time']

    #if len(state_machine['urgent_queue']) > 1:
    #	print("Fila urgente: {}".format(len(state_machine['urgent_queue'])))
    #if len(state_machine['queue']) > 1:
   # 	print("Fila normal : {}".format(len(state_machine['queue'])))

    features = []
    features.append(state_machine[patient_id]['assessment_end_time'])

    patient_priority = state_machine[patient_id]['priority']

    # Inclui o tempo de consulta de todos os pacientes que precedem o paciente em questao
    if patient_priority == 'normal':

        # Se for paciente normal, todos os urgentes passam na frente; 
        for patientID, consultationTime in state_machine['urgent_queue']:
            features.append(consultationTime)

        for patientID, consultationTime in state_machine['queue']:
            features.append(consultationTime)
            if patientID == patient_id:
                break

    elif patient_priority == 'urgent':

        # Se for paciente urgente, apenas outros urgentes que chegaram antes passam na sua frente
        for patientID, consultationTime in state_machine['urgent_queue']:
            features.append(consultationTime)
            if patientID == patient_id:
                break

    return features


def get_estimate(model, features):
    #return features + 500

    # Retorna a estimativa de tempo para ser atendido, que eh: t_final_assessment + delta_fila + delta_consulta + erro
    
    # TODO: falta incluir um epsilon do tempo de espera minimo para liberar um medico. 

    return sum([int(x) for x in features])


def update_state(state_machine, event):

	# TODO: remover essa linha no final do codigo! 
    #state_machine['time'] = event.time

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

        # TODO: obtem a previsao de tempo de consulta
        consultation_duration = predict_consultation_duration(state_machine[event.patient], urgency)

        # Coloca o paciente na fila adequada
        patient4queue = (event.patient, consultation_duration)
        if urgency == 'normal': 
            state_machine['queue'].append(patient4queue)
        elif urgency == 'urgent':
            state_machine['urgent_queue'].append(patient4queue)

    elif event.event == EVENT_CONSULTATION_STARTED:

        state_machine[event.patient]['consultation_start_time'] = event.time

        # Tira o primeiro paciente da fila, de acordo com a prioridade
        patient_priority = state_machine[event.patient]['priority']

        if patient_priority == 'normal':
            state_machine['queue'].pop(0)
        elif patient_priority == 'urgent':
            state_machine['urgent_queue'].pop(0)

    elif event.event == EVENT_CONSULTATION_FINISHED:

        state_machine[event.patient]['consultation_end_time'] = event.time

    else:

        # TODO: apagar, ficou soh para debug
        print("----------> [ERRO: evento desconhecido ")
