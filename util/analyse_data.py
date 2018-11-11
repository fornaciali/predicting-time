import os
import argparse

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


def get_args():
    parser = argparse.ArgumentParser(description='Generates and persists the 5-folds.')
    parser.add_argument('summary_data_file', type=str)
    args = parser.parse_args()
    return args


def save_info(file_path_and_name, info):
    f = open(file_path_and_name, 'w')
    f.write(info)
    f.close()


# PARAMETROS:
#   - arquivo de dados (summary)


# arrival_time    assessment_end_time assessment_start_time   consultation_end_time   consultation_start_time day duration    pain    patient priority    temperature


# Carregar os dados (todos)
print("Carregando...")
args = get_args()
data = pd.read_csv(args.summary_data_file)
print("Done!")



# Ampliar os dados com features extras

# Salvar arquivo extendido e usar o bonito na geracao do modelo ? 


# Exibir estatisticas consolidadas 
#print(data.head())

#save_info("stats.txt", data.describe().to_string())

print(data.describe())

#print(set(data['pain'].values))

#print(data[data['day'] == 1])

#print(data.temperature)




# TODO : tracar a media em vermelho como reta!!!


# grafico de TEMPERATURA x DURACAO DA CONSULTA 

time_queue = data.consultation_start_time - data.assessment_end_time

print(np.corrcoef(data.temperature, data.duration))

plt.scatter(data.temperature, time_queue) #data.duration)
plt.title("TEMPERATURA x DURACAO DA CONSULTA")
plt.show(block=True)


# grafico de TEMPO PARA 1º ATENDIMENTO x DURACAO DA CONSULTA 

print(np.corrcoef(data.assessment_start_time - data.arrival_time, data.duration))

plt.scatter(data.assessment_start_time - data.arrival_time, time_queue) #data.duration)
plt.title("TEMPO PARA 1º ATENDIMENTO x DURACAO DA CONSULTA")
plt.show(block=True)


#print(np.corrcoef(data.pain, data.duration))

plt.scatter(data.pain, time_queue) #data.duration)
plt.title("DOR x DURACAO DA CONSULTA")
plt.show(block=True)


#print(np.corrcoef(data.priority, data.duration))

plt.scatter(data.priority, time_queue) #data.duration)
plt.title("PRIORIDADE x DURACAO DA CONSULTA")
plt.show(block=True)


print(np.corrcoef(data.arrival_time, data.duration))

plt.scatter(data.arrival_time, time_queue) #data.duration)
plt.title("HORARIO CHEGADA x DURACAO DA CONSULTA")
plt.show(block=True)


# distribuicao percentual de DOR
labels = set(data['pain'].values)
sizes = [len(data[data['pain'] == x]) for x in labels]
explode = (0, 0, 0)  

#fig1, ax1 = plt.subplots()
#ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
#ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

#plt.show()



# histograma de HORÁRIO DE CHEGADA x DURAÇÃO DA CONSULTA
x = list(data['arrival_time'])

n, bins, patches = plt.hist(x, 40, density=False, color='g')
plt.xlabel('Arrival Time')
plt.ylabel('Quantity')
plt.title('HISTOGRAMA DE DURAÇÃO DA CONSULTA POR HORÁRIO DE CHEGADA')
plt.axis([-500, 15000, 0, 200])
plt.grid(True)
plt.show()

# descobrir a feature mais importante (maior correlacao) com duracao de consulta

# correlação entre horário de pico X tempo de consulta (geral?);
# correlação de tempo de atendimento com urgência;
