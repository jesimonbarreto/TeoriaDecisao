import numpy as np
import os,sys,math, copy
from scipy.spatial import distance



C = [] #conjunto de clientes (x,y)
cc = [] #consumo do cliente i
P = [] #conjunto de pontos(x,y) de possiveis locais para AP
cp = 150 #capacidade do ponto de acesso p
rp = 85 # limite do sinal do ponto de acesso
N = 0.95 # taxa de cobertura dos clientes
n_max = 100 #numero maximo de PAs disponiveis
ap = [] # vetor len == numero de PAs e binario para indicar se a PA é usada ou nao
acp = [[]]#Matriz de numero de clientes por numero de PAs indica se a PA é utilizada pelo cliente

#Code 

#função que será otimizada
def f1():
    pass

def f2():
    #Calcula o valor da função objetivo f2
    f2_value = 0
    for id_c,c in enumerate(C):
        for id_p,p in enumerate(P):
            f2_value += distance.euclidean(c, p) * acp[id_c][id_p]
    
    #valor de f2 penalizada pelos valores das restrições
    f2_value += rest3() + rest4() + rest5() + rest6() + rest7()

    return f2_penal


#atender N % dos clientes
def rest3():
    total_clients_att = 0
    for id_c,c in enumerate(C):
        for id_p,p in enumerate(P):
            total_clients_att+= acp[id_c][id_p]
    
    penal = N * len(C) - total_clients_att

    return penal

#Garante nao estourar a capacidade do ponto de acesso
def rest4():
    pontos_cap = 0 
    for id_c,c in enumerate(C):
        pontos_cap += acp[id_c] * cc
    
    penal = pontos_cap - cp
    return penal

#garante que cada cliente estará conectado a 1 AP
def rest5():
    pert = [] 
    for id_p,p in enumerate(P):
        ac = np.sum(acp[:,id_p])
        pert.append(ac-1)
    penal = np.sum(pert)
    return penal

#garante que nao exceda o numero de pontos de acesso
def rest6():
    penal = np.sum(ap) - n_max
    return penal

#Garante que a distancia nao exceda a distancia max de um ponto de acesso
def rest7():
    dist = 0
    for id_c,c in enumerate(C):
        for id_p,p in enumerate(P):
            dist += distance.euclidean(c, p) - rp
    return dist


#retorna um valor inicial de x aleatório e factível
def initialSol():
    pass

#pertubação do domínio de solução
def shake(x, k):
    pass

def neighborhoodChange(x, x_line, k, f):
    
    if f(x_line) < f(x):
        x = x_line
        k = 1
    else:
        k  += 1
    
    return x, k

def bestImprovement(x_line):
    pass

#k_max Número de estruturas de vizinhaças definidas
#max_int numero maximo de tentativas de novos valores
def BVNS(k_max = 4, max_int = 5000):
    nfe = 0
    x_save = []
    y_save = []
    
    # Solução inicial
    x = initialSol()
    x_save.append(x)
    y_save.append(f(x))
    
    while (nfe<max_int):

        k = 1
        while(k<k_max):
            # Gera uma solução na k-esima vizinhança de x
            x_line = shake(x,k) #shaking
            x_line_line = bestImprovement(x_line)
            #update x
            x, k = neighborhoodChange(x, x_line_line, k, f)
            #save 
            x_save.append(x)
            y_save.append(f(x))
            nfe +=1
        
    x_sol = x

    return x_sol, x_save, y_save


if __name__ == "__main__":
    pass
    #upar o arquivo
    #carregar os valores 
    #colocar no formato
    #chamar a função pra otimizar as duas funções