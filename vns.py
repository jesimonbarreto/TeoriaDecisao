import numpy as np
import os,sys,math, copy
from scipy.spatial import distance

#função que será otimizada
def f():
    pass

C = [] #conjunto de clientes (x,y)
cc = [] #consumo do cliente i
P = [] #conjunto de pontos(x,y) de possiveis locais para AP
cp = 150 #capacidade do ponto de acesso p
rp = 85 # limite do sinal do ponto de acesso
N = 0.95 # taxa de cobertura dos clientes
n_max = 100 #numero maximo de PAs disponiveis
ap = [] # vetor len == numero de PAs e binario para indicar se a PA é usada ounao
acp = [[]]#Matriz de numeto de clientes por numero de PAs indica se a PA é utilizada pelo cliente


def f2():
    #Calcula o valor da função objetivo f2
    f2_value = 0
    for id_c,c in enumerate(C):
        for id_p,p in enumerate(P):
            f2_value += distance.euclidean(c, p) * acp[id_c][id_p]
    
    #restrições
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

def rest6():
    penal = np.sum(ap) - n_max
    return penal

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
    x_per = copy.copy(x)
    n = len(x)
    #permuta a posição
    r = np.random.permutation(n)

    if k == 1:      # exchange two random positions
        x_per[r[1]] = x[r[2]]
        x_per[r[2]] = x[r[1]]
    elif k == 2:    #exchage three random positions
        x_per[r[1]] = x[r[2]]
        x_per[r[2]] = x[r[3]]
        x_per[r[3]] = x[r[1]]    
    elif k == 3:     # shift positions
        if r[1] < r[2]:
            r1, r2 = r[1], r[2]
        else:
            r1, r2 = r[2], r[1]

        x_per = [x_per[1:r1-1], x_per[r1+1:r2], x_per[r1], x_per[r2+1:]]

    return x_pert   

def neighborhoodChange(x, x_line, k, f):
    
    if f(x_line) < f(x):
        x  = x_line
        k = 1
    else:
        k  += 1
    
    return x, k
    

#k_max Número de estruturas de vizinhaças definidas
#max_int numero maximo de tentativas de novos valores
def VNS(k_max = 3, max_int = 5000):
    
    nfe = 0
    x_save = []
    y_save = []
    
    # Solução inicial
    x = initialSol()
    x_save.append(x)
    y_save.append(f(x))
    
    while (nfe<=max_int):
        
        k = 1
        while(k<k_max):
            # Gera uma solu��o na k-�sima vizinhan�a de x
            x_line = shake(x,k)
            #update x
            x, k = neighborhoodChange(x, x_line, k, f)
            #save 
            x_save.append(x)
            y_save.append(f(x))
            nfe +=1
        
    x_sol = x

    return x_sol, x_save, y_save


if __name__ == "__main__":
    pass