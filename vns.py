import numpy as np
import os,sys,math, copy
from scipy.spatial import distance
from random import seed,random,randint
seed(1234)
np.random.seed(1234)

limits_ap_x = (0,100) #grid x 0 - 100 valores possiveis para um ponto de acesso
limits_ap_y = (0,100) #grid y 0 - 100
C = [] #conjunto de clientes (x,y)
cc = [] #consumo do cliente i
P = [] #conjunto de pontos(x,y) de possiveis locais para AP
cp = 150 #capacidade do ponto de acesso p
rp = 85 # limite do sinal do ponto de acesso
N = 0.95 # taxa de cobertura dos clientes
n_max = 100 #numero maximo de PAs disponiveis
ap = [] # vetor len == numero de PAs e binario para indicar se a PA é usada ou nao
acp = [[]]#Matriz de numero de clientes por numero de PAs indica se a PA é utilizada pelo cliente

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

def novoLocalPontoAcesso(ponto):
    novo_x = randint(limits_ap_x[0], limits_ap_x[1])
    novo_y = randint(limits_ap_y[0], limits_ap_y[1])
    AP[ponto] = np.array([novo_x, novo_y])

def novoPontoAcessoClient():
    

def usoPontoAcesso(n):
    ap[n] = int(not ap[n])

def usoPontoAcesso():
    indices_one = ap == 1
    indices_zero = ap == 0
    ap[indices_one] = 0
    ap[indices_zero] = 1


#retorna um valor inicial de x aleatório e factível
def initialSol():
    pass

#pertubação do domínio de solução
def shake(x, k):
    #Estruturas de vizinhaças
    #1 - Coloca o ponto de acesso em um novo ponto aleatorio 
    #2 - Trocar um cliente para outro ponto de acesso aleatorio 
    #3 - Trocar o estado de uso de um ponto de acesso aleatorio
    #4 - Altera o estado de uso de todos os pontos de acesso
    if k == 1:
        #Sorteia um dos pontos de acesso que está sendo usado
        ponto = False
        while(not ponto):
            p = np.random.choice(ap.shape[0], size=1, replace=False)
            ponto = ap[p]
        novoLocalPontoAcesso(p)
    elif k == 2:
        
        novoPontoAcessoClient(pontovelho, cliente, pontonovo)
    elif k == 3:
        p = np.random.choice(ap.shape[0], size=1, replace=False)
        usoPontoAcesso(p)
    elif k == 4:
        usoPontoAcesso()
    
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