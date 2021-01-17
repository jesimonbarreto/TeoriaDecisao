import numpy as np
import os,sys,math,copy,random
from scipy.spatial import distance
seed(1234)
np.random.seed(1234)


def f1():
    # Calcula o valor da função objetivo f1
    f1 = sum(ap)

    # valor de f1 penalizada pelos valores das restrições
    f1_penal = f1 + 10 * (rest3() + rest4() + rest5() + rest6() + rest7())

    return f1_penal


def rest3():
    penal = N * len(C) - acp.sum()
    penal = max(0, penal) ** 2
    return penal


# Garante nao estourar a capacidade do ponto de acesso
def rest4():
    pontos_cap = np.matmul(acp.transpose(), cc) - cp

    # soma somente os pontos que nao obedecem a restricao
    penal = 0
    for i in pontos_cap:
        if (i > 0):
            penal += i

    return penal


# garante que cada cliente estará conectado a no maximo 1 PA
def rest5():
    pert = acp.sum(axis=1) - 1

    penal = 0
    for i in pert:
        if (i > 0):
            penal += i
    return penal


# garante que nao exceda o numero de pontos de acesso
def rest6():
    penal = np.sum(ap) - n_max
    penal = max(0, penal) ** 2

    return penal


# Garante que a distancia nao exceda a distancia max de um ponto de acesso
def rest7():
    dist = acp * d - rp
    penal = 0
    for i in dist:
        for j in i:
            if (j > 0):
                penal += j
    return penal

if __name__ == "__main__":
    clientes = np.loadtxt(open("clientes.csv", "r"), delimiter=",")

    C = clientes[:, 0:2]  # posicao clientes
    cc = clientes[:, 2]  # consumo clientes
    N = 0.95  # taxa de cobertura
    n_max = 100  # numero maximo de PAs
    cp = 150  # capacidade do PA
    rp = 85  # alcance do sinal
    P = np.zeros((25921, 2))  # PAs
    d = np.zeros((len(C), len(P)))   # distancias
    acp = np.zeros((len(C), len(P)))   # clientes atendidos

    # Calcula as coordenadas dos possíveis PAs
    i = 0
    for x in range(0, 805, 5):
        for y in range(0, 805, 5):
            P[i, 0] = x
            P[i, 1] = y
            i = i + 1

    # calcula distancia entre cada cliente a cada ponto de acesso
    for id_c, c in enumerate(C):
        for id_p, p in enumerate(P):
            d[id_c, id_p] = distance.euclidean(c, p)

    # gera solucao inicial - PAs ativos
    ap = np.zeros(len(P))
    index = np.random.randint(0, ap.size, nMax)
    ap[index] = 1

    # gera solucao inicial - clientes atendidos
    for i in range(0, len(acp)):
        j = random.choice(index)
        acp[i, j] = 1