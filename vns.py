import numpy as np
import os,sys,math, copy
from scipy.spatial import distance
from random import seed,random,randint
import pandas as pd
import matplotlib.pyplot as plt

seed(1234)
np.random.seed(1234)


def plot_value(dados, y):

    P = dados['P']
    pos = dados['ap'] == 1
    C = dados['C']
    x_sol = P[pos]
    plt.ylabel('Erro')
    plt.xlabel('Interações')
    plt.plot(y)
    plt.show()
    plt.plot(C[:,0], C[:,1], 'bs')
    plt.plot(x_sol[:,0], x_sol[:,1], 'ro')
    plt.show()

#função que será otimizada
def f1(dados):
    ap = dados['ap']

    # Calcula o valor da função objetivo f1
    f1 = sum(ap)

    # valor de f1 penalizada pelos valores das restrições
    f1_penal = f1 + 10 * (rest3(dados) + rest4(dados) + rest5(dados) + rest6(dados) + rest7(dados))

    return f1_penal

def f2(dados):
    acp = dados['acp']
    rp = dados['rp']
    d = dados['d']
    P = dados['P']
    
    #Calcula o valor da função objetivo f2
    f2_value = 0
    for id_c,c in enumerate(C):
        for id_p,p in enumerate(P):
            f2_value += d[id_c, id_p] * acp[id_c][id_p]
    
    #valor de f2 penalizada pelos valores das restrições
    f2_penal = f2_value + 10 * (rest3(dados) + rest4(dados) + rest5(dados) + rest6(dados) + rest7(dados))

    return f2_penal


def f1_e_restrito(dados, f):
    ap = dados['ap']

    # Calcula o valor da função objetivo f1
    f1 = sum(ap)

    # valor de f1 penalizada pelos valores das restrições
    f1_penal = f1 + 10 * (rest3(dados) + rest4(dados) + rest5(dados) + rest6(dados) + rest7(dados) + rest_e_restrito_f2(dados, f))

    return f1_penal


def f2_e_restrito(dados):
    acp = dados['acp']
    rp = dados['rp']
    d = dados['d']
    P = dados['P']

    # Calcula o valor da função objetivo f2
    f2_value = 0
    for id_c, c in enumerate(C):
        for id_p, p in enumerate(P):
            f2_value += d[id_c, id_p] * acp[id_c][id_p]

    # valor de f2 penalizada pelos valores das restrições
    f2_penal = f2_value + 10 * (rest3(dados) + rest4(dados) + rest5(dados) + rest6(dados) + rest7(dados) + rest_e_restrito_f1(dados, f1))

    return f2_penal


# função objetivo 2 convertida para restrição
# epsilon => média dos resultados max e min obtidos para f2 no tc02
# epsilon = mean(f2(.)_max, f2(.)_min) = (1052924.9992 + 969495.2694) / 2
def rest_e_restrito_f1(dados, e=1011210.1343):
    return max(0, f1(dados) - e) ** 2

def rest_e_restrito_f2(dados, e=1011210.1343):
    return max(0, f2(dados) - e) ** 2


def rest3(dados):
    N = dados['N']
    C = dados['C']
    acp = dados['acp']

    penal = N * len(C) - acp.sum()
    penal = max(0, penal) ** 2
    return penal

# Garante nao estourar a capacidade do ponto de acesso
def rest4(dados):
    acp = dados['acp']
    cc = dados['cc']
    cp = dados['cp']

    pontos_cap = np.matmul(acp.transpose(), cc) - cp

    # soma somente os pontos que nao obedecem a restricao
    penal = 0
    for i in pontos_cap:
        if (i > 0):
            penal += i

    return penal

# garante que cada cliente estará conectado a no maximo 1 PA
def rest5(dados):
    acp = dados['acp']

    pert = acp.sum(axis=1) - 1
    penal = 0
    for i in pert:
        if (i > 0):
            penal += i
    return penal

# garante que nao exceda o numero de pontos de acesso
def rest6(dados):
    ap = dados['ap']
    n_max = dados['n_max']

    penal = np.sum(ap) - n_max
    penal = max(0, penal) ** 2

    return penal

# Garante que a distancia nao exceda a distancia max de um ponto de acesso
def rest7(dados):
    acp = dados['acp']
    rp = dados['rp']
    d = dados['d']

    dist = acp * d - rp
    penal = 0
    for i in dist:
        for j in i:
            if (j > 0):
                penal += j
    return penal


#coloca o ponto de acesso 'ponto' em um novo lugar aleatorio
def novoLocalPontoAcesso(dados):
    ap = dados['ap'].copy()
    acp = dados['acp'].copy()

    #ponto velho
    ponto = False
    while(not ponto):
        pv = np.random.choice(ap.shape[0], size=1, replace=False)
        ponto = ap[pv]
    #novo ponto
    ponto = True
    while(ponto):
        pn = np.random.choice(ap.shape[0], size=1, replace=False)
        ponto = ap[pn]
    
    #passando todos os clientes que são atendidos pelo ponto de acesso para o novo ponto de acesso
    nad = acp[:,pn]
    acp[:,pn] = acp[:,pv]
    acp[:,pv] = nad
    #trocando o estados de utilização dos pontos de acesso
    ap[pv] = 0
    ap[pn] = 1
    #atualizando 
    dados['ap'] = ap
    dados['acp'] = acp

    return dados

#Coloca para um cliente ser atendido por outro ponto de acesso ativo
def novoPontoAcessoClient(dados):
    ap = dados['ap'].copy()
    acp = dados['acp'].copy()

    #soteia um cliente
    cliente = np.random.choice(acp.shape[0], size=1, replace=False)

    #seleciona o novo ponto de acesso que o cliente ira conectar
    ponto = False
    while(not ponto):
        p = np.random.choice(ap.shape[0], size=1, replace=False)
        ponto = ap[p]
    
    pontovelho = acp[cliente,:] == 1
    acp[cliente,pontovelho[0]] = 0
    acp[cliente,p] = 1
    
    dados['ap'] = ap
    dados['acp'] = acp
    
    return dados
    
#Altera o estado do ponto de acesso 
def usoUmPontoAcesso(dados):
    ap = dados['ap'].copy()
    acp = dados['acp'].copy()
    P = dados['P'].copy()
    C = dados['C'].copy()

    p = np.random.choice(ap.shape[0], size=1, replace=False)
    #se for ponto ativo passa os clientes para outro ponto de acesso
    if ap[p]:
        ponto = False
        while(not ponto):
            ps = np.random.choice(ap.shape[0], size=1, replace=False)
            ponto = ap[ps]
        
        acp[:,ps] = acp[:,ps] + acp[:,p] #atribui os clientes para o outro ponto
        for i in range(len(acp[:,p])):
            acp[i,p] = 0
        
    
    ap[p] = int(not ap[p])
    dados['ap'] = ap
    dados['acp'] = acp

    return dados


#Redistribui aleatoriamente os clientes
def usoPontoAcesso(dados):
    ap = dados['ap'].copy()
    acp = dados['acp'].copy()
    P = dados['P'].copy()
    C = dados['C'].copy()

    pas_ligados = np.where(ap == 1)[0]
    #pas_deslig = np.where(ap==0)[0]
    #selecionar um ligado para distribuir entre outros
    io = np.random.choice(pas_ligados, size=1, replace=False)
    #para cada cliente redistribui os clientes para cada um que será ativo
    clientes = acp[:,io]
    for id_, clt in enumerate(clientes):
        #sorteia um novo ponto de acesso que sera ativado
        if clt:
            p = np.random.choice(pas_ligados, size=1, replace=False)
            while(p!=io):
                p = np.random.choice(pas_ligados, size=1, replace=False)    
            acp[id_,p] = 1


    for i in range(len(acp[:,io])):
        acp[i,io] = 0
    #troca os estados
    ap[io] = 0


    dados['ap'] = ap
    dados['acp'] = acp
    return dados

    
#pertubação do domínio de solução
def shake(dados, k):
    #Estruturas de vizinhaças
    #1 - Coloca o ponto de acesso em um novo ponto aleatorio - passa os clientes para o novo ponto
    #2 - Trocar um cliente para outro ponto de acesso aleatorio 
    #3 - Trocar o estado de uso de um ponto de acesso aleatorio
    #4 - Altera o estado de uso de todos os pontos de acesso se tiver ativo e tiver cliente passa para outro ponto de acesso ativo

    if k == 1:
        n_change = 10
        for n in range(n_change):
            dados = novoLocalPontoAcesso(copy.copy(dados))
    elif k == 2:
        n_change = 10
        for n in range(n_change):
            dados = novoPontoAcessoClient(copy.copy(dados))
    elif k == 3:
        n_change = 10
        for n in range(n_change):
            dados = usoUmPontoAcesso(copy.copy(dados))
    else:
        n_change = 10
        for n in range(n_change):
            dados = usoPontoAcesso(copy.copy(dados))
    
    return dados

#retorna um valor inicial de x aleatório e factível
def initialSol(dados):
    # Calcula as coordenadas dos possíveis PAs
    C = dados['C']
    P = dados['P']
    d = dados['d']
    ap = dados['ap']
    n_max = dados['n_max']
    acp = dados['acp']
    sizex = dados['sizex']
    sizey = dados['sizey']
    grid = dados['grid']

    i=0
    for x in range(sizex[0], sizex[1], grid[0]):
        for y in range(sizey[0], sizey[1], grid[1]):
            P[i, 0] = x
            P[i, 1] = y
            i = i + 1

    # calcula distancia entre cada cliente a cada ponto de acesso
    for id_c, c in enumerate(C):
        for id_p, p in enumerate(P):
            d[id_c, id_p] = distance.euclidean(c, p)

    # gera solucao inicial - PAs ativos
    ap = np.zeros(len(P))
    index = np.random.randint(0, ap.size, n_max)
    ap[index] = 1

    # gera solucao inicial - clientes atendidos
    for i in range(0, len(acp)):
        j = np.random.choice(index)
        acp[i, j] = 1

    #atualiza os dados
    dados['P'] = P
    dados['ap'] = ap
    dados['acp'] = acp

    return dados


def neighborhoodChange(x_, x_line_, k, f):
    
    if f(x_line_) < f(x_):
        x_ = x_line_
        k = 1
    else:
        k += 1
    
    return x_, k

def bestImprovement(dados, f):

    new_dados_1 = copy.copy(dados)
    new_dados_2 = copy.copy(dados)

    # seleciona index de ap válido
    ponto = False
    while not ponto:
        pv = np.random.choice(dados['ap'].shape[0], size=1, replace=False) # pv é o index de um ponto ativo
        ponto = dados['ap'][pv]

    if pv == 0:
        pv+=1
    elif pv>=int(dados['ap'].shape[0])-2:
        pv-=1

    for i in range(len(new_dados_1['acp'][:, pv])):
        current_client = new_dados_1['acp'][i, pv]
        new_dados_1['acp'][i, pv] = new_dados_1['acp'][i, pv - 1]
        new_dados_1['acp'][i, pv - 1] = current_client

        current_client = new_dados_2['acp'][i, pv]
        new_dados_2['acp'][i, pv] = new_dados_2['acp'][i, pv + 1]
        new_dados_2['acp'][i, pv + 1] = current_client

    cpy = new_dados_1['ap'][pv] 
    new_dados_1['ap'][pv-1] = new_dados_1['ap'][pv]
    new_dados_1['ap'][pv] = cpy

 
    cpy = new_dados_2['ap'][pv] 
    new_dados_2['ap'][pv+1] = new_dados_2['ap'][pv]
    new_dados_2['ap'][pv] = cpy

    f_1 = float(f(dados))
    f_2 = float(f(new_dados_1))
    f_3 = float(f(new_dados_2))

    if f_1 > f_2 and f_1 > f_3:
        return dados
    elif f_2 > f_1 and f_2> f_3:
        return new_dados_1
    else:
        return new_dados_2

    
#k_max Número de estruturas de vizinhaças definidas
#max_int numero maximo de tentativas de novos valores
def BVNS(dados, f, k_max, e, max_int = 5000, plot = False):
    nfe = 0
    x_save = []
    y_save = []
    
    # Solução inicial
    dados = initialSol(copy.copy(dados))

    #x_save.append(x)
    y = f(dados)
    y_save.append(y)
    
    while(nfe<max_int):
        print('Interação: {}'.format(nfe))
        print('Valor f(x): {}'.format(y))
        if plot:
            plot_value(dados)

        k = 1
        while(k<=k_max):
            # Gera uma solução na k-esima vizinhança de x
            dados_line = shake(copy.copy(dados),k) #shaking
            dados_line_line = bestImprovement(copy.copy(dados_line), f)
            #update x
            dados, k = neighborhoodChange(copy.copy(dados), copy.copy(dados_line_line), k, f)
            #save 
            #x_save.append(x)
            y = f(dados)
            y_save.append(y)
        
        nfe +=1

    dados_sol = dados

    return dados_sol, y_save


if __name__ == "__main__":
    
    path_file = 'clientes.csv'
    df = pd.read_csv(path_file)
    value = df.values
    #Inicializa variáveis
    #(0,805)
    sizex = (0,800)
    sizey = (0,800)
    #5x5
    grid = (20,20)
    n_P = int((sizex[1]*sizey[1])/ (grid[0]*grid[1]))
    C = value[:,:2] #conjunto de clientes (x,y)
    cc = value[:,2:] #consumo do cliente i
    #setar valores dos grides como (x,y) ou como x e y separados
    P =  np.zeros((n_P, 2))#conjunto de pontos(x,y) de possiveis locais para AP
    cp = 150 #capacidade do ponto de acesso p
    rp = 85 # limite do sinal do ponto de acesso
    N = 0.95 # taxa de cobertura dos clientes
    n_max = 100 #numero maximo de PAs disponiveis
    ap = np.zeros(len(P)) # vetor len == numero de PAs e binario para indicar se a PA é usada ou nao
    #acp = np.zeros((int(C.shape[0]), n_max))#Matriz de numero de clientes por numero de PAs indica se a PA é utilizada pelo cliente
    d = np.zeros((len(C), len(P)))
    acp = np.zeros((len(C), len(P)))

    x = {
        'd':d,
        'C': C,
        'cc': cc,
        'P': P,
        'cp': cp,
        'rp': rp,
        'N': N,
        'n_max': n_max,
        'ap': ap,
        'acp': acp,
        'grid':grid,
        'sizex':sizex,
        'sizey':sizey
    }

    #Otimizando
    sol, y = BVNS(x, f2, k_max = 4, e=66666 , max_int = 20, plot = False)
    #plot solution
    plot_value(sol, y)
    #print solution
    print('Resultado:')
    print('Numero de Pontos de acesso: {}'.format(np.sum(sol['ap'])))