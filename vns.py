import numpy as np
import os,sys,math, copy

#Code 

#função que será otimizada
def f():
    pass

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