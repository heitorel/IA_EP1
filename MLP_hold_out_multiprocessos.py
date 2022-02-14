import pandas as pd
import numpy as np
from math import *
import matplotlib as plt
from modelo_MLP import modelo, embaralhar_dado, tranformar_categoria_em_numeros, split_dados, funcao_para_multithead, k_fold_cross_validation_multithead, retorna_erro
from multiprocessing import Process as worker
from multiprocessing import Queue
from time import time
import sys

path = sys.argv[1]
nome_arquivo_saida = sys.argv[2]

# path = 'optdigits.dat'
df = pd.read_csv(path)

#embaralhar os dados
df = embaralhar_dado(df)
#obtenho o valor de todos os y e transformo uma coluna de vetores
y = tranformar_categoria_em_numeros(df[str(df.shape[1]-1)])

#obtenho o valor de todos os X
X = df.drop([str(df.shape[1]-1)],axis=1).values

#Separado o X e y da base de treinamento, validacao e teste
x_train, x_val, y_train, y_val = split_dados(X,y,tamanho_percentual=.5)
x_val, x_test, y_val, y_test = split_dados(x_val,y_val,tamanho_percentual=.5)

def rodar_hold_out_multi_thread(min_nos,max_nos,num_nos,min_eta,max_eta,num_eta,min_alpha,max_alpha,num_alpha,nome_arquivo_resultados="resultados_holdout",threads_max=1):
    experimentos = []
    #Listo todos os experimentos que preciso fazer
    for no in np.linspace(min_nos,max_nos,num=num_nos):
        for eta in np.linspace(min_eta,max_eta,num=num_eta):
            for alpha in np.linspace(min_alpha,max_alpha,num=num_alpha):
                experimentos.append([int(no),eta,alpha])
  


    start = time()
    #Loop com o intuito de nao estourar a quantidade de threads
    queue = Queue()
    
    resultados_bloco = []

    #realizo o loop em todos os experimentos
    realizados = 0
    while realizados < len(experimentos):
        #Se quantidade que falta e menor que a quantidade de threads maximo    
        if realizados+threads_max > len(experimentos):
            bloco = len(experimentos) - realizados
        else:
            bloco = threads_max         
        
        t = []
        resultados = []        
        i=0        
        
        #gero ate threads_max execucoes em paralelo via multiprocessos
        for [nos_cam_oculta,eta,alpha] in experimentos[realizados:realizados+bloco]:
            aux = worker(target=funcao_para_multithead, args=(x_train,y_train, x_test, y_test, int(nos_cam_oculta),eta,alpha,resultados,queue))
            t.append(aux)
            t[i].start()
            i+=1

        # Barreira the threads: nenhum thread pode percorrer sozinha após este ponto...eu mesclo novamente as threads neste ponto
        for i in range(len(t)):
            t[i].join()  
        
        #registro via queue todos os resultados de cada thread        
        for i in range(len(t)):
            resultados_bloco.append(queue.get())
        
        #salvo todos os resultados em um csv
        pd.DataFrame(resultados_bloco, columns=["Qtd_nos_camada_oculta","Eta","Alpha","Acuracidade_treinamento","Erro_quadratico_medio"]).sort_values(["Erro_quadratico_medio"], ascending=False).to_csv(nome_arquivo_resultados+'.csv')
        realizados += bloco
                
    print('tempo:',time()-start)
    
    
    #salvo todos os resultados em um csv
    pd.DataFrame(resultados_bloco, columns=["Qtd_nos_camada_oculta","Eta","Alpha","Acuracidade_treinamento","Erro_quadratico_medio"]).sort_values(["Erro_quadratico_medio"], ascending=False).to_csv(nome_arquivo_resultados+'.csv')
    
    #retorno um dataframe com resultados
    return pd.DataFrame(resultados_bloco, columns=["Qtd_nos_camada_oculta","Eta","Alpha","Acuracidade_treinamento","Erro_quadratico_medio"]).sort_values(["Erro_quadratico_medio"], ascending=False)

if (__name__ == '__main__'):     
    #Loop com o intuito de nao estourar a quantidade de threads     
    df_resultados_1 = rodar_hold_out_multi_thread(
        min_nos = 20,
        max_nos = 100,
        num_nos = 20,
        min_eta = 0.001,
        max_eta = 0.02,
        num_eta = 10,
        min_alpha = 0.001,
        max_alpha = 0.02,
        num_alpha = 10,
        # nome_arquivo_resultados = "Resultados_hold_out_1_VSCODE",
        nome_arquivo_resultados = nome_arquivo_saida,
        threads_max = 20)
