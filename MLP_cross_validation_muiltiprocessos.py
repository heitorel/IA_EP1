import pandas as pd
import numpy as np
from math import *
import matplotlib as plt
from modelo_MLP import modelo, embaralhar_dado, tranformar_categoria_em_numeros, split_dados, funcao_para_multithead, k_fold_cross_validation_multithead, retorna_erro, gera_matriz_confusao
from multiprocessing import Process as worker
from multiprocessing import Queue
from time import time
import sys


def rodar_cross_validation_multi_thread():
    ################################################################################
    ################################################################################
    #Importar dados:################################################################
    #path = 'optdigits.dat'
    path = sys.argv[1]
    nome_arquivo_saida = sys.argv[2]
    df = pd.read_csv(path)

    #embaralhar os dados
    df = embaralhar_dado(df)
    #obtenho o valor de todos os y e transformo uma coluna de vetores
    y = tranformar_categoria_em_numeros(df[str(df.shape[1]-1)])

    #obtenho o valor de todos os X
    X = df.drop([str(df.shape[1]-1)],axis=1).values
    #Separo 80% do X e y para treinar e validar e 20% para conjunto de teste
    X, x_test, y, y_test = split_dados(X,y,tamanho_percentual=.8)

    ################################################################################
    ################################################################################
    #Definir os hiperparâmetros:####################################################
    k = 10
    nos_cam_oculta = 83
    eta = 0.02
    alpha=0.017889
    ################################################################################
    ################################################################################
    ################################################################################

    intervalo = int(y.shape[0]/k)
    intervalos = [(x + 1)*intervalo for x in range(k)]
    intervalos[-1] = y.shape[0]    
    limite_inferior = 0
    erro_folds = []
    t = []
    i = 0
    queue = Queue()
    for limite_superior in intervalos:
        x_test = X[limite_inferior:limite_superior]            
        y_test = y[limite_inferior:limite_superior] 
        x_train = np.concatenate((X[0:limite_inferior],X[limite_superior:]))
        y_train = np.concatenate((y[0:limite_inferior],y[limite_superior:]))
        limite_inferior = limite_superior


        
        aux = worker(target=k_fold_cross_validation_multithead, args=(x_train,y_train,x_test,y_test,nos_cam_oculta,eta,alpha,queue))
        t.append(aux)
        t[i].start()
        i+=1      

    # Barreira the threads: nenhum thread pode percorrer sozinha após este ponto...eu mesclo novamente as threads neste ponto
    for i in range(len(t)):
        t[i].join()    
        
    erro_folds = []
    for i in range(len(t)):
        erro_folds.append(queue.get())


    #Registra o erro de cada fold
    df_saida = pd.DataFrame(erro_folds,columns=["Qtd_nos_camada_oculta","Eta","Alpha","Acuracidade_treinamento","Erro_quadratico_medio_fold"])[["Qtd_nos_camada_oculta","Eta","Alpha","Erro_quadratico_medio_fold"]]

    #Adiono no DataFrame a media com index = Media
    media = df_saida["Erro_quadratico_medio_fold"].mean()
    linha_media = pd.DataFrame([[nos_cam_oculta,eta,alpha,None, media]],index=['Media'])
    df_saida.append(linha_media)

    # Salvo arquivo
    df_saida.to_csv(nome_arquivo_saida+"_Erro_de_cada_fold.csv")

    #Treino novamente o modelo sem os dados de teste:
    obj = modelo(X,y,1,nos_cam_oculta)
    obj.mlp(eta=eta,epocas=120,alpha=alpha, registrar_erro_epocas=False)

    #registrar a media artimetica dos erros    
    pd.DataFrame(
        [[nos_cam_oculta,eta,alpha,obj.erro_quadratico_medio[-1], retorna_erro(obj.prever(x_test),y_test)]],
        columns=["Qtd_nos_camada_oculta","Eta","Alpha","Acuracidade_treinamento","Erro_quadratico_medio_conjunto_teste"],
        ).to_csv(nome_arquivo_saida+'_Erro_Verdadeiro.csv')

    gera_matriz_confusao(obj.prever(x_test),y_test).to_csv(nome_arquivo_saida+"_Matriz_Confusao.csv")

    print("Erro médio do K-Fold Cross Validation: ", df_saida["Erro_quadratico_medio_fold"].mean())

if (__name__ == "__main__"):
    rodar_cross_validation_multi_thread()

################################################################################
################################################################################
#Importar dados:################################################################
#path = 'optdigits.dat'
path = sys.argv[1]
nome_arquivo_saida = sys.argv[2]
df = pd.read_csv(path)

#embaralhar os dados
df = embaralhar_dado(df)
#obtenho o valor de todos os y e transformo uma coluna de vetores
y = tranformar_categoria_em_numeros(df[str(df.shape[1]-1)])

#obtenho o valor de todos os X
X = df.drop([str(df.shape[1]-1)],axis=1).values

################################################################################
################################################################################
#Definir os hiperparâmetros:####################################################
k = 10
nos_cam_oculta = 83
eta = 0.02
alpha=0.011556
epocas = 120
################################################################################
################################################################################
################################################################################

intervalo = int(y.shape[0]/k)
intervalos = [(x + 1)*intervalo for x in range(k)]
intervalos[-1] = y.shape[0]    
limite_inferior = 0
erro_folds = []
t = []
i = 0
queue = Queue()
for limite_superior in intervalos:
    x_test = X[limite_inferior:limite_superior]            
    y_test = y[limite_inferior:limite_superior] 
    x_train = np.concatenate((X[0:limite_inferior],X[limite_superior:]))
    y_train = np.concatenate((y[0:limite_inferior],y[limite_superior:]))
    limite_inferior = limite_superior
    
    aux = worker(target=k_fold_cross_validation_multithead, args=(x_train,y_train,x_test,y_test,nos_cam_oculta,eta,alpha,queue))
    t.append(aux)
    t[i].start()
    i+=1      

# matriz de confusão do CV
obj = modelo(x_train,y_train,1,nos_cam_oculta)
obj.mlp(eta=eta,epocas=epocas,alpha=alpha)
y_previsto = obj.prever(x_test)
matriz_confusao = pd.DataFrame(gera_matriz_confusao(y_previsto,y_test), index_label=True)

def plt_matriz_confusao(matriz_confusao):
    plt.matshow(matriz_confusao, cmap=plt.cm.gray_r)
    plt.title('Matriz de Confusão')
    plt.colorbar()
    tick_marks = np.arange(len(matriz_confusao.columns))
    plt.xticks(tick_marks, matriz_confusao.columns, rotation=45)
    plt.yticks(tick_marks, matriz_confusao.index)
    plt.ylabel(matriz_confusao.index.name)
    plt.xlabel(matriz_confusao.columns.name)
    plt.savefig(f'Matriz_Confusao_{0}.png'(nome_arquivo_saida), format='png')
    
plt_matriz_confusao(matriz_confusao)
    
# Barreira the threads: nenhum thread pode percorrer sozinha após este ponto...eu mesclo novamente as threads neste ponto
for i in range(len(t)):
    t[i].join()    
    
erro_folds = []
for i in range(len(t)):
    erro_folds.append(queue.get())
df_saida = pd.DataFrame(erro_folds,columns=["Index","Qtd_nos_camada_oculta","Eta","Alpha","Acuracidade_treinamento","Erro_quadratico_medio"])[["Index","Qtd_nos_camada_oculta","Eta","Alpha","Erro_quadratico_medio"]]
df_saida.to_csv(nome_arquivo_saida, index_label=True)
#registrar a media artimetica dos erros
print("Erro médio do K-Fold Cross Validation: ", df_saida["Erro_quadratico_medio"].mean())

