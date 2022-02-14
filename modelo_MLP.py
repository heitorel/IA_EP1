import pandas as pd
import numpy as np
from math import *
import matplotlib as plt
import sys

# função ativação
def funcao_logistica(x):
    return (1.0/(1.0+exp(-x)))

def funcao_normalizar(x):
    media = np.mean(x)
    desvio = np.std(x)
    return (x - media)/desvio

def embaralhar_dado(df):
    aleatorio = np.random.rand(len(df))
    df['aleatorio'] = aleatorio
    df_embaralhado = df.sort_values(['aleatorio'])
    df_embaralhado = df_embaralhado.drop(['aleatorio'], axis=1)
    return df_embaralhado

def split_dados(X, y, tamanho_percentual=.5): #com 4 casas decimais dá a melhor divisão
    # embaralho os indices do data frame
    # df_emb = embaralhar_dado(df)
    
    # defino o tamanho do conjunto de treinamento
    tam_teste = int(tamanho_percentual * X.shape[0])

    # populo os novos conjuntos com as linhas embaralhadas
    x_train = X[:tam_teste]    
    x_test = X[tam_teste:]
    y_train = y[:tam_teste]
    y_test = y[tam_teste:]

    return x_train, x_test, y_train, y_test


# Transformar variável categórica em variável numérica
# parâmetro df precisa ser uma coluna Series e não DataFrame
def tranformar_categoria_em_numeros(df:pd.DataFrame):    
    # crio colunas dummies
    y_hotencode = pd.get_dummies(df)

    # gero uma lista de listas com os valores da classe
    vetorizacao = []
    for x,y in y_hotencode.iterrows():
        vetorizacao.append((list(y)))
    
    matriz = np.array(vetorizacao)
    # df = pd.DataFrame(vetorizacao,columns=['class'],index=df.index)
    return matriz

def acuracidade(y_real, y_previsto):
    acertos = 0
    for x,y in zip(y_previsto,y_real):
        acertos += list(x)==list(y)
    print("Acuracidade: ", acertos/y_previsto.shape[0])

def retorna_erro(y_real, y_previsto):
    acertos = 0
    for x,y in zip(y_previsto,y_real):
        acertos += list(x)==list(y)
    return (1 - acertos/y_previsto.shape[0])

def gera_matriz_confusao(y_previsto,y_real):
    matriz_confusao = np.array([[0 for x in range(y_previsto.shape[1])] for y in range(y_real.shape[1])])
    for prev,real in zip(y_previsto,y_real):
        linha = 0
        coluna = 0
        for atr,(p,r) in enumerate(zip(prev,real)):
            if (p == 1):
                linha = atr
            if (r == 1):
                coluna = atr
        matriz_confusao[linha][coluna]+=1
        columns = [["Dados Previstos" for x in range(10)],[x for x in range(10)]]
        index = [["Dados Reais" for x in range(10)],[x for x in range(10)]]
        matriz_df = pd.DataFrame(matriz_confusao,columns=columns,index=index)
    return matriz_df


# Classe que instancia uma variável para poder rodar MLP como método de ML
class modelo:
    # construtor
    def __init__(self,x,y,qtd_camadas_ocultas,nos_cam_oculta):
        # aproveito e ja adiciono o bias no X
        self.x = np.c_[x, np.ones(x.shape[0])]
        self.x_original = x
        self.y = y
        self.qtd_camadas_ocultas=qtd_camadas_ocultas
        self.nos_cam_oculta=nos_cam_oculta
        self.output_epoca = np.zeros(list(y.shape))

    
    # chamo o método que executa MLP
    def mlp(self,eta, epocas, pesos_de_cada_camada = None, normalizar=True, usar_termo_momento=True, alpha=0.01, imprime_erro_quadratico_da_epoca=False, arquivo_saida=None, x_val = None, y_val = None, registrar_erro_epocas=True):
        # lista com todos as matrizes de peso!!
        # Caso o usuário não entre com pesos eu os inicializo aleatoriamente
        # Portanto, oportunidade para realizar transferencia de aprendizado ou partir de uma época de treinamento
        if pesos_de_cada_camada == None:
            # Crio uma lista já com valores zero para facilitar as próximas contas
            self.pesos_de_cada_camada = [0 for x in range(2 + self.qtd_camadas_ocultas - 1)]            
            
            # inicializar pesos entre camada entrada e 1ª camada oculta
            pesos_entrada_oculta = np.random.rand(self.x[0].size,self.nos_cam_oculta)        
            self.pesos_de_cada_camada[0] = funcao_normalizar(pesos_entrada_oculta) # Normalizo os pesos

            # Se tivermos mais de uma camada oculta preciso considerar uma nova matriz
            for camada in range(self.qtd_camadas_ocultas-1):            
                    #             pesos_oculta_oculta = np.random.rand(qtd_camadas_ocultas - 1, nos_cam_oculta, nos_cam_oculta)
                    #             pesos_oculta_oculta = funcao_normalizar(pesos_oculta_oculta) # Normalizo os pesos
                    pesos_oculta_oculta = np.random.rand(self.nos_cam_oculta, self.nos_cam_oculta)
                    self.pesos_de_cada_camada[camada + 1] = funcao_normalizar(pesos_oculta_oculta) # Normalizo os pesos

            # inicializar pesos entre última camada oculta e camada saida
            pesos_oculta_saida = np.random.rand(self.nos_cam_oculta, self.y[0].size)
            self.pesos_de_cada_camada[self.qtd_camadas_ocultas] = funcao_normalizar(pesos_oculta_saida) # Normalizo os pesos            
        else:
            # Entra aqui caso o usuário queira inicializar com alguns pesos
            self.pesos_de_cada_camada = pesos_de_cada_camada

        #Lista usada para guardar os delta pesos entre iteracoes e usar o termo momento
        self.delta_peso = [0 for x in range(2 + self.qtd_camadas_ocultas - 1)]
            
        # Rodar uma quantidade de epocas
        self.erro_quadratico_medio = [0 for x in range(epocas)]
        erros_epocas = []
        for epoca in range(epocas):
            # Passar por todos os exemplos da base
            for id_exemplo, (x_exemplo , y_exemplo) in enumerate(zip(self.x,self.y)):
                self.__executa_mlp(x_exemplo,y_exemplo,self.qtd_camadas_ocultas,self.nos_cam_oculta, eta, normalizar, usar_termo_momento, alpha, id_exemplo=id_exemplo)
            
            self.erro_quadratico_medio[epoca] = sum([list(pr)==list(y_t) for pr,y_t in zip(self.output_epoca,self.y)])/self.y.shape[0]
            if (imprime_erro_quadratico_da_epoca):
                self.erro_quadratico_medio[epoca]
            
            #registro no arquivo de saida o erro considerando em comparacao com o conjunto de validacao
            if (registrar_erro_epocas):
                previsao_epoca = self.prever(x_val)
                previsao_epoca_treinamento = self.prever(self.x_original)
                erros_epocas.append([epoca,retorna_erro(previsao_epoca_treinamento,self.y),retorna_erro(previsao_epoca,y_val)])
        if (registrar_erro_epocas):
            pd.DataFrame(erros_epocas,columns=["Epoca","Erro Quadratico Treinamento por Epoca","Erro Quadratico Validacao por Epoca"]).to_csv(arquivo_saida+'_Erros_Epoca.csv')

    
    # Método que executa MLP (veja que os pesos são inicializados no construtor)
    def __executa_mlp(self,x_train_orginial,y_train,qtd_camadas_ocultas:int, nos_cam_oculta:int, eta, normalizar, usar_termo_momento, alpha, id_exemplo=None):
        self.normalizar = normalizar
        if (self.normalizar):
            x_train = funcao_normalizar(x_train_orginial)
        else:
            x_train = x_train_orginial

    
        # para cada época
        # inicializo listas
        output_de_cada_camada = [0 for x in range(2 + qtd_camadas_ocultas - 1)]
        termo_erro_neur_de_cada_camada = [0 for x in range(2 + qtd_camadas_ocultas - 1)]

        ##################
        # Feedfoward######
        ##################

        # soma ponderada entre entradas e seus pesos
        soma_ponderada_entrada = np.matmul(x_train , self.pesos_de_cada_camada[0])

        # Aplicar funcao ativacao
        output_de_cada_camada[0] = np.array(list(map(funcao_logistica , soma_ponderada_entrada)))
                
        if (qtd_camadas_ocultas > 1):   
            # loop com a quantidade de camadas ocultas MENOS UM...está menos 2 no "for" para ajustar o loop
            for camada_oculta in range(qtd_camadas_ocultas - 1):
                # Se tivermos mais de uma camada oculta preciso considerar uma nova matriz                    
                soma_ponderada_oculta_oculta = np.matmul(output_de_cada_camada[camada_oculta],self.pesos_de_cada_camada[camada_oculta + 1])
                output_de_cada_camada[camada_oculta + 1] = (np.array(list(map(funcao_logistica , soma_ponderada_oculta_oculta))))
                    
        # soma ponderada entre a última camada oculta e seus pesos
        soma_ponderada_ultima_camada_oculta = np.matmul(output_de_cada_camada[qtd_camadas_ocultas - 1], self.pesos_de_cada_camada[qtd_camadas_ocultas])

        # Aplicar funcao ativacao
        output_de_cada_camada[qtd_camadas_ocultas] = np.array(list(map(funcao_logistica , soma_ponderada_ultima_camada_oculta)))

        # Verifica qual das foi a maior
        output_aux = output_de_cada_camada[qtd_camadas_ocultas] == output_de_cada_camada[qtd_camadas_ocultas].max()

        # output com número 1 para classe escolhida         
        output = np.array([1 if x else 0 for x in output_aux]) 
        
        # registro a saida desta iteracao para esta epoca para no final calcular o erro quadratico medio
        self.output_epoca[id_exemplo] = output.copy()
        
        
        ##################
        # Backpropagation#
        ##################

        # Passo 2
        # Termos de erro neuronios de saida
        termo_erro_neur_de_cada_camada[qtd_camadas_ocultas] = output_de_cada_camada[qtd_camadas_ocultas] * (1-output_de_cada_camada[qtd_camadas_ocultas]) * (y_train - output_de_cada_camada[qtd_camadas_ocultas])
        
        # Termos de erro camada oculta
        for camada_oculta in range(qtd_camadas_ocultas-1,-1,-1):
            aux = np.matmul(self.pesos_de_cada_camada[camada_oculta+1],termo_erro_neur_de_cada_camada[camada_oculta+1])
            termo_erro_neur_de_cada_camada[camada_oculta] = aux * output_de_cada_camada[camada_oculta]*(1-output_de_cada_camada[camada_oculta])
        
        
        # Passo 3
        # AJUSTAR PESOS DA REDE

        # Devo usar o termo momento?
        if (usar_termo_momento):
            self.delta_peso[0] = alpha * self.delta_peso[0] + np.concatenate([x*termo_erro_neur_de_cada_camada[0]*eta for x in x_train]).reshape(self.pesos_de_cada_camada[0].shape)
        else:
            self.delta_peso[0]=np.concatenate([x*termo_erro_neur_de_cada_camada[0]*eta for x in x_train]).reshape(self.pesos_de_cada_camada[0].shape)
        
        # Atualizar pesos entre camada entrada e primeira camada oculta
        self.pesos_de_cada_camada[0]+=self.delta_peso[0]

        # Atualizar pesos entre camadas ocultas e também entre última cada oculta e camada de saída
        for camada_oculta in range(qtd_camadas_ocultas):
            # Devo usar o termo momento?
            if (usar_termo_momento):
                self.delta_peso[camada_oculta+1] = alpha * self.delta_peso[camada_oculta+1] + np.concatenate([x*termo_erro_neur_de_cada_camada[camada_oculta + 1]*eta for x in output_de_cada_camada[camada_oculta]]).reshape(self.pesos_de_cada_camada[camada_oculta+1].shape)
            else:
                self.delta_peso[camada_oculta+1] = np.concatenate([x*termo_erro_neur_de_cada_camada[camada_oculta + 1]*eta for x in output_de_cada_camada[camada_oculta]]).reshape(self.pesos_de_cada_camada[camada_oculta+1].shape)
            self.pesos_de_cada_camada[camada_oculta+1] += self.delta_peso[camada_oculta+1]
        

    def prever(self, X):
        # aproveito e ja adiciono o bias no X
        x_para_prever = np.c_[X, np.ones(X.shape[0])]
        
        # Passar por todos os exemplos da base
        output = []
        for x in x_para_prever:
            if (self.normalizar):
                x_train = funcao_normalizar(x)
            else:
                x_train = x

        
            # para cada época
            output_de_cada_camada = [0 for x in range(2 + self.qtd_camadas_ocultas - 1)]

            ##################
            # Feedfoward######
            ##################

            # soma ponderada entre entradas e seus pesos
            soma_ponderada_entrada = np.matmul(x_train , self.pesos_de_cada_camada[0])

            # Aplicar funcao ativacao
            output_de_cada_camada[0] = np.array(list(map(funcao_logistica , soma_ponderada_entrada)))
                    
            if (self.qtd_camadas_ocultas > 1):   
                # loop com a quantidade de camadas ocultas MENOS UM...está menos 2 no "for" para ajustar o loop
                for camada_oculta in range(self.qtd_camadas_ocultas - 1):
                    # Se tivermos mais de uma camada oculta preciso considerar uma nova matriz                    
                    soma_ponderada_oculta_oculta = np.matmul(output_de_cada_camada[camada_oculta],self.pesos_de_cada_camada[camada_oculta + 1])
                    output_de_cada_camada[camada_oculta + 1] = (np.array(list(map(funcao_logistica , soma_ponderada_oculta_oculta))))
                        
            # soma ponderada entre a última camada oculta e seus pesos
            soma_ponderada_ultima_camada_oculta = np.matmul(output_de_cada_camada[self.qtd_camadas_ocultas - 1], self.pesos_de_cada_camada[self.qtd_camadas_ocultas])

            # Aplicar funcao ativacao
            output_de_cada_camada[self.qtd_camadas_ocultas] = np.array(list(map(funcao_logistica , soma_ponderada_ultima_camada_oculta)))

            # Verifica qual das foi a maior
            output_aux = output_de_cada_camada[self.qtd_camadas_ocultas] == output_de_cada_camada[self.qtd_camadas_ocultas].max()

            # output com número 1 para classe escolhida         
            output.append(np.array([1 if x else 0 for x in output_aux]))
            # output.append(output_de_cada_camada[self.qtd_camadas_ocultas])

        return np.array(output)


def funcao_para_multithead(x_train,y_train,x_val,y_val,nos_cam_oculta,eta,alpha,resultados, pqueue):
    qtd_camadas_ocultas = 1
    obj = modelo(x_train,y_train,qtd_camadas_ocultas,nos_cam_oculta)
    epocas = 120
    obj.mlp(eta=eta,epocas=epocas,alpha=alpha)
    pqueue.put([nos_cam_oculta,eta,alpha,obj.erro_quadratico_medio[-1], retorna_erro(obj.prever(x_val),y_val)])
    print("Terminou: Qtd nos: " + str(nos_cam_oculta) + " - Eta: " + str(eta) + " - Alpha: " + str(alpha))


def k_fold_cross_validation_multithead(x_train,y_train,x_test,y_test,nos_cam_oculta,eta,alpha, pqueue):
    qtd_camadas_ocultas = 1
    obj = modelo(x_train,y_train,qtd_camadas_ocultas,nos_cam_oculta)
    epocas = 120
    obj.mlp(eta=eta,epocas=epocas,alpha=alpha,registrar_erro_epocas=False)
    
    pqueue.put([nos_cam_oculta,eta,alpha,obj.erro_quadratico_medio[-1], retorna_erro(obj.prever(x_test),y_test)])
    print("Terminou: Qtd nos: " + str(nos_cam_oculta) + " - Eta: " + str(eta) + " - Alpha: " + str(alpha))

def hold_out(nos_cam_oculta, eta, alpha):
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

    qtd_camadas_ocultas = 1
    obj = modelo(x_train,y_train,qtd_camadas_ocultas,nos_cam_oculta)
    epocas = 120
    obj.mlp(eta=eta,epocas=epocas,alpha=alpha,arquivo_saida=nome_arquivo_saida, x_val=x_val, y_val=y_val)
    pd.DataFrame(
        [[nos_cam_oculta,eta,alpha,obj.erro_quadratico_medio[-1], retorna_erro(obj.prever(x_test),y_test)]],
        columns=["Qtd_nos_camada_oculta","Eta","Alpha","Acuracidade_treinamento","Erro_quadratico_medio_conjunto_teste"],
        ).to_csv(nome_arquivo_saida+'_Erro_Verdadeiro.csv')

    gera_matriz_confusao(obj.prever(x_test),y_test).to_csv(nome_arquivo_saida+"_Matriz_Confusao.csv")

def k_fold_cross_validation(nos_cam_oculta, eta, alpha):
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

    erro_folds = []
    fold=0
    for limite_superior in intervalos:
        x_val = X[limite_inferior:limite_superior]            
        y_val = y[limite_inferior:limite_superior] 
        x_train = np.concatenate((X[0:limite_inferior],X[limite_superior:]))
        y_train = np.concatenate((y[0:limite_inferior],y[limite_superior:]))
        limite_inferior = limite_superior
        
        qtd_camadas_ocultas = 1
        obj = modelo(x_train,y_train,qtd_camadas_ocultas,nos_cam_oculta)
        epocas = 120
        obj.mlp(eta=eta,epocas=epocas,alpha=alpha,registrar_erro_epocas=False)
        erro_folds.append([nos_cam_oculta,eta,alpha,obj.erro_quadratico_medio[-1], retorna_erro(obj.prever(x_val),y_val)])
        print("Terminou: Qtd nos: " + str(nos_cam_oculta) + " - Eta: " + str(eta) + " - Alpha: " + str(alpha))
        fold+=1


    #Registra o erro de cada fold
    df_saida = pd.DataFrame(erro_folds,columns=["Qtd_nos_camada_oculta","Eta","Alpha","Acuracidade_treinamento","Erro_quadratico_medio_fold"])[["Qtd_nos_camada_oculta","Eta","Alpha","Erro_quadratico_medio_fold"]]
    
    #Adiono no DataFrame a media com index = Media
    media = df_saida["Erro_quadratico_medio"].mean()
    linha_media = pd.DataFrame([[nos_cam_oculta,eta,alpha,None, media]],index=['Media'])
    df_saida.append(linha_media)

    # Salvo arquivo
    df_saida.to_csv(nome_arquivo_saida+"_Erro_de_cada_fold.csv")
    
    #Treino novamente o modelo sem os dados de teste:
    obj = modelo(X,y,qtd_camadas_ocultas,nos_cam_oculta)
    obj.mlp(eta=eta,epocas=epocas,alpha=alpha, registrar_erro_epocas=False)

    #registrar a media artimetica dos erros    
    pd.DataFrame(
        [[nos_cam_oculta,eta,alpha,obj.erro_quadratico_medio[-1], retorna_erro(obj.prever(x_test),y_test)]],
        columns=["Qtd_nos_camada_oculta","Eta","Alpha","Acuracidade_treinamento","Erro_quadratico_medio_conjunto_teste"],
        ).to_csv(nome_arquivo_saida+'_Erro_Verdadeiro.csv')
    
    gera_matriz_confusao(obj.prever(x_test),y_test).to_csv(nome_arquivo_saida+"_Matriz_Confusao.csv")
    
    print("Erro médio do K-Fold Cross Validation: ", df_saida["Erro_quadratico_medio"].mean())






if (__name__ == '__main__'):

    path = 'optdigits.dat'
    nome_arquivo_saida = "Cross_nova_tentativa"
    nos_cam_oculta = 83
    eta = 0.02
    alpha=0.011556

    ################################################################################
    ################################################################################
    #Importar dados:################################################################
    #path = 'optdigits.dat'
    df = pd.read_csv(path)

    #embaralhar os dados
    df = embaralhar_dado(df)
    #obtenho o valor de todos os y e transformo uma coluna de vetores
    y = tranformar_categoria_em_numeros(df[str(df.shape[1]-1)])

    #obtenho o valor de todos os X
    X = df.drop([str(df.shape[1]-1)],axis=1).values

    #Separo 80% do X e y para treinar e validar e 20% para conjunto de teste
    X, x_test, y, y_test = split_dados(X,y,tamanho_percentual=.8)

    #Treino novamente o modelo sem os dados de teste:
    obj = modelo(X,y,qtd_camadas_ocultas,nos_cam_oculta)
    obj.mlp(eta=eta,epocas=120,alpha=alpha, registrar_erro_epocas=False)

    #registrar a media artimetica dos erros    
    pd.DataFrame(
        [[nos_cam_oculta,eta,alpha,obj.erro_quadratico_medio[-1], retorna_erro(obj.prever(x_test),y_test)]],
        columns=["Qtd_nos_camada_oculta","Eta","Alpha","Acuracidade_treinamento","Erro_quadratico_medio_conjunto_teste"],
        ).to_csv(nome_arquivo_saida+'_Erro_Verdadeiro.csv')
    
    gera_matriz_confusao(obj.prever(x_test),y_test).to_csv(nome_arquivo_saida+"NOVA_Matriz_Confusao.csv")