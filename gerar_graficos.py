import re 
import seaborn as sns
import numpy as np 
import pandas as pd 
import random
from pylab import *
from matplotlib.pyplot import plot, show, draw, figure, cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sns.set_style("whitegrid", {'axes.grid' : False})



base_teste = pd.read_csv('IA-main\Resultados_hold_out_1_VSCODE.csv')
base_teste = base_teste.sort_values('Erro_quadratico_medio')

plt.figure(figsize=(30,12))
plt.plot(base_teste['Eta'])
plt.legend()
plt.rcParams.update({'font.size': 20})
plt.show()
plt.savefig('Erro Quadratico Treinamento e Erro Quadratico Validacao x Epoca.png', format='png')
