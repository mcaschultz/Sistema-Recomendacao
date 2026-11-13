import numpy as np
from scipy.io import loadmat

'''
Classe principal para manuseamento dos dados do problema.

Atributos:
    self.dados: Dados brutos do arquivo .mat.
    self.matriz_notas: Matriz (número filmes)x(números usuarios) contendo notas de 1 a 5 dada pelo usuario j para o filme i.
    self.matriz_relacao: Matriz (número filmes)x(números usuarios) com 1 se o usuario j avaliou o filme i, e 0 caso contrario.

Métodos:
    __init__(self):
        Inicializa todos os atributos com None.
    
    ler(self,caminho_arquivo):
        Realiza a leitura e separação das matrizes, salvando nos atributos correspondentes.

        Args:
            caminho_arquivo: Caminho contendo o arquivo .mat.

'''

class Dados():
    def __init__(self):
        self.dados = None
        self.matriz_notas = None
        self.matriz_relacao = None

    def ler(self,caminho_arquivo):
        self.dados = loadmat(caminho_arquivo)
        self.matriz_notas = np.asanyarray(self.dados["Y"], dtype=np.float32)
        self.matriz_relacao = np.asanyarray(self.dados["R"],dtype=np.float32)