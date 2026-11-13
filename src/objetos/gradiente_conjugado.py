import numpy as np
from scipy.optimize import minimize

'''
Classe contendo a implementação do gradiente conjugado.

Args:
    matriz_notas: Matriz contendo a notas de cado usuario para cada filme.
    matriz_relacao: Matriz contendo 1 se o usuario avaliou o filme, e 0 caso contrario.
    dimensao: Dimensão dos vetores de atributos e parametros.

Atributos:
    self.dimensao: Dimensão dos vetores de atributos e parametros.
    self.matriz_notas: Matriz contendo a notas de cado usuario para cada filme.
    self.matriz_relacao: Matriz contendo 1 se o usuario avaliou o filme, e 0 caso contrario.
    self.numero_filmes: Número de filmes.
    self.numero_usuarios: Número de usuarios.
    self.matriz_atributos: Matriz de atributos para o filme i.
    self.matriz_parametros: Matriz de parametros para o usuario j.
    self.entrada: Vetor contendo os atributos e parametros, para o calculo do gradiente conjugado e da função custo.
    
Métodos:
    inicializar(self):
        Inicializa as matrizes e vetores com valores aleatorios.

    funcao_custo_gradiente(self,entrada):
        Cálcula o valor da função custo e dos gradientes.

        Args:
            entrada: Vetor contendo os atributos e parametros, para o calculo do gradiente conjugado e da função custo.
        
        Retorna:
            custo: Valor da função custo.
            grad: vetor concatenado dos gradientes
    
    rodar(self):
        Aplica o algoritmo do gradiente conjugado utilizando o scipy.minimize, obtendo as matrizes de parametros e atributos atualizada.

    notas_preditas(self):
        Calcula as notas preditas para cada filme com base na matriz de atributos e a matriz de parametros.

'''

class Gradiente_Conjugado():
    def __init__(self, matriz_notas, matriz_relacao, dimensao):
        self.dimensao = dimensao
        self.matriz_notas = matriz_notas
        self.matriz_relacao = matriz_relacao

        self.numero_filmes = self.matriz_notas.shape[0]
        self.numero_usuarios = self.matriz_notas.shape[1]
        
        self.matriz_atributos = None
        self.matriz_parametros = None

        self.matriz_atributos_flat = None
        self.matriz_parametros_flat = None

        self.entrada = None

        self.limite_randomico = 0.1

    def inicializar(self):
        self.matriz_atributos = np.random.uniform(-self.limite_randomico, self.limite_randomico, (self.numero_filmes,self.dimensao))
        self.matriz_parametros = np.random.uniform(-self.limite_randomico, self.limite_randomico, (self.numero_usuarios,self.dimensao))
        
        self.matriz_atributos_flat = self.matriz_atributos.ravel()
        self.matriz_parametros_flat = self.matriz_parametros.ravel()
        
        self.entrada = np.concatenate([self.matriz_atributos_flat,self.matriz_parametros_flat])

    def funcao_custo_gradiente(self, entrada):
#        valor = 0.0
#        relacao = np.where(self.matriz_relacao == 1)
#
#        tamanho_matriz = self.dimensao ** 2
#
#        for k in range(relacao[0].shape[0]):
#            i = relacao[0][k]
#            j = relacao[1][k]
#
#            inicio_i = i * self.dimensao
#            fim_i = (i + 1) * self.dimensao
#            atributo_i = entrada[inicio_i : fim_i]
#
#            inicio_j = tamanho_matriz + (j * self.dimensao)
#            fim_j = tamanho_matriz + ((j + 1) * self.dimensao)
#            parametro_j = entrada[inicio_j : fim_j]
#
#            valor += (1/2) * ((parametro_j.T @ atributo_i - self.matriz_notas[i,j])**2)
#
#        return valor

        matriz_atributos_reconstruida = entrada[:(self.numero_filmes * self.dimensao)].reshape((self.numero_filmes,self.dimensao))
        matriz_parametros_reconstruida = entrada[(self.numero_filmes * self.dimensao):].reshape((self.numero_usuarios, self.dimensao))

        erro = ((matriz_atributos_reconstruida @ matriz_parametros_reconstruida.T) - self.matriz_notas) * self.matriz_relacao

        custo = 0.5 * np.sum(erro ** 2)  

        grad_atributos = erro @ matriz_parametros_reconstruida
        grad_parametros = erro.T @ matriz_atributos_reconstruida

        grad = np.concatenate([grad_atributos.ravel(),grad_parametros.ravel()])

        return custo, grad

    def progresso(self, xk):
        self.iteracao += 1

        print(f"Iteracao: {self.iteracao} | Custo: {self.funcao_custo_gradiente(xk)[0]}")

    def rodar(self):
        self.iteracao = 0
        resultado = minimize(
            fun = self.funcao_custo_gradiente,
            x0 = self.entrada,
            method='CG',
            jac=True,
            callback=self.progresso,
            tol=1e-3,
            options={'disp': True, 'return_all':True}
        )

        self.entrada = resultado.x
        self.matriz_atributos_flat = self.entrada[:(self.numero_filmes * self.dimensao)]
        self.matriz_parametros_flat = self.entrada[(self.numero_filmes * self.dimensao):]
        self.matriz_atributos = self.matriz_atributos_flat.reshape((self.numero_filmes,self.dimensao))
        self.matriz_parametros = self.matriz_parametros_flat.reshape((self.numero_usuarios,self.dimensao))
        
    def notas_preditas(self):
        return self.matriz_atributos @ self.matriz_parametros.T