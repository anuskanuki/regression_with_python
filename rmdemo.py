"""
Fase 2: Análise de Regressão Linear Múltipla
"""

from numpy import genfromtxt
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def column(matrix, i):
    return [row[i] for row in matrix]

# 0 - Tamanho da casa
# 1 - Número de quartos
# 2 - Preço da casa
def lerDados():
    return genfromtxt('data.csv', delimiter=',')


def pegarColunas(matriz, vetorColunas):
    result = matriz[:, vetorColunas]
    return result


def pegarDuasPrimeirasColunasMatriz(matrizCsv):
    result = np.delete(matrizCsv, np.s_[2:3], axis=1)
    return result


def pegarTamanhoENumeroQuartos(matrizCsv):
    result = [[1, elem[0], elem[1]] for elem in matrizCsv]
    return np.array(result)


def pegarVetorPrecos(matrizCsv):
    return pegarColunas(matrizCsv, 2)


#Esta função deve calcular o coeficiente de correlação r
def correlacao(x, y):
    resultParteCima = 0
    resultParteBaixo = 0
    mediaX = np.sum(x) / len(x)
    mediaY = np.sum(y) / len(y)

    somaX = 0
    somaY = 0

    for indexI in range(0, len(x)):
        resultParteCima += (x[indexI] - mediaX) * (y[indexI] - mediaY)

        somaX += math.pow(x[indexI] - mediaX, 2)
        somaY += math.pow(y[indexI] - mediaY, 2)

    resultParteBaixo += math.sqrt(somaX * somaY)

    result = round((resultParteCima / resultParteBaixo), 4)
    return result


# Função para cálculo do b1
def calcularB1(x, y):
    resultParteCima = 0
    resultParteBaixo = 0
    mediaX = np.sum(x) / len(x)
    mediaY = np.sum(y) / len(y)

    for indexI in range(0, len(x)):
        resultParteCima += (x[indexI] - mediaX) * (y[indexI] - mediaY)
        resultParteBaixo += math.pow(x[indexI] - mediaX, 2)

    result = round((resultParteCima / resultParteBaixo), 4)
    return result


# Função para cálculo do b0
def calcularB0(x, y, b1):
    mediaX = np.sum(x) / len(x)
    mediaY = np.sum(y) / len(y)
    result = round((mediaY - (b1*mediaX)), 4)

    return result


#Esta função deve calcular a regressão, isto é, β0 e β1.
def regressao(x, y):
    b1 = calcularB1(x, y)
    b0 = calcularB0(x, y, b1)

    return [b0, b1]


#Retorna a reta de regressão
def retaRegressao(b0, b1, x):
    return b0 + np.dot(b1, x)


#Efetua o cálculo da regressão múltipla
def regmultipla(x, y):
    transpose = x.transpose()
    return np.linalg.inv(transpose.dot(x)).dot(transpose.dot(y))


#Retorna a reta da regressao multipla
def retaRegressaoMultipla(x, regressaoMultipla):
    if x is None:
        return x.dot(regressaoMultipla)
    return np.array(x).dot(regressaoMultipla)


# a) - Faça o download dos dados do arquivo data.mat ou data.csv. A primeira coluna é o tamanho da casa,
#     a segunda coluna é o número de quartos, e a terceira coluna é o preço da casa.
matrizCsv = lerDados()


# b) - Gere uma matriz X para as variáveis independentes (que são o tamanho da casa e o número de
#     quartos) e o vetor y da variável dependente (que é o preço).
#Primeira coluna preenchida com valor 1
matrizX = pegarTamanhoENumeroQuartos(matrizCsv)
vetorY = pegarVetorPrecos(matrizCsv)


# c) - Verifique a correlação e a regressão para Tamanho da casa e Preço, e, Número de quartos e Preço e
#    apresente os valores no gráfico de dispersão.
#Para a segunda coluna da matrizX
# Verificar a correlação e a regressão para Tamanho da casa e Preço e gerar o gráfico de dispersão
x0 = pegarColunas(matrizX, 1)
r = correlacao(x0, vetorY)
resultRegressao = regressao(x0, vetorY)
b0 = resultRegressao[0]
b1 = resultRegressao[1]
pontosRegressao = retaRegressao(b0, b1, x0)

fig = plt.figure('Figura 1 - Regressão entre o Tamanho da Casa e o Preço')
plt.scatter(x0, vetorY)
plt.title("Correlação: %s" % (str(r)))
plt.plot(x0, pontosRegressao, c=[1, 0, 0, 0.5])
plt.xlabel('Tamanho da Casa')
plt.ylabel('Preço')
plt.show()

#Para a terceira coluna da matrizX
# Verificar a correlação e a regressão para Número de Quartos e Preço e gerar o gráfico de dispersão
x0 = pegarColunas(matrizX, 2)
r = correlacao(x0, vetorY)
resultRegressao = regressao(x0, vetorY)
b0 = resultRegressao[0]
b1 = resultRegressao[1]
pontosRegressao = retaRegressao(b0, b1, x0)

fig = plt.figure('Figura 2 - Regressão entre o número de quartos e o preço')
plt.scatter(x0, vetorY)
plt.title("Correlação: %s" % (str(r)))
plt.plot(x0, pontosRegressao, c=[1, 0, 0, 0.5])
plt.xlabel('Número de quartos')
plt.ylabel('Preço')
plt.show()


# d) Faça o gráfico de dispersão em 3D com o tamanho da casa, número de quartos, e o preço da casa.
# Neste caso iremos trabalhar com o espaço 3D (verifique como usar Axes3D).
# e) Trace a linha da regressão no Gráfico de Dispersão. Você pode girar este gráfico para visualizar melhor os dados.
# f) Mostre na figura os coeficientes de correlação entre Tamanho da casa e Preço e Número de quartos e Preço.
regressaoMultiplaResult = regmultipla(matrizX, vetorY)
retaRegressaoMultiplaResult = retaRegressaoMultipla(matrizX, regressaoMultiplaResult)
fig = plt.figure('Figura 3 - Linha da regressão no Gráfico de Dispersão 3D')
ax = fig.gca(projection='3d')
ax.scatter(matrizX[:, 1], matrizX[:, 2], vetorY)
ax.plot(matrizX[:, 1], matrizX[:, 2], retaRegressaoMultiplaResult, c=[1, 0, 0, 0.5])
ax.set_xlabel('Tamanho da Casa')
ax.set_ylabel('Número de quartos')
ax.set_zlabel('Preço por (*100.000)')
plt.show()

# g) Calcule o preço de uma casa que tem tamanho de 1650 e 3 quartos. O resultado deve ser igual a 293081.
resultadoG = round(retaRegressaoMultipla(regressaoMultiplaResult, [1, 1650, 3]), 2)
print("Preço de uma casa de tamanho 1650 e 3 quartos: " + str(resultadoG))
