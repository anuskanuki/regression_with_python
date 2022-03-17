"""
Fase 1: Análise de Correlação e Regressão Linear
"""

import numpy as np
import math
import matplotlib.pyplot as plt

x1 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]

x2 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y2 = [9.14, 8.14, 8.47, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]

x3 = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19]
y3 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50]


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


#Efetua os chamados aos métodos do enunciado
def calcula(x, y):
    # a. Faça um Gráfico de Dispersão (veja função scatter).
    plt.scatter(x, y)

    # b. Calcule o coeficiente de correlação.
    r = correlacao(x, y)

    # c. Trace a linha da regressão no Gráfico de Dispersão (veja a função plot)
    resultRegressao = regressao(x, y)
    b0 = resultRegressao[0]
    b1 = resultRegressao[1]
    pontosRegressao = retaRegressao(b0, b1, x)

    #Exibindo no console
    print("r: " + str(r))
    print("b0: " + str(b0))
    print("b1: " + str(b1))
    print("pontosRegressao " + str(pontosRegressao))
    print()

    # d. Mostre os coeficientes de correlação e regressão no Gráfico de Dispersão (utilize a função title)
    plt.plot(x, pontosRegressao)
    plt.title("r: %s    β0: %s    β1: %s" % (str(r), str(b0), str(b1)))
    plt.show()


#Chamada dos métodos
print("x1 | y1")
calcula(x1, y1)

print("x2 | y2")
calcula(x2, y2)

print("x3 | y3")
calcula(x3, y3)