"""
Fase 3: Regressão Polinomial - Overfitting
"""

import random
from numpy import genfromtxt
import numpy as np
import math
import matplotlib.pyplot as plt
from functools import reduce
import operator
from mpl_toolkits.mplot3d import Axes3D


def lerDados():
    return genfromtxt('data_preg.csv', delimiter=',')


def pegarColunas(matriz, vetorColunas):
    result = matriz[:, vetorColunas]
    return result


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


#Gera a linha de regressão polinomial
def geraLinhaRegressaoPolinomial(x, y, n, outroX=None, outroY=None):
    # Chama a função polyfit e inverte seu resultado, pois sua numeração coeficiente é invertida
    b = np.polyfit(x, y, n)[::-1]

    def somatoria(b, x):
        resultado = 0
        for index in range(n + 1):
            resultado += b[index] * x ** index
        return np.array(resultado)

    if outroX is None and outroY is None:
        return somatoria(b, x)
    return somatoria(b, x), somatoria(b, outroX)
    return somatoria(b, x), somatoria(b, x)


#Gera o erro quadrático médio
def erroQuadraticoMedio(y1, y2):
    if len(y1) != len(y2):
        y1y = y1 if len(y1) > len(y2) else np.array([y1[i] if i < len(y1) else y2[i] for i in range(len(y2))])
        y2y = y2 if len(y2) > len(y1) else np.array([y2[i] if i < len(y2) else y1[i] for i in range(len(y1))])
        return reduce(operator.add, (y1y - y2y) ** 2) / len(y2y)
    return reduce(operator.add, (y1 - y2) ** 2) / len(y2)


# a) Baixe o arquivo data_preg.mat ou data_preg.svg. A primeira coluna representa os valores de x e a segunda
# coluna representa os valores de y.
matriz = lerDados()
x = pegarColunas(matriz, 0)
y = pegarColunas(matriz, 1)

# b) Faça o Gráfico de dispersão dos dados.
fig = plt.figure('Gráfico de disperção')
plt.title('Vermelho: N1  Verde: N2  Preto: N3  Amarelo: N8')
plt.grid(True)
plt.scatter(x, y)

# c) Use a função polyfit para gerar a linha de regressão para N = 1 e trace-o no gráfico de dispersão na cor
# vermelha (plot (x, y, 'r')). (observe que nesta função a numeração coeficiente é invertida! β0=βN , β1=βN−1 ,
# β2=βN−2 , ...βN=β0)
y1 = geraLinhaRegressaoPolinomial(x, y, 1)
plt.plot(x, y1, 'r')

# d) Trace a linha de regressão para N = 2 no gráfico na cor verde.
y2 = geraLinhaRegressaoPolinomial(x, y, 2)
plt.plot(x, y2, 'g')

# e) Trace a linha de regressão para N = 3 no gráfico na cor preta.
y3 = geraLinhaRegressaoPolinomial(x, y, 3)
plt.plot(x, y3, 'k')

# f) Trace a linha de regressão para N = 8 no gráfico na cor amarela.
y4 = geraLinhaRegressaoPolinomial(x, y, 8)
plt.plot(x, y4, 'y')


# g) Calcule o Erro Quadrático Médio (EQM) para cada linha de regressão. Qual é o mais preciso?
print("Erro Quadrático Médio (EQM) para cada linha de regressão:")
tuplaItens = [('N1', y1), ('N2', y2), ('N3', y3), ('N8', y4)]
eqm = [[item, erroQuadraticoMedio(item[1], y)] for item in tuplaItens]
for i in range(len(eqm)):
    print('EQM da regressão de %s' % eqm[i][0][0], eqm[i][1])
plt.show()

# h) Para evitar o overfitting, divida os dados aleatoriamente em Dados de Treinamento e Dados de Teste. Use
# os primeiros 10% dos dados como conjunto de teste, e o resto como de treinamento.
treinamentoResults = []
iteracoes = 5000

print("\nTreinamento para evitar underfitting e overfitting:")
print("Será feita " + str(iteracoes) + " iterações\n")

for i in range(iteracoes):
    print("iteração nº " + str(i))
    #Criando uma cópia dos dados da matriz e embarralhando os valores da mesma
    copiaMatriz = np.copy(matriz)
    random.shuffle(copiaMatriz)

    tamanho = len(x)
    indiceDos10Porcento = round((tamanho * 10) / 100)

    #Selecionando apenas 10% dos itens para teste
    matrizTeste = copiaMatriz[0:indiceDos10Porcento]
    #Mantendo 90% dos itens para treinamento
    matrizTreinamento = copiaMatriz[indiceDos10Porcento:tamanho]
    matrizTreinamento.sort()

    matrizTesteX = pegarColunas(matrizTeste, 0)
    matrizTesteY = pegarColunas(matrizTeste, 1)
    matrizTreinamentoX = pegarColunas(matrizTreinamento, 0)
    matrizTreinamentoY = pegarColunas(matrizTreinamento, 1)

    # i) Repita os passos de c - f, mas agora use apenas os dados de treinamento para ajustar a linha de regressão.
    #Calculando regressão polinomial para treino e teste
    treinamentoY1, testeY1 = geraLinhaRegressaoPolinomial(matrizTreinamentoX, matrizTreinamentoY, 1, matrizTesteX, matrizTesteY)
    treinamentoY2, testeY2 = geraLinhaRegressaoPolinomial(matrizTreinamentoX, matrizTreinamentoY, 2, matrizTesteX, matrizTesteY)
    treinamentoY3, testeY3 = geraLinhaRegressaoPolinomial(matrizTreinamentoX, matrizTreinamentoY, 3, matrizTesteX, matrizTesteY)
    treinamentoY4, testeY4 = geraLinhaRegressaoPolinomial(matrizTreinamentoX, matrizTreinamentoY, 8, matrizTesteX, matrizTesteY)

    # j) Repita o passo g, mas agora utilize somente os dados de Teste para calcular o erro.
    # Calcula o erro quadratico médio EQM para cada uma das regressões
    tuplaItens = [
        ('N1', treinamentoY1, testeY1),
        ('N2', treinamentoY2, testeY2),
        ('N3', treinamentoY3, testeY3),
        ('N8', treinamentoY4, testeY4)]

    eqm = [[y, erroQuadraticoMedio(y[1], y[2])] for y in tuplaItens]
    # Salva o melhor resultado na lista de resultados
    melhorResultado = None
    for item in eqm:
        melhorResultado = item if melhorResultado is None or item[1] < melhorResultado[1] else melhorResultado
    treinamentoResults.append(melhorResultado)

print("\nResultado:")
# Informando o número de vezes em que cada valor de N teve o melhor resultado
for treino in ['N1', 'N2', 'N3', 'N8']:
    numeroVezes = list(filter(lambda item: item[0][0] == treino, treinamentoResults))
    print('- %s teve o melhor EQM %s vezes' % (treino, len(numeroVezes)))



# k) Que método é o mais preciso neste caso?

"""
Após executar o programa 10 vezes, utilizando 5 mil iterações, obtemos o seguintes resultados:

                    Execução 1:
                    Resultado:
                    - N1 teve o melhor EQM 4252 vezes
                    - N2 teve o melhor EQM 566 vezes
                    - N3 teve o melhor EQM 169 vezes
                    - N8 teve o melhor EQM 13 vezes
                    
                    Execução 2:
                    Resultado:
                    - N1 teve o melhor EQM 4241 vezes
                    - N2 teve o melhor EQM 561 vezes
                    - N3 teve o melhor EQM 176 vezes
                    - N8 teve o melhor EQM 22 vezes
                    
                    Execução 3:
                    Resultado:
                    - N1 teve o melhor EQM 4271 vezes
                    - N2 teve o melhor EQM 563 vezes
                    - N3 teve o melhor EQM 153 vezes
                    - N8 teve o melhor EQM 13 vezes
                    
                    Execução 4:
                    Resultado:
                    - N1 teve o melhor EQM 4281 vezes
                    - N2 teve o melhor EQM 532 vezes
                    - N3 teve o melhor EQM 175 vezes
                    - N8 teve o melhor EQM 12 vezes
                    
                    Execução 5:
                    Resultado:
                    - N1 teve o melhor EQM 4248 vezes
                    - N2 teve o melhor EQM 564 vezes
                    - N3 teve o melhor EQM 172 vezes
                    - N8 teve o melhor EQM 16 vezes
                    
                    Execução 6:
                    Resultado:
                    - N1 teve o melhor EQM 4228 vezes
                    - N2 teve o melhor EQM 585 vezes
                    - N3 teve o melhor EQM 175 vezes
                    - N8 teve o melhor EQM 12 vezes
                    
                    Execução 7:
                    Resultado:
                    - N1 teve o melhor EQM 4228 vezes
                    - N2 teve o melhor EQM 595 vezes
                    - N3 teve o melhor EQM 165 vezes
                    - N8 teve o melhor EQM 12 vezes
                    
                    Execução 8:
                    Resultado:
                    - N1 teve o melhor EQM 4231 vezes
                    - N2 teve o melhor EQM 590 vezes
                    - N3 teve o melhor EQM 167 vezes
                    - N8 teve o melhor EQM 12 vezes
                    
                    Execução 9:
                    Resultado:
                    - N1 teve o melhor EQM 4278 vezes
                    - N2 teve o melhor EQM 541 vezes
                    - N3 teve o melhor EQM 166 vezes
                    - N8 teve o melhor EQM 15 vezes
                    
                    Execução 10:
                    Resultado:
                    - N1 teve o melhor EQM 4237 vezes
                    - N2 teve o melhor EQM 562 vezes
                    - N3 teve o melhor EQM 179 vezes
                    - N8 teve o melhor EQM 22 vezes

Portanto, podemos notar que a regressão N1 é aproximadamente 84% melhor em relação as demais. 

"""