from sklearn.linear_model import Perceptron
from preprocessamento import atributos_treinamento, classes_treinamento,atributos_treinamento_perfeitos ,classes_treinamento_perfeitos

neuronio_treinamento_geral = Perceptron()
neuronio_treinamento_perfeitos = Perceptron()

print("Treinando o neurônio com todos os dados...\n")
neuronio_treinamento_geral.fit(atributos_treinamento, classes_treinamento)

print("Acurácia do modelo(dados gerais):")
print(neuronio_treinamento_geral.score(atributos_treinamento, classes_treinamento),"\n")

print("Acurácia do modelo(dados perfeitos):")
print(neuronio_treinamento_geral.score(atributos_treinamento_perfeitos, classes_treinamento_perfeitos),"\n")

print("Pesos do neurônio:")
print(neuronio_treinamento_geral.coef_,"\n")

print("Bias do neurônio:")
print(neuronio_treinamento_geral.intercept_,"\n")

print("\nTreinando o neurônio com dados perfeitos...\n")
neuronio_treinamento_perfeitos.fit(atributos_treinamento_perfeitos, classes_treinamento_perfeitos)

print("Acurácia do modelo (dados gerais):")
print(neuronio_treinamento_perfeitos.score(atributos_treinamento, classes_treinamento),"\n")

print("Acurácia do modelo (dados perfeitos):")
print(neuronio_treinamento_perfeitos.score(atributos_treinamento_perfeitos, classes_treinamento_perfeitos),"\n")

print("Pesos do neurônio:")
print(neuronio_treinamento_perfeitos.coef_,"\n")

print("Bias do neurônio:")
print(neuronio_treinamento_perfeitos.intercept_,"\n")