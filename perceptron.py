from sklearn.linear_model import Perceptron
from preprocessamento import atributos_treinamento, classes_treinamento,atributos_treinamento_perfeitos ,classes_treinamento_perfeitos

neuronio_treinamento_geral = Perceptron()
neuronio_treinamento_perfeitos = Perceptron()

print("Treinando o neurônio com todos os dados...")
neuronio_treinamento_geral.fit(atributos_treinamento, classes_treinamento)

print("Acurácia do modelo:")
print(neuronio_treinamento_geral.score(atributos_treinamento, classes_treinamento))

print("Pesos do neurônio:")
print(neuronio_treinamento_geral.coef_)

print("Bias do neurônio:")
print(neuronio_treinamento_geral.intercept_)

print("Treinando o neurônio com dados perfeitos...")
neuronio_treinamento_perfeitos.fit(atributos_treinamento_perfeitos, classes_treinamento_perfeitos)

print("Acurácia do modelo:")
print(neuronio_treinamento_perfeitos.score(atributos_treinamento_perfeitos, classes_treinamento_perfeitos))

print("Pesos do neurônio:")
print(neuronio_treinamento_perfeitos.coef_)

print("Bias do neurônio:")
print(neuronio_treinamento_perfeitos.intercept_)