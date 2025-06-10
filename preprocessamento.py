import pandas as pd
import numpy as np
from sklearn import preprocessing


file_path = "https://raw.githubusercontent.com/amandexspeed/Diabetes-IA/b260364770c2f24cae65a1f0ef76dea6f5f6dd5c/treinamento.csv"

base_dados = pd.read_csv(file_path, sep=',',encoding='utf-8')

print("Pré-processando dados...")

#Processa todos os dados da base de dados
base_dados_perfeitos = base_dados.dropna()
atributos_treinamento = np.column_stack((base_dados['Pregnancies'], base_dados['Glucose'], base_dados['BloodPressure'], base_dados['SkinThickness'], base_dados['Insulin'], base_dados['BMI'], base_dados['DiabetesPedigreeFunction'], base_dados['Age']))
classes_treinamento = np.hstack(base_dados['Outcome'])

#Processa apenas os dados perfeitos da base de dados

# Filtrar apenas linhas onde nenhum dos atributos é zero
base_dados_filtrado = base_dados[
    (base_dados[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] != 0).all(axis=1)
]
# Criar os arrays de atributos e classes
atributos_treinamento_perfeitos = base_dados_filtrado[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values
classes_treinamento_perfeitos = base_dados_filtrado['Outcome'].values

print("Dados pré-processados com sucesso!")

if __name__ == "__main__":
    print("Dados pré-processados:"
                "\n\nAtributos de treinamento:\n", atributos_treinamento_perfeitos,
          "\n\nClasses de treinamento:\n", classes_treinamento_perfeitos)