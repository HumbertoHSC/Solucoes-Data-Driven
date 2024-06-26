from etapas.conjunto_treinamento import funcao_conjunto_treinamento
from etapas.EDA import funcao_EDA
from etapas.preparacao import funcao_preparacao
from etapas.KNN import funcao_KNN
from etapas.random_forest import funcao_random_forest
import pandas as pd

df = pd.read_excel('arquivo.xlsx')

def main():
    # Chamar as funções conforme necessário
    funcao_conjunto_treinamento()
    funcao_EDA()
    funcao_preparacao()
    funcao_KNN()
    funcao_random_forest()

if __name__ == "__main__":
    main()