import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def funcao_EDA(df):
    # 1. Retorna a forma do DataFrame
    print(df.shape)

    # 2. Fornece informações sobre o DataFrame
    print(df.info())

    # 3. Exibe as primeiras linhas do DataFrame
    print(df.head())

    # 4. Exibe as últimas linhas do DataFrame
    print(df.tail())

    # 5. Retorna os tipos de dados de cada coluna do DataFrame
    print(df.dtypes)

    # 6. Gera estatísticas descritivas das colunas numéricas do DataFrame
    print(df.describe())

    # 7. Calcula o número de valores nulos em cada coluna do DataFrame
    print(df.isnull().sum())

    # 8. Retorna os valores únicos da coluna 'Class'
    print(df['Class'].unique())

    # 9. Exibe a contagem de cada valor único na coluna 'Class'
    print(df['Class'].value_counts())

    # 10. Cria um gráfico de contagem para a coluna 'Class' usando Seaborn
    _ = sns.countplot(x='Class', data=df)

    # 11. Seleciona todas as colunas numéricas, exceto a coluna 'Class'
    Numeric_cols = df.drop(columns=['Class']).columns

    # 12. Gráfico de histogramas para cada coluna numérica com linhas verticais indicando a média de cada variável
    fig, ax = plt.subplots(4, 4, figsize=(15, 12))
    for variable, subplot in zip(Numeric_cols, ax.flatten()):
        g=sns.histplot(df[variable], bins=30, kde=True, ax=subplot)
        g.lines[0].set_color('crimson')
        g.axvline(x=df[variable].mean(), color='m', label='Mean', linestyle='--', linewidth=2)
    plt.tight_layout()

    # 13. Gráfico de boxplot para visualizar a distribuição das colunas numéricas em relação à coluna 'Class'
    fig, ax = plt.subplots(8, 2, figsize=(15, 25))
    for variable, subplot in zip(Numeric_cols, ax.flatten()):
        sns.boxplot(x=df['Class'], y= df[variable], ax=subplot)
    plt.tight_layout()

    # 14. Cria um mapa de calor para visualizar a correlação entre as colunas do DataFrame
    plt.figure(figsize=(12,12))
    sns.heatmap(df.corr("pearson"), vmin=-1, vmax=1, cmap='coolwarm', annot=True, square=True)
    plt.show()
