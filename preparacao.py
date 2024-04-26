import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder


def funcao_preparacao(df):
    # Carregar os dados
    df = pd.read_excel('arquivo.xlsx')

    # 1. Tratamento de valores ausentes
    mean = df['coluna'].mean()
    df['coluna'].fillna(mean, inplace=True)

    # 2. Tratamento de outliers
    z_scores = stats.zscore(df['coluna'])
    outliers = (abs(z_scores) > 3)
    df = df[~outliers]

    # 3. Normalização e Padronização de Dados
    scaler = MinMaxScaler()
    df_normalized = scaler.fit_transform(df)

    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(df)

    # 4. Tratamento de Variáveis Categóricas
    df_encoded = pd.get_dummies(df, columns=['coluna_categorica'])

    label_encoder = LabelEncoder()
    df['coluna_categorica'] = label_encoder.fit_transform(df['coluna_categorica'])

    # 5. Remoção de Duplicatas
    df = df.drop_duplicates()

    # 6. Tratamento de Dados Temporais
    df['data'] = pd.to_datetime(df['data'])
    df['ano'] = df['data'].dt.year
    df['mes'] = df['data'].dt.month

    # Exibir informações sobre os dados limpos
    print(df.info())
