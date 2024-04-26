import pandas as pd
from sklearn.model_selection import train_test_split

def funcao_conjunto_treinamento(df):
    # Carregar os dados
    df = pd.read_excel('arquivo.xlsx')

    # 1. Seleção de características
    features = df.drop(columns=['Class']).columns

    # 2. Divisão dos dados em conjuntos de treinamento e validação
    train_df, val_df = train_test_split(df, test_size=0.05, random_state=42, stratify=df['Class'])

    # 3. Aplicação de filtros nas características do conjunto de treinamento
    train_df = train_df[train_df['Eccentricity'] >= 0.25]
    train_df = train_df[train_df['Solidity'] >= 0.94]
    train_df = train_df[train_df['roundness'] >= 0.53]
    train_df = train_df[train_df['ShapeFactor4'] >= 0.96]

    # 4. Separação das características e variáveis alvo dos conjuntos de treinamento e validação
    X_train = train_df[features]
    y_train = train_df['Class']

    X_val = val_df[features]
    y_val = val_df['Class']


    # Lista de características para padronização
    cols = ['Area', 'ConvexArea', 'MajorAxisLength', 'Perimeter', 'MinorAxisLength', 'EquivDiameter', 'Eccentricity',
            'ShapeFactor2', 'Extent', 'roundness', 'AspectRation', 'Compactness', 'ShapeFactor1', 'ShapeFactor3',
            'ShapeFactor4', 'Solidity', 'Bounding_rectangular_area']

    # Instanciar o StandardScaler
    sc = StandardScaler()

    # Padronizar características nos conjuntos de treinamento
    X_train_scaled = sc.fit_transform(X_train[cols])  # Ajusta o scaler e transforma os dados
    X_train[cols] = pd.DataFrame(X_train_scaled, index=X_train.index)  # Substitui as características padronizadas no DataFrame

    # Padronizar características nos conjuntos de validação (usando o scaler ajustado no conjunto de treinamento)
    X_val_scaled = sc.transform(X_val[cols])  # Apenas transforma os dados usando o scaler ajustado
    X_val[cols] = pd.DataFrame(X_val_scaled, index=X_val.index)  # Substitui as características padronizadas no DataFrame de validação

    # Inicializando o classificador MLP com os parâmetros especificados
    model_mlp = MLPClassifier(random_state=1, max_iter=500, alpha=0.005)

    # Treinando o modelo com os dados de treinamento
    model_mlp.fit(X_train, y_train)

    # Calculando e imprimindo o F1 score do conjunto de treinamento
    print("F1 Score do Treinamento: ", metrics.f1_score(y_train, model_mlp.predict(X_train), average='micro'))

    # Calculando e imprimindo o F1 score do conjunto de validação
    print("F1 Score da Validação: ", metrics.f1_score(y_val, model_mlp.predict(X_val), average='micro'))

    # Calculando a matriz de confusão para os dados de validação
    conf_matrix = confusion_matrix(y_val, model_mlp.predict(X_val))

    # Exibindo a matriz de confusão
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predição')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão - Feijão')
    plt.show()


    # Inicializando o classificador XGBoost com os parâmetros especificados
    model_xgb = xgb.XGBClassifier(random_state=42, verbosity=0, min_child_weight=2,
                                max_depth=4, learning_rate=0.15, gamma=0.22, colsample_bytree=0.5)

    # Treinando o modelo com os dados de treinamento
    model_xgb.fit(X_train, y_train)

    # Calculando e imprimindo o F1 score do conjunto de treinamento
    print("F1 Score do Treinamento: ", metrics.f1_score(y_train, model_xgb.predict(X_train), average='micro'))

    # Calculando e imprimindo o F1 score do conjunto de validação
    print("F1 Score da Validação: ", metrics.f1_score(y_val, model_xgb.predict(X_val), average='micro'))

    # Exibindo a matriz de confusão
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predição')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão - Dry Bean')
    plt.show()

    # Lista para armazenar os resultados da acurácia
    test_accuracies = []
    num_tests = 10