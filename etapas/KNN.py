def funcao_KNN(df):
    # Lista para armazenar os resultados da acurácia
    test_accuracies = []
    num_tests = 5

    # Loop para realizar os testes múltiplas vezes
    for i in range(num_tests):
        # Dividir os dados de forma aleatória em features (X) e target (y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)  # Modifique a semente aleatória aqui

        # Modelo KNN
        knn = KNeighborsClassifier()

        # Definir o espaço de busca para hiperparâmetros do KNN
        param_dist_knn = {
            'n_neighbors': randint(1, 20),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]
        }

        # RandomizedSearch para KNN
        random_search_knn = RandomizedSearchCV(knn, param_distributions=param_dist_knn, n_iter=100, cv=5, scoring='accuracy', random_state=i)  # Modifique a semente aleatória aqui
        random_search_knn.fit(X_train, y_train)

        # Avaliar o modelo nos dados de teste
        y_pred = random_search_knn.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_accuracies.append(test_accuracy)

        # Exibir os melhores parâmetros encontrados para KNN
        print(f"Melhores parâmetros encontrados para o teste {i+1}:")
        print(random_search_knn.best_params_)
        print(f"Acurácia de teste para o teste {i+1}: {test_accuracy}")
