def funcao_random_forest(df):
    # Loop para realizar os testes múltiplas vezes
    for i in range(num_tests):
        # Dividir os dados em features (X) e target (y)
        X = df.drop('Class', axis=1)
        y = df['Class']

        # Dividir os dados em conjunto de treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modelo Random Forest
        rf = RandomForestClassifier()

        # Definir o espaço de busca para hiperparâmetros do Random Forest
        param_dist_rf = {
            'n_estimators': randint(10, 100),
            'max_depth': randint(1, 20),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'max_features': ['auto', 'sqrt', 'log2']
        }

        # RandomizedSearch para Random Forest
        random_search_rf = RandomizedSearchCV(rf, param_distributions=param_dist_rf, n_iter=100, cv=5, scoring='accuracy', random_state=42)
        random_search_rf.fit(X_train, y_train)

        # Avaliar o modelo nos dados de teste
        y_pred = random_search_rf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_accuracies.append(test_accuracy)

        # Exibir os melhores parâmetros encontrados para Random Forest
        print(f"Melhores parâmetros encontrados para o teste {i+1}:")
        print(random_search_rf.best_params_)
        print(f"Acurácia de teste para o teste {i+1}: {test_accuracy}")