# Aluno: Igor Ferreira
# Atividade: Iris Dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Divisão entre treino e teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Função para treinar e avaliar o Modelo
def treinar_modelo(X_train, X_test, y_train, y_test, nome_modelo):
    modelo = DecisionTreeClassifier()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    print(f"\n--- {nome_modelo} ---")
    print("Acurácia:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

print(treinar_modelo(X_train, X_test, y_train, y_test, "Modelo 1 (Todos os atributos)"))

print(treinar_modelo(X_train[:, [1, 3]], X_test[:, [1, 3]], y_train, y_test, "Modelo 2 (Largura)"))

print(treinar_modelo(X_train[:, [0, 2]], X_test[:, [0, 2]], y_train, y_test, "Modelo 3 (Comprimento)"))
