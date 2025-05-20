from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

wine = load_wine()
X = wine.data
y = wine.target
target_names = wine.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

start = time.time()
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
execution_time = time.time() - start

print("Modelo: Decision Tree")
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Tempo de execução: {:.4f} segundos".format(execution_time))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred, target_names=target_names))

cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:\n", cm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=target_names, yticklabels=target_names)
plt.title("Matriz de Confusão - Decision Tree")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.tight_layout()
plt.savefig('DecisionTree_MatrizDeConfusao.pdf')
plt.show()

importances = model.feature_importances_
features = wine.feature_names
df_importances = pd.DataFrame({
    'Atributo': features,
    'Importância': importances
}).sort_values(by='Importância', ascending=False)

print("\nImportância das variáveis (ordem decrescente):\n")
print(df_importances)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importância', y='Atributo', data=df_importances, palette="crest")
plt.title('Importância dos Atributos - Decision Tree')
plt.xlabel('Importância')
plt.ylabel('Atributo')
plt.tight_layout()
plt.savefig('DecisionTree_ImportanciaAtributos.pdf')
plt.show()