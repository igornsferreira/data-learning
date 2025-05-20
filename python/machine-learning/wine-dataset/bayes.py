from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.decomposition import PCA

wine = load_wine()
X = wine.data
y = wine.target
target_names = wine.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

start = time.time()
model = GaussianNB(var_smoothing=1e-8)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
execution_time = time.time() - start

print("Modelo: Naive Bayes")
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Tempo de execução: {:.4f} segundos".format(execution_time))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred, target_names=target_names))

cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:\n", cm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.title("Matriz de Confusão - Naive Bayes")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.tight_layout()
plt.savefig('Bayes_MatrizDeConfusão.pdf')
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="Set1", s=70)
plt.title("Redução de Dimensionalidade com PCA (2D)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Classe")
plt.tight_layout()
plt.savefig("PCA_WineDataset_2D.pdf")
plt.show()