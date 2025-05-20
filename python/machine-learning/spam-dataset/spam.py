# Aluno: Igor Ferreira
# Atividade: Spam Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv("Conjunto_de_Dados_de_E-mail_para_Classifica__o_de_Spam (1).csv")

df["tem_anexo"] = df["tem_anexo"].map({"Sim": 1, "Não": 0})

X = df[["num_palavras", "num_links", "tem_anexo", "num_caracteres_especiais"]]
y = df["spam"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Acurácia: {accuracy * 100:.2f} %")
print("Matriz de Confusão:")
print(conf_matrix)
