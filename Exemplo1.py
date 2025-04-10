# Curso: Análise e Desenvolvimento de Sistemas - UNIFACiSA
# Autor: Ricardo Roberto. Data: 10/04 - 19h27.

# Importando os dados para a máquina local.
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# criando objetos que vão aparecer na tela do sistema.

st.title(" Sistema Classificação de Flores ìris")

# Carregando a base de dados para um dataframe e separando as features de seleção e classicação.
iris = load_iris()
X = iris.data
y = iris.target 

# exibindo os dados em formato de planilhas.
st.write("### Visualização de dados")
st.dataframe(pd.DataFrame(X, columns=iris.feature_names).assign(target=y))

# Separando a base em 70% pra treino e 30% para testes.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
k = st.slider("Escolha o valor de K para o KNN:", 1, 15, 5)

# Fazer o treinamento do Modelo de ML.
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Avaliação do Modelo de Machine Learning com Acuracia e o relatório de classificação.
acc = accuracy_score(y_test, y_pred)
st.success(f"Acurácia do Modelo foi: {acc:.2f}")

# Exibir o relatório de Classificação..
st.text("Relatório de Classificação")
st.text(classification_report(y_test, y_pred, target_names=iris.target_names))


