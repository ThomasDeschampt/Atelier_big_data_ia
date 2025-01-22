from sklearn import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_extraction.text import CountVectorizer

#Récupération du fichier de data depuis /Data/dataset.csv
dataset = pd.read_csv('Data/dataset.csv')
dataset.head()

# Différencation de la maladie et des symptômes
# On prend seulement la première colones pour la maladie
X = dataset.iloc[:, 0].values

# On prend les autres colones pour les symptômes
y = dataset.iloc[:, 1:].values

print('Les maladies sont : ')
print(X.head())
print('Les symptomes sont : ')
print(y.head())

vectorizer = CountVectorizer()
# Division du dataset en 2 parties : 80% pour l'entrainement et 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("taille de l'ensemble d'apprentissage : ", X_train.shape)
print("taille de l'ensemble de test : ", X_test.shape)

#Creation de l'arbre d'entrainement
arbre = DecisionTreeClassifier(criterion = "gini", max_depth=3, random_state=42)
arbre.fit(X_train, y_train)

#Evaluation de la precision de l'arbre
score = arbre.score(X_test, y_test)
print("Precision de l'arbre : ", score)




