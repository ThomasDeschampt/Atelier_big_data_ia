import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
import joblib

#Lecture du fichier csv
def charger_donnees(fichier):
    print(f"Chargement des données depuis {fichier}...")
    data = pd.read_csv(fichier)
    return data

#Préparation des données
#On identifie la dernière colone comme la colonne cible
def preparer_donnees(data):
    cols = data.columns[:-1]
    X = data[cols]
    y = data['prognosis']

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    return X, y, cols, le

#Entrainement du modèle
#On divise les données en données d'entrainement et de test
def entrainer_modele(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # On utilise un arbre de décision
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # On évalue le modèle
    scores = cross_val_score(clf, X_test, y_test, cv=3)
    print(f"Précision moyenne de l'arbre de décision : {scores.mean():.2f}")

    return clf

#Sauvegarde du modèle entrainé
def sauvegarder_modele(clf, le, cols, fichier_modele):
    modele = {'modele': clf, 'encoder': le, 'colonnes': cols}
    joblib.dump(modele, fichier_modele)
    print(f"Modèle sauvegardé dans {fichier_modele}")

#Fonction principale
def main():
    fichier_training = 'Data/Training.csv'
    data = charger_donnees(fichier_training)
    X, y, cols, le = preparer_donnees(data)

    clf = entrainer_modele(X, y)

    fichier_modele = 'Models/modele_ml.joblib'
    sauvegarder_modele(clf, le, cols, fichier_modele)

if __name__ == "__main__":
    main()