import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import joblib

# Lecture des données
def charger_donnees(fichier):
    print(f"Chargement des données depuis {fichier}...")
    data = pd.read_csv(fichier)
    return data

# Préparation des données
def preparer_donnees(data):
    cols = data.columns[:-2]
    X = data[cols]
    y = data['doctor'] + data['adresse']

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    return X, y, cols, le

# Entraînement du modèle
def entrainer_modele(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Utilisation d'un Random Forest pour des prédictions probabilistes
    clf = DecisionTreeClassifier(random_state=42, max_depth=100)
    clf.fit(X_train, y_train)

    return clf

# Sauvegarde du modèle
def sauvegarder_modele(clf, le, cols, fichier_modele):
    modele = {'modele': clf, 'encoder': le, 'colonnes': cols}
    joblib.dump(modele, fichier_modele)
    print(f"Modèle sauvegardé dans {fichier_modele}")

# Fonction principale
def main():
    fichier_training = 'Data/doctor.csv'
    data = charger_donnees(fichier_training)
    X, y, cols, le = preparer_donnees(data)

    clf = entrainer_modele(X, y)

    fichier_modele = 'Models/modele_mll.joblib'
    sauvegarder_modele(clf, le, cols, fichier_modele)

if __name__ == "__main__":
    main()
