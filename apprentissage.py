import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

def charger_dataset(fichier_csv):
    try:
        data = pd.read_csv(fichier_csv)
        print("Dataset chargé avec succès.")
        return data
    except FileNotFoundError:
        print(f"Erreur : Le fichier {fichier_csv} est introuvable.")
        exit()
    except Exception as e:
        print(f"Erreur lors du chargement du dataset : {e}")
        exit()

def preparer_donnees(dataset):
    y = dataset.iloc[:, 0].values

    X = dataset.iloc[:, 1:].fillna("missing").values 

    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X).toarray()
    print(f"Dimensions après encodage : {X_encoded.shape}")
    return X_encoded, y, encoder

def entrainer_modele(X_train, y_train):
    arbre = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
    arbre.fit(X_train, y_train)
    return arbre

def sauvegarder_modele(modele, encoder, maladies, chemin_fichier):
    model_data = {
        'modele': modele,
        'encoder': encoder,
        'maladies': maladies
    }
    try:
        os.makedirs(os.path.dirname(chemin_fichier), exist_ok=True)
        joblib.dump(model_data, chemin_fichier)
        print(f"Modèle sauvegardé avec succès dans '{chemin_fichier}'")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle : {e}")
        exit()

def main():
    fichier_csv = 'Data/dataset.csv'
    fichier_modele = 'Models/modele_ml.joblib'

    dataset = charger_dataset(fichier_csv)

    print("Aperçu des données :")
    print(dataset.head())

    X, y, encoder = preparer_donnees(dataset)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Taille de l'ensemble d'apprentissage : ", X_train.shape)
    print("Taille de l'ensemble de test : ", X_test.shape)

    print("Entraînement du modèle...")
    arbre = entrainer_modele(X_train, y_train)

    y_pred = arbre.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"Précision de l'arbre : {score:.2f}")

    sauvegarder_modele(arbre, encoder, list(set(y)), fichier_modele)

if __name__ == "__main__":
    main()
