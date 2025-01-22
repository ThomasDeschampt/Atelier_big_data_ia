import joblib
import pandas as pd

def charger_dictionnaires():
    print("Chargement des descriptions et précautions...")
    descriptions = pd.read_csv('Data/symptom_Description.csv')
    precautions = pd.read_csv('Data/symptom_precaution.csv')

    desc_dict = dict(zip(descriptions['Symptom'], descriptions['Description']))
    precaution_dict = dict(zip(precautions['Disease'], precautions.iloc[:, 1:].values.tolist()))

    return desc_dict, precaution_dict

def afficher_description(symptomes, desc_dict):
    for symptome in symptomes:
        description = desc_dict.get(symptome, "Aucune description disponible.")
        print(f"- {symptome}: {description}")

def afficher_precautions(maladie, precaution_dict):
    precautions = precaution_dict.get(maladie, ["Aucune précaution disponible."])
    print(f"Précautions pour {maladie}:")
    for precaution in precautions:
        print(f"- {precaution}")

def main():
    print("Bienvenue dans le chatbot médical AI !")

    fichier_modele = 'Models/modele_ml.joblib'
    modele = joblib.load(fichier_modele)
    clf = modele['modele']
    le = modele['encoder']
    cols = modele['colonnes']

    desc_dict, precaution_dict = charger_dictionnaires()

    while True:
        print("\nEntrez vos symptômes séparés par des virgules (ou tapez 'quitter') :")
        symptomes = input(">>> ").strip().split(',')

        if 'quitter' in symptomes:
            print("Merci d'avoir utilisé le chatbot médical. Prenez soin de vous !")
            break

        afficher_description(symptomes, desc_dict)

        input_data = [1 if col in symptomes else 0 for col in cols]
        prediction = clf.predict([input_data])
        maladie_predite = le.inverse_transform(prediction)[0]

        print(f"\nMaladie probable : {maladie_predite}")
        afficher_precautions(maladie_predite, precaution_dict)

if __name__ == "__main__":
    main()
