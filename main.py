import re
import joblib


def charger_dictionnaire(fichier_dictionnaire):
    with open(fichier_dictionnaire, 'r', encoding='utf-8') as f:
        return set(mot.strip().lower() for mot in f.readlines())


def extraire_mots_cles(texte, dictionnaire):
    mots = re.findall(r'\b\w+\b', texte.lower())
    return [mot for mot in mots if mot in dictionnaire]


def predire_maladie(symptomes, modele):
    vecteur = [1 if symptome in symptomes else 0 for symptome in modele['symptomes']]
    prediction = modele['modele'].predict([vecteur])[0]
    return modele['maladies'][prediction]


def main():
    print("Bienvenue dans le chatbot médical AI !")
    dictionnaire_fichier = "Data/dataset.csv"
    modele_fichier = "Models/modele_ml.joblib"

    dictionnaire = charger_dictionnaire(dictionnaire_fichier)
    modele = joblib.load(modele_fichier)

    while True:
        print("\nVeuillez décrire vos symptômes (ou tapez 'quitter' pour sortir) :")
        entree_utilisateur = input(">>> ").strip()

        if entree_utilisateur.lower() == 'quitter':
            print("Merci d'avoir utilisé le chatbot médical. Prenez soin de vous !")
            break

        mots_cles = extraire_mots_cles(entree_utilisateur, dictionnaire)

        if not mots_cles:
            print("\nAucun symptôme reconnu dans votre requête. Veuillez réessayer.")
            continue

        maladie = predire_maladie(mots_cles, modele)

        print(f"\nMaladie probable : {maladie}")
        print("Prenez les précautions suivantes et consultez un spécialiste si nécessaire.")

if __name__ == "__main__":
    main()
