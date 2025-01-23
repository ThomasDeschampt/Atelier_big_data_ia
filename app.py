from flask import Flask, render_template, request, flash
import joblib
import pandas as pd
import os
from vocal import lancer_parler_en_thread

app = Flask(__name__)
app.secret_key = 'votre_cle_secrète'  # Nécessaire pour les messages flash

# Charger le modèle et les dictionnaires
def charger_modeles():
    modele_path = "Models/modele_ml.joblib"
    modele2_path = "Models/modele_mll.joblib"

    modele = joblib.load(modele_path)
    modele2 = joblib.load(modele2_path)

    # Charger les colonnes et encodeurs
    clf, le, cols = modele['modele'], modele['encoder'], modele['colonnes']
    clf2, le2, cols_clf2 = modele2['modele'], modele2['encoder'], modele2['colonnes']
    return clf, le, cols, clf2, le2, cols_clf2

# Charger les dictionnaires de descriptions et précautions
def charger_dictionnaires():
    try:
        descriptions = pd.read_csv('Data/symptom_Description.csv')
        precautions = pd.read_csv('Data/symptom_precaution.csv')
    except FileNotFoundError as e:
        print(f"Erreur : {e}")
        exit()

    # Nettoyage des colonnes
    descriptions.columns = descriptions.columns.str.strip().str.lower()
    precautions.columns = precautions.columns.str.strip().str.lower()

    # Création des dictionnaires
    desc_dict = dict(zip(descriptions.iloc[:, 0], descriptions.iloc[:, 1]))
    precaution_dict = dict(zip(precautions.iloc[:, 0], precautions.iloc[:, 1:].values.tolist()))

    return desc_dict, precaution_dict

clf, le, cols, clf2, le2, cols_clf2 = charger_modeles()
desc_dict, precaution_dict = charger_dictionnaires()

# Vérifier si les symptômes sont valides
def verifier_symptomes(symptomes, cols):
    symptomes = [s.strip().lower() for s in symptomes]
    symptomes_invalides = [s for s in symptomes if s not in cols]
    return symptomes_invalides

# Afficher la description d'une maladie
def afficher_description(maladie):
    if maladie in desc_dict:
        return f"\nDescription de la maladie {maladie} : {desc_dict[maladie]}\n"
    else:
        return f"\nAucune description disponible pour la maladie {maladie}.\n"

# Afficher les précautions pour une maladie
def afficher_precautions(maladie):
    precautions = precaution_dict.get(maladie, ["Aucune précaution disponible."])
    return precautions

# Route principale
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        symptomes = request.form.get("symptomes").lower().split(",")
        symptomes = [s.strip() for s in symptomes]  # Supprimer les espaces

        # Vérification des symptômes
        symptomes_invalides = verifier_symptomes(symptomes, cols)
        if symptomes_invalides:
            flash(f"Symptômes inconnus : {', '.join(symptomes_invalides)}. Veuillez entrer des symptômes valides.", "error")
            return render_template("index.html", result=result)

        # Préparer les données
        input_data = [1 if col in symptomes else 0 for col in cols]
        input_df = pd.DataFrame([input_data], columns=cols)

        try:
            maladie_predite = le.inverse_transform(clf.predict(input_df))[0]
            probabilities = clf.predict_proba(input_df)[0]

            # Recommandation docteur
            input_data_clf2 = [1 if col == maladie_predite else 0 for col in cols_clf2]
            input_df_clf2 = pd.DataFrame([input_data_clf2], columns=cols_clf2)
            docteur = le2.inverse_transform(clf2.predict(input_df_clf2))[0]

            # Résultats
            result = {
                "maladie": maladie_predite,
                "docteur": docteur,
                "probabilites": sorted(zip(le.classes_, probabilities * 100), key=lambda x: x[1], reverse=True)[:5],
            }
            # Ajouter la description et les précautions
            result["description"] = afficher_description(maladie_predite)
            result["precautions"] = afficher_precautions(maladie_predite)
            
            lancer_parler_en_thread(result)
        except Exception as e:
            flash(f"Erreur lors de la prédiction : {e}", "error")
            result = {"error": f"Erreur : {e}"}

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
