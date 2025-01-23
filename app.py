from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

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

clf, le, cols, clf2, le2, cols_clf2 = charger_modeles()

# Route principale
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        symptomes = request.form.get("symptomes").lower().split(",")

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
                "probabilites": sorted(zip(le.classes_, probabilities), key=lambda x: x[1], reverse=True)[:5],
            }
        except Exception as e:
            result = {"error": f"Erreur : {e}"}

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
