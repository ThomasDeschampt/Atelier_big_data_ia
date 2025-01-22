import joblib
import os
import subprocess
import pandas as pd
import tkinter as tk
from tkinter import messagebox

# Charger les dictionnaires de descriptions et précautions
def charger_dictionnaires():
    print("Chargement des descriptions et précautions...")
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

# Vérifier et entraîner le modèle si nécessaire
def verifier_modele():
    fichier_modele = 'Models/modele_ml.joblib'
    if not os.path.exists(fichier_modele):
        print(f"Le modèle '{fichier_modele}' est introuvable. Entraînement du modèle...")
        subprocess.run(['python', 'apprentissage.py'])
    else:
        print(f"Le modèle '{fichier_modele}' est disponible.")

# Vérifier si les symptômes sont valides
def verifier_symptomes(symptomes, cols):
    symptomes = [s.strip().lower() for s in symptomes]
    symptomes_invalides = [s for s in symptomes if s not in cols]

    if symptomes_invalides:
        messagebox.showerror("Erreur", f"Symptômes inconnus : {', '.join(symptomes_invalides)}.\nVeuillez entrer des symptômes valides.")
        return False
    return True

# Afficher la description d'une maladie
def afficher_description(maladie, desc_dict):
    if maladie in desc_dict:
        return f"\nDescription de la maladie {maladie} : {desc_dict[maladie]}\n"
    else:
        return f"\nAucune description disponible pour la maladie {maladie}.\n"

# Afficher les précautions pour une maladie
def afficher_precautions(maladie, precaution_dict):
    precautions = precaution_dict.get(maladie, ["Aucune précaution disponible."])
    precautions_str = "\n".join([f"- {precaution}" for precaution in precautions])
    return f"\nPrécautions pour {maladie} :\n{precautions_str}"

# Interface principale avec Tkinter
def main_gui():
    root = tk.Tk()
    root.title("Chatbot Médical AI")

    # Charger le modèle et les dictionnaires
    verifier_modele()
    fichier_modele = 'Models/modele_ml.joblib'
    modele = joblib.load(fichier_modele)
    clf = modele['modele']
    le = modele['encoder']
    cols = modele['colonnes']
    desc_dict, precaution_dict = charger_dictionnaires()

    # Fonction de prédiction (appelée lors de l'appui sur Entrée)
    def predict_disease(event=None):
        symptomes = entry_symptomes.get().strip().split(',')
        symptomes = [s.strip().lower() for s in symptomes]

        if not verifier_symptomes(symptomes, cols):
            text_result.delete(1.0, tk.END)
            text_result.insert(tk.END, "Symptômes invalides. Veuillez vérifier votre saisie.")
            return

        # Préparation des données d'entrée avec les noms de colonnes alignés
        input_data = [1 if col in symptomes else 0 for col in cols]
        input_data = pd.DataFrame([input_data], columns=cols)  # Créer un DataFrame avec les noms de colonnes
        
        # Prédiction et probabilités
        probabilities = clf.predict_proba(input_data)[0]
        prediction = clf.predict(input_data)
        maladie_predite = le.inverse_transform(prediction)[0]

        result = f"Maladie probable : {maladie_predite}\n\n"
        result += "Autres maladies possibles avec leurs probabilités :\n"
        for maladie, prob in zip(le.classes_, probabilities):
            if prob > 0:  # Seulement les maladies avec une probabilité > 10%
                result += f"- {maladie} : {prob * 100:.2f}%\n"

        # Ajouter description et précautions
        result += afficher_description(maladie_predite, desc_dict)
        result += afficher_precautions(maladie_predite, precaution_dict)

        # Afficher le résultat dans l'interface
        text_result.delete(1.0, tk.END)
        text_result.insert(tk.END, result)

    # Widgets Tkinter
    label = tk.Label(root, text="Entrez vos symptômes séparés par des virgules, puis appuyez sur Entrée :")
    label.pack(pady=10)

    entry_symptomes = tk.Entry(root, width=50)
    entry_symptomes.pack(pady=10)
    entry_symptomes.bind('<Return>', predict_disease)

    text_result = tk.Text(root, width=60, height=20)
    text_result.pack(pady=10)

    button_quitter = tk.Button(root, text="Quitter", command=root.quit)
    button_quitter.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main_gui()
