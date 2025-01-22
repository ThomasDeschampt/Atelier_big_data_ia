import joblib
import os
import subprocess
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from vocal import afficher_parler

# Charger les dictionnaires de descriptions et précautions
def charger_dictionnaires():
    print("Chargement des descriptions et précautions...")
    try:
        descriptions = pd.read_csv('Data/symptom_Description.csv')
        precautions = pd.read_csv('Data/symptom_precaution.csv')
    except FileNotFoundError as e:
        print(f"Erreur : {e}")
        exit()
    
    descriptions.columns = descriptions.columns.str.strip()
    precautions.columns = precautions.columns.str.strip()

    desc_dict = dict(zip(descriptions.iloc[:, 0], descriptions.iloc[:, 1]))
    precaution_dict = dict(zip(precautions.iloc[:, 0], precautions.iloc[:, 1:].values.tolist()))

    return desc_dict, precaution_dict

def verifier_modele():
    fichier_modele = 'Models/modele_ml.joblib'
    if not os.path.exists(fichier_modele):
        print(f"Le modèle '{fichier_modele}' est introuvable. Entraînement du modèle...")
        subprocess.run(['python', 'apprentissage.py'])
    else:
        print(f"Le modèle '{fichier_modele}' est disponible.")

def afficher_description(maladie, desc_dict):
    if maladie in desc_dict:
        return f"La description de la maladie {maladie} : {desc_dict[maladie]}"
    else:
        return f"Aucune description disponible pour la maladie {maladie}."

def afficher_precautions(maladie, precaution_dict):
    precautions = precaution_dict.get(maladie, ["Aucune précaution disponible."])
    precautions_str = "\n".join([f"- {precaution}" for precaution in precautions])
    return f"\nPrécautions pour {maladie} :\n{precautions_str}"

def verifier_symptomes(symptomes):
    list_symptomes = ['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
                      'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_urination',
                      'fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy',
                      'patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion',
                      'headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain',
                      'diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
                      'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure',
                      'runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region',
                      'bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels',
                      'puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts',
                      'drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
                      'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell',
                      'bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look',
                      'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation',
                      'dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum',
                      'lack_of_concentration','visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections','coma',
                      'stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum',
                      'prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
                      'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze',
                      'prognosis']
        
    for symptome in symptomes:
        if symptome not in list_symptomes:
            afficher_parler(f"Symptôme inconnu : {symptome}. Veuillez entrer un symptôme valide.")
            return False
    return True

def main_gui():
    # Initialisation de la fenêtre Tkinter
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

    # Fonction de prédiction
    def predict_disease():
        symptomes = entry_symptomes.get().strip().split(',')
        if not verifier_symptomes(symptomes):
            return
        
        input_data = [1 if col.strip() in symptomes else 0 for col in cols]
        probabilities = clf.predict_proba([input_data])[0]
        prediction = clf.predict([input_data])
        maladie_predite = le.inverse_transform(prediction)[0]

        result = f"Maladie probable : {maladie_predite}\n"
        for maladie, prob in zip(le.classes_, probabilities):
            if prob > 0:
                proba = prob * 100
                result += f"Probabilité de {maladie} : {prob:.2f} ({proba:.2f}%)\n"

        # Afficher description et précautions
        result += afficher_description(maladie_predite, desc_dict)
        result += afficher_precautions(maladie_predite, precaution_dict)

        # Afficher les résultats dans la fenêtre
        text_result.delete(1.0, tk.END)
        text_result.insert(tk.END, result)

    # Création des widgets Tkinter
    label = tk.Label(root, text="Entrez vos symptômes séparés par des virgules :")
    label.pack(pady=10)

    entry_symptomes = tk.Entry(root, width=50)
    entry_symptomes.pack(pady=10)

    button_predict = tk.Button(root, text="Prédire la maladie", command=predict_disease)
    button_predict.pack(pady=10)

    text_result = tk.Text(root, width=60, height=15)
    text_result.pack(pady=10)

    button_quitter = tk.Button(root, text="Quitter", command=root.quit)
    button_quitter.pack(pady=10)

    # Lancer la boucle Tkinter
    root.mainloop()

if __name__ == "__main__":
    main_gui()
