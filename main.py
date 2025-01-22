import joblib
import os
import subprocess
import pandas as pd

from vocal import afficher_parler

# Charger les dictionnaires de descriptions et précautions
def charger_dictionnaires():
    print("Chargement des descriptions et précautions...")

    # On charge les fichiers csv
    try:
        descriptions = pd.read_csv('Data/symptom_Description.csv')
        precautions = pd.read_csv('Data/symptom_precaution.csv')
    except FileNotFoundError as e:
        print(f"Erreur : {e}")
        exit()

    # On nettoie les données
    descriptions.columns = descriptions.columns.str.strip()
    precautions.columns = precautions.columns.str.strip()

    # On crée les dictionnaires
    desc_dict = dict(zip(descriptions.iloc[:, 0], descriptions.iloc[:, 1]))
    precaution_dict = dict(zip(precautions.iloc[:, 0], precautions.iloc[:, 1:].values.tolist()))

    # On retourne les dictionnaires
    return desc_dict, precaution_dict

# Vérifier si le modèle est disponible
def verifier_modele():
    fichier_modele = 'Models/modele_ml.joblib'

    # Si le modèle n'existe pas, on entraîne un nouveau modèle
    if not os.path.exists(fichier_modele):
        print(f"Le modèle '{fichier_modele}' est introuvable. Entraînement du modèle...")
        subprocess.run(['python', 'apprentissage.py'])  
    else:
        print(f"Le modèle '{fichier_modele}' est disponible.")

# Afficher les descritpions des symptômes
def afficher_description(symptomes, desc_dict):
    print("\nDescriptions des symptômes :")

    # On affiche la description de chaque symptôme
    for symptome in symptomes:
        description = desc_dict.get(symptome.strip(), "Aucune description disponible.")
        print(f"- {symptome}: {description}")

# Afficher les précautions pour une maladie
def afficher_precautions(maladie, precaution_dict):
    print(f"\nPrécautions pour {maladie} :")

    # On affiche les précautions pour la maladie
    precautions = precaution_dict.get(maladie, ["Aucune précaution disponible."])
    for precaution in precautions:
        print(f"- {precaution}")

# Vérifier si les symptômes sont valides
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

#Fonction principale
def main():
    afficher_parler("Bienvenue dans le chatbot médical AI !")

    verifier_modele()

    # Récupérer le modèle
    fichier_modele = 'Models/modele_ml.joblib'
    modele = joblib.load(fichier_modele)
    clf = modele['modele']
    le = modele['encoder']
    cols = modele['colonnes']

    desc_dict, precaution_dict = charger_dictionnaires()

    # Boucle principale
    while True:
        # On demande à l'utilisateur d'entrer ses symptômes ou de quitter l'app
        afficher_parler("\nEntrez vos symptômes séparés par des virgules (ou tapez 'quitter') :")
        symptomes = input(">>> ").strip().split(',')

        # On vérifie si les symptômes sont valides
        if not verifier_symptomes(symptomes):
            continue


        # On vérifie si l'utilisateur veut quitter
        if 'quitter' in symptomes:
            afficher_parler("Merci d'avoir utilisé le chatbot médical. Prenez soin de vous !")
            break

        # On affiche les descriptions des symptômes
        afficher_description(symptomes, desc_dict)

        # On prédit la maladie probable et on affiche les précautions
        # On prédit la maladie probable et on affiche les précautions
        input_data = [1 if col.strip() in symptomes else 0 for col in cols]
        probabilities = clf.predict_proba([input_data])[0]  # Récupérer les probabilités
        prediction = clf.predict([input_data])
        maladie_predite = le.inverse_transform(prediction)[0]

        # Afficher toutes les probabilités supérieures à 0
        for maladie, prob in zip(le.classes_, probabilities):
            if prob > 0:
                # On met la proba en pourcentage
                proba = prob * 100
                print(f"Probabilité de {maladie} : {prob:.2f} ({proba:.2f}%)")

        afficher_parler(f"\nMaladie probable : {maladie_predite}")
        afficher_precautions(maladie_predite, precaution_dict)


if __name__ == "__main__":
    main()
