print("init")

import numpy as np 
import pandas as pd
import pyttsx3 
import csv

def lire_et_parler(texte):
    moteur = pyttsx3.init()
    moteur.setProperty('voice', "french")
    moteur.setProperty('rate', 130)
    moteur.say(texte)
    moteur.runAndWait()
    moteur.stop()

def diagnostiquer():
    print("Bienvenue dans le ChatBot de soins de santé !")
    nom = input("Quel est votre nom ? -> ")
    print(f"Bonjour, {nom}! Diagnostiquons vos symptômes.")

    symptomes_presentes = []
    while True:
        symptome = input