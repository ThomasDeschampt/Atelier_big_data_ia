import pyttsx3
import threading

def afficher_parler(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)

    engine.say(text)
    print(text)
    engine.runAndWait()
    engine.stop()

def lancer_parler_en_thread(text):
    thread = threading.Thread(target=afficher_parler, args=(text,))
    thread.start()