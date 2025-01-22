import pyttsx3

def afficher_parler(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    voices = engine.getProperty('voices')

    engine.say(text)
    print(text)
    engine.runAndWait()
    engine.stop()