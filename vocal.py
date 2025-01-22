import pyttsx3




def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)

    engine.say("Hello World!")
    engine.runAndWait()
    engine.stop()

speak("Hello World!")