from model import detect
import pyttsx3

import datetime
import speech_recognition as sr

from faceoriginal import only_faces

import requests
import  wikipedia





def takeCommand2():

    
    query=None
    while True:
        print("Take command 2...")
        #It takes microphone input from the user and returns string output
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Say something 2!")
            # render_speak("Say something")
            r.pause_threshold = 1


            r.adjust_for_ambient_noise(source)  # listen for 1 second to calibrate the energy threshold for ambient noise levels

            audio = r.listen(source)



        
        # r = sr.Recognizer()
        # with sr.Microphone() as source:
        #     print("Listening...")
        #     r.pause_threshold = 1
        #     audio = r.listen(source)

        try:
            print("Recognizing...") 
            # render_speak("Recognizing...")   
            query = r.recognize_google(audio, language='en-in')
            print(f"You said: {query}\n")
            # render_speak(f"you said {query}")
            if "activate God of rome" in query:
                break

        except Exception as e:
            # print(e)   
            # render_speak("Say that again please")
            print("Say that again please...")  
            return "None"
    return query