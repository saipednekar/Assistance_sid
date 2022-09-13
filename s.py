from vosk import Model,KaldiRecognizer

import pyaudio
import pyttsx3
engine = pyttsx3.init()


model=Model("/home/sai/assistantAI/vosk-model-small-en-in-0.4")


recognizer= KaldiRecognizer(model,16000)


mic=pyaudio.PyAudio()

stream= mic.open(format=pyaudio.paInt16,channels=1,rate=16000,input=True,frames_per_buffer=8192)

stream.start_stream()


while True:
    print("listening")
    data= stream.read(4096)
    if recognizer.AcceptWaveform(data):
        text=recognizer.Result()
        
        print(text)
        engine.say(text)
        engine.runAndWait()