

import pyttsx3
import random
import datetime
import speech_recognition as sr

from weather import w

from faceoriginal import only_faces

from model import detect
import requests
import  wikipedia
import pygame
from pygame import mixer
from background import takeCommand2

from tracking import only_tracking

pygame.init()

engine = pyttsx3.init()
screen = pygame.display.set_mode([1000, 1000])
pygame.display.set_caption("GOD OF ROM SID 👾")
screen.fill((255,255,255))

font=pygame.font.Font("/home/sai/assistantAI/font/font.TTF",40)

pygame.display.update()




def speak(audio):
    engine.setProperty('rate', 125)     # setting up new voice rate


    engine.say(audio) 

    engine.runAndWait() 




def render_speak(text):
    global i
    global x 
    global y

    global s
    global speed
    


    
    screen.fill((0, 0, 0))
    # screen.blit(background,(0,0))
    a1=pygame.image.load("/home/sai/assistantAI/G_O_R/1.jpg")
    a2=pygame.image.load("/home/sai/assistantAI/G_O_R/2.jpg")
    a3=pygame.image.load("/home/sai/assistantAI/G_O_R/3.jpg")
    a4=pygame.image.load("/home/sai/assistantAI/G_O_R/4.jpg")

    emoji=pygame.image.load("/home/sai/assistantAI/emoji.png")




    a1= pygame.transform.scale(a1, (200,200))
    a2= pygame.transform.scale(a2, (200,200))
    a3= pygame.transform.scale(a3, (200,200))
    a4= pygame.transform.scale(a4, (200,200))


    emoji= pygame.transform.scale(emoji, (s,s))
    screen.blit(emoji, (x,y))
    
    
    if x>950:
        speed=-10
        s=10
    elif x<20:
        speed=10
        s=10
    x+=speed
    s+=5
    print(s)

    


    sidd=[a1,a2,a3,a4]


    if i < len(sidd):
                
        screen.blit(sidd[i], (400,300))
        
        i+=1
            # print(i)
        
    elif i ==len(sidd):

        screen.blit(sidd[1], (400,300))
            

        i=0 
    
        

    if len(text)>30:
        font=pygame.font.Font("/home/sai/assistantAI/font/font.TTF",28)

        t=font.render(text,True,(0,255,255))
        screen.blit(t,(10,600))
        pygame.display.update()
        speak(text)
        

    else:
        font=pygame.font.Font("/home/sai/assistantAI/font/font.TTF",40)

        t=font.render(text,True,(0,255,255))
        screen.blit(t,(20,600))
        pygame.display.update()
        speak(text)

def render_speak_list(list):
    global i
    global x 
    global y
    
    global s
    global speed

    screen.fill((0, 0, 0))
    # screen.blit(background,(0,0))
    a1=pygame.image.load("/home/sai/assistantAI/G_O_R/1.jpg")
    a2=pygame.image.load("/home/sai/assistantAI/G_O_R/2.jpg")
    a3=pygame.image.load("/home/sai/assistantAI/G_O_R/3.jpg")
    a4=pygame.image.load("/home/sai/assistantAI/G_O_R/4.jpg")


    emoji=pygame.image.load("/home/sai/assistantAI/emoji.png")


    a1= pygame.transform.scale(a1, (200,200))
    a2= pygame.transform.scale(a2, (200,200))
    a3= pygame.transform.scale(a3, (200,200))
    a4= pygame.transform.scale(a4, (200,200))

    emoji= pygame.transform.scale(emoji, (s,s))
    screen.blit(emoji, (x,y))
    
    
    if x>950:
        speed=-10
        s=10
    elif x<20:
        speed=10
        s=10
    x+=speed
    s+=5
    print(s)


    sidd=[a1,a2,a3,a4]


    if i < len(sidd):
                
        screen.blit(sidd[i], (400,300))
        
        i+=1
            # print(i)
        
    elif i ==len(sidd):

        screen.blit(sidd[0], (400,300))
            

        i=0 
    
    if list == object_detected:
        if len(list) == 1:
            text= f"the possible detected object in the room is {[i for i in list]}"


        else:    
            text= f"the possible detected objects in the room are {[i for i in list]}"

        



    elif list == person_detected:
        if len(list) == 1:
            text= f"the possible detected person in the room is {[i for i in list]}"


        else:    
            text= f"the possible detected people in the room are {[i for i in list]}"
    




    elif list == items_detected: 
        if len(list) == 1:
            text= f"the possible detected items in the room is {[i for i in list]}"


        else:    
            text= f"the possible detected items in the room are {[i for i in list]}"
        
    


    font=pygame.font.Font("/home/sai/assistantAI/font/font.TTF",20)

    t=font.render(text,True,(0,255,255))
    screen.blit(t,(10,600))
    pygame.display.update()
    speak(text)



def wishme(info):
    text=f"Todays weather is {info}"
    hour = int(datetime.datetime.now().hour)

    if hour>=0 and hour<12:

        screen.fill((255, 255, 255))
        # screen.blit(background,(0,0))


        mt=font.render("Good Morning",True,(255,60,60))
        screen.blit(mt,(20,600))
        pygame.display.update()
        speak("Good Morning")


    elif hour>=12 and hour<18:

        screen.fill((255, 255, 255))

        # screen.blit(background,(0,0))


        mt=font.render("Good Afternoon",True,(255,60,60))
        screen.blit(mt,(20,299))
        pygame.display.update()  
        speak("Good Afternoon") 


    else:

        screen.fill((255, 255, 255))

        # screen.blit(background,(0,0))


        mt=font.render("Good Evening!",True,(255,60,60))
        screen.blit(mt,(20,299))
        pygame.display.update()

        speak("Good Evening!")  

    
    screen.fill((255, 255, 255))
    # screen.blit(background,(0,0))


    mt=font.render("I am GOD OF ROM Sir  Please tell me how may I help you",True,(255,60,60))
    screen.blit(mt,(20,299))

    mt2=font.render(text,True,(255,60,60))
    screen.blit(mt2,(20,400))
    pygame.display.update()
    speak("I am GOD OF ROM Sir. Please tell me how may I help you")    
    speak(text)   



def takeCommand():
    #It takes microphone input from the user and returns string output
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        render_speak("Say something")
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
        render_speak("Recognizing...")   
        query = r.recognize_google(audio, language='en-in')
        print(f"You said: {query}\n")
        render_speak(f"you said {query}")

    except Exception as e:
        # print(e)   
        render_speak("Say that again please")
        print("Say that again please...")  
        return "None"
    return query













def del_duplicates(list):
    new_list=[]
    for i in list:
        if i not in new_list:
            new_list.append(i)
    return new_list        

# def speak_detected_items(list):
        
    # if list == object_detected:
    #     if len(list) == 1:
    #         text= f"the possible detected object in the room is {[i for i in list]}"
    #         speak(text)


    #     else:    
    #         text= f"the possible detected objects in the room are {[i for i in list]}"
    #         speak(text)

        



    # elif list == person_detected:
    #     if len(list) == 1:
    #         text= f"the possible detected person in the room is {[i for i in list]}"
    #         speak(text)


    #     else:    
    #         text= f"the possible detected people in the room are {[i for i in list]}"
    #         speak(text)
    




    # elif list == items_detected: 
    #     if len(list) == 1:
    #         text= f"the possible detected items in the room is {[i for i in list]}"
    #         speak(text)


    #     else:    
    #         text= f"the possible detected items in the room are {[i for i in list]}"
    #         speak(text)
        



def joke():
    response = requests.get("https://api.chucknorris.io/jokes/random")
    if response.status_code == 200:
        print("sucessfully fetched the data")
        print(response.json())
        dictt=response.json()
        print(dictt["value"])
        render_speak(dictt["value"])
    else:
        
        print(f"Hello person, there's a {response.status_code} error with your request")
        render_speak(f"Hello person, there's a {response.status_code} error with your request")
i=0  
info=w()

wishme(info)
activate=True
loop =True
x=random.randint(30,500)
y=random.randint(30,700)



speed=10
s=30
while  activate:


    if loop:
        print(i)
        #query for voice recognition
        query=takeCommand()
        if "detect " in query:
            run=detect()
            results=run.detector()    
            print(f"results are{results}") 
            object_detected=results[0]

            person_detected=results[1]

            items_detected=results[2]

            object_detected=del_duplicates(object_detected)

            person_detected=del_duplicates(person_detected)

            items_detected=del_duplicates(items_detected)

            print(object_detected)

            print(person_detected)

            print(items_detected)

            render_speak_list(object_detected)

            render_speak_list(person_detected)

            render_speak_list(items_detected)
        elif "search" in query:
            try:
                ans=wikipedia.summary(query, sentences=1)
                render_speak(ans)
            except Exception as e:
                print(e)
                render_speak(str(e))

        elif "joke" in query:
            joke()
        elif "face" in query:    
            f=only_faces()
            real_people=f.d()
            person_detected=del_duplicates(real_people)
            print(person_detected)
            object_detected=None
            items_detected=None
            render_speak_list(person_detected)


        elif "stop" in query:

            activate=False

        # elif "keep quiet" or "pause" in query:
            
        #     takeCommand2()


        elif "tracking detector" in query:

            t=only_tracking()
            t.d()

        




                

                

