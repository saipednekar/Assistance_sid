import tensorflow_hub as hub
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pyttsx3

import face_recognition

# Import the required libraries

# Create an instance of tkinter frame or window



class only_tracking:
    def d():
        print("inside")
        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2


        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')






        font = cv2.FONT_HERSHEY_SIMPLEX

        fontt = cv2.FONT_HERSHEY_SIMPLEX



        cap = cv2.VideoCapture(0)
        # address="http://192.168.0.112:4747/video"
        # cap.open(address)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

        width = 260
        height = 260

        frame_rate = 10
        prev = 0

        locateface=False

        name=None

        #face recognitiion

        sai_image = face_recognition.load_image_file("/home/sai/assistantAI/faces/sai.jpg")
        sai_face_encoding = face_recognition.face_encodings(sai_image)[0]
        print("sai",sai_face_encoding)

        
        d_image = face_recognition.load_image_file("/home/sai/assistantAI/faces/dinesh.jpg")
        d_face_encoding = face_recognition.face_encodings(d_image)[0]
        print("dinesh face encoding",d_face_encoding)


        v_image = face_recognition.load_image_file("/home/sai/assistantAI/faces/vanita.jpg")
        v_face_encoding = face_recognition.face_encodings(v_image)[0]
        print("vanita face encoding",v_face_encoding)
        


        su_image = face_recognition.load_image_file("/home/sai/assistantAI/faces/sudha.jpg")
        su_face_encoding = face_recognition.face_encodings(su_image)[0]
        print("sudha face encoding",su_face_encoding)

        sid_image = face_recognition.load_image_file("/home/sai/assistantAI/faces/siddesh.jpg")
        sid_face_encoding = face_recognition.face_encodings(sid_image)[0]
        print("sid face encoding",sid_face_encoding)

        datta_image = face_recognition.load_image_file("/home/sai/assistantAI/faces/datta.jpg")
        datta_face_encoding = face_recognition.face_encodings(datta_image)[0]
        print("datta face encoding",datta_face_encoding)

        a_image = face_recognition.load_image_file("/home/sai/assistantAI/faces/aditya.jpg")
        a_face_encoding = face_recognition.face_encodings(a_image)[0]
        print("aditya face encoding",a_face_encoding)

        deep_image = face_recognition.load_image_file("/home/sai/assistantAI/faces/deepa.jpg")
        deep_face_encoding = face_recognition.face_encodings(deep_image)[0]
        print("deepa face encoding",deep_face_encoding)


        m_image = face_recognition.load_image_file("/home/sai/assistantAI/faces/mahesh.jpg")
        m_face_encoding = face_recognition.face_encodings(m_image)[0]
        print("mahesh face encoding",m_face_encoding)


        v_image = face_recognition.load_image_file("/home/sai/assistantAI/faces/vaibhav.jpg")
        v_face_encoding = face_recognition.face_encodings(v_image)[0]
        print("vaibhav face encoding",v_face_encoding)


        g_image = face_recognition.load_image_file("/home/sai/assistantAI/faces/geeta.jpg")
        g_face_encoding = face_recognition.face_encodings(g_image)[0]
        print("geeta face encoding",g_face_encoding)



        n_image = face_recognition.load_image_file("/home/sai/assistantAI/faces/naresh.jpg")
        n_face_encoding = face_recognition.face_encodings(n_image)[0]
        print("naresh face encoding",n_face_encoding)


        sah_image = face_recognition.load_image_file("/home/sai/assistantAI/faces/sahil.jpg")
        sah_face_encoding = face_recognition.face_encodings(sah_image)[0]
        print("sahil face encoding",sah_face_encoding)


        soh_image = face_recognition.load_image_file("/home/sai/assistantAI/faces/sohan.jpg")
        soh_face_encoding = face_recognition.face_encodings(soh_image)[0]
        print("sohan face encoding",soh_face_encoding)


        sun_image = face_recognition.load_image_file("/home/sai/assistantAI/faces/sunita.jpg")
        sun_face_encoding = face_recognition.face_encodings(sun_image)[0]
        print("sunita face encoding",sun_face_encoding)











    




        known_face_encodings = [
            sai_face_encoding,
            d_face_encoding,
            v_face_encoding,
            su_face_encoding,
            sid_face_encoding,
            datta_face_encoding,
            a_face_encoding,
            deep_face_encoding,
            m_face_encoding,
            v_face_encoding,
            g_face_encoding,
            n_face_encoding,
            sah_face_encoding,
            soh_face_encoding,
            sun_face_encoding
        ]

        known_face_names = [
            "Sai",
            "Dinesh",
            "vanita",
            "sudha",
            "siddesh",
            "datta",
            "aditya",
            "deepa",
            "mahesh",
            "vaibhav",
            "geetan",
            "naresh",
            "sahil",
            "sohan",
            "sunita"


            
        ]

    
        person_name=[]
        a=0
        
        x_cor=[]

        n=0
        i=0
        engine = pyttsx3.init()

        def speak(audio):
            engine.setProperty('rate', 125)     # setting up new voice rate


            engine.say(audio) 

            engine.runAndWait() 
        while (True):
            
            time_elapsed = time.time() - prev
            
            ret, frame = cap.read()
            
            if time_elapsed > 1./frame_rate:
                prev = time.time()
            
            #Resize to respect the input_shape
                inp = cv2.resize(frame, (width , height ))

            #Convert img to RGB
                rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        #         small_frame = cv2.resize(rgb, (0, 0), fx=0.25, fy=0.25)

                
                face_locations = face_recognition.face_locations(rgb)
                current_face_encoding = face_recognition.face_encodings(rgb)
                print("currentface_encoding",current_face_encoding)

            
                if face_locations:
                        print("facelocations",face_locations)
                        print(len(face_locations[0]))


                        print (face_locations[0])



                        top=face_locations[0][0]

                        right=face_locations[0][1]

                        bottom=face_locations[0][2]
                        left=face_locations[0][3]

                        locateface = True
                        
                        matches=face_recognition.compare_faces(known_face_encodings, current_face_encoding[0])
                        print("matches",matches)
                        face_distances = face_recognition.face_distance(known_face_encodings, current_face_encoding[0])
                        print("distance",face_distances)
                        
                        best_match_index = np.argmin(face_distances)
                        if matches:
                            name = known_face_names[best_match_index]

                            
                        

                
                else:

                        locateface = False





            
            
            
            

                

                if locateface:
                    img_boxes =cv2.rectangle(rgb, (left, top), (right, bottom), color, 2)
                    cv2.putText(img_boxes, str(name), (left, top-10), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA )
                    print( str(name))
                    x_cor.append(left)
                    person_name.append(name)






                else:
                    
                    img_boxes=rgb
                    person_name.append("none")
                
                
                # if i>149 and i<150: 
                #     print("person is at the center") 
                #     cv2.putText(img_boxes, str(f"center{i}"), (left, top-50), fontt, 0.5, (255, 0, 0), 2)


                # elif i >150:
                #     print("person is moving  his left") 
                #     cv2.putText(img_boxes, str(f"his left{i}"), (left+30, top-50), fontt, 0.5, (255, 0, 0), 2)


                # elif i <150:
                #     print("person is moving towards right")
                #     cv2.putText(img_boxes, str(f"his right{i}"), (left+30, top-50), fontt, 0.5, (255, 0, 0), 2)

                # n+=1 
                
                if n < len(x_cor):
                    if x_cor[n]>149 and x_cor[n]<150: 
                        print("person is at the center") 


                        c=f"{name} is at the center"
                        #speak(c)




                        cv2.putText(img_boxes, str(f"center{x_cor[n]}"), (left, top-50), fontt, 0.5, (255, 0, 0), 2)


                    elif x_cor[n]>150:
                        print("person is moving to his left") 

                        c=f"{name} is is moving towards his left"
                        #speak(c)
                        cv2.putText(img_boxes, str(f"his left{x_cor[n]}"), (left, top-50), fontt, 0.5, (255, 0, 0), 2)


                    elif x_cor[n] <150:
                        print("person is moving towards his right")

                        c=f"{name} is is moving towards his right"
                        #speak(c)
                        cv2.putText(img_boxes, str(f"his right{x_cor[n]}"), (left, top-50), fontt, 0.5, (255, 0, 0), 2)

                        
                    
                    n+=1
                



                    
            


        
                img_boxes=cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB)
                

                
        
                img_boxes=cv2.resize(img_boxes, (900 , 900 ))


                # Display the resulting frame
                cv2.imshow('frame', img_boxes)
                print("frame over")

                i+=1


            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            # a+=1
            # if a==70:
            #     print(a)
            #     break
        # After the loop release the cap object
        cap.release()
        # Destroy all the windows
        cv2.destroyAllWindows()
r=only_tracking
r.d()

