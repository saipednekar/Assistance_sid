import tensorflow_hub as hub
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import face_recognition

class detect:
    def detector(self):
        print("inside")
        a=0
        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2


        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')



        print(tf.config.list_physical_devices('GPU'))

        print("loading")
        # detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite4/detection/2")

        # m = tf.keras.Sequential([
        #     hub.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b2/classification/2")
        # ])
        # m.build([1, 260, 260, 3])  
        detector=tf.saved_model.load("/home/sai/assistantAI/efficientdet_lite4_detection_2")

        m=tf.saved_model.load("/home/sai/assistantAI/imagenet_efficientnet_v2_imagenet1k_b2_classification_2")
  
        print("loaded")



        font = cv2.FONT_HERSHEY_SIMPLEX


        cap = cv2.VideoCapture(0)
        address="http://192.168.43.37:4747/video"
        cap.open(address)
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

        l_actual=[]
        l_obj=[]
        person_name=[]
        while(a<70):
            
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




                #Is optional but i recommend (float convertion and convert img to tensor image)
                rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

            #Add dims to rgb_tensor
                rgb_tensor = tf.expand_dims(rgb_tensor , 0)
            
            
                rgb_tensor
            
                #for object classification model
                rgb_tensor2=tf.cast(rgb_tensor,tf.float32)
        #         rgb_tensor2 = tf.convert_to_tensor(rgb_tensor, dtype=tf.float32)
                rgb_tensor2=rgb_tensor2/255
                


            #Add dims to rgb_tensor
        #         rgb_tensor2 = tf.expand_dims(rgb_tensor2 , 0)
                predictions = m(rgb_tensor2)
                
                dataf=pd.DataFrame(predictions[0])
            
                result=tf.argmax(dataf)

                
                labelss=pd.read_csv("labels2.csv")

            
                labels=np.array(labelss["background"][result])[0]
                
                print(labels)
                l_obj.append(labels)
                cv2.putText(rgb, str(labels), (20, 20), font, 0.3, (0, 255, 10), 1 )

            
            

                
            
            
                
                
                
                
                
                
                
                
                
                
                boxes, scores, classes, num_detection = detector(rgb_tensor)
            
            
                labelss=pd.read_csv("labels.csv",sep=";",index_col="ID")
            
            
                labels=labelss["OBJECT (2017 REL.)"]
            
                print(classes.numpy().astype('int'))
                pred_labels = classes.numpy().astype('int')[0]

                pred_labels = [labels[i] for i in pred_labels]

                pred_boxes = boxes.numpy()[0].astype('int')
                pred_scores = scores.numpy()[0]
                
                print(pred_scores)
                print(pred_labels)
            
            
            
            
            

                for score, (ymin, xmin, ymax, xmax),label in zip(pred_scores, pred_boxes,pred_labels):
                
                    if score > 0.5:
                        print("""after processing with detection probability here are the reduced disected detected classes below
                        
                        detected classes are below
                        
                        
                        
                        
                        """)
                        # print(score, (ymin, xmin, ymax, xmax),label)
                        # continue
                        
                    
                        
                        img_boxes = cv2.rectangle(rgb, (xmin, ymin), (xmax,ymax), (0,255,0), 2)

                        if locateface:
                            img_boxes =cv2.rectangle(img_boxes, (left, top), (right, bottom), color, 2)
                            cv2.putText(img_boxes, str(name), (left, top-20), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA )
                            print( str(name))
                            person_name.append(name)


                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(img_boxes, label, (xmin, ymax-10), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA )
                            score_txt = f"{100 * round(score)}%"
                            cv2.putText(img_boxes, score_txt, (xmin, ymax+10), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA )
                            print(score, (ymin, xmin, ymax, xmax),label)
                            l_actual.append(label)



                        else:
                            
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(img_boxes, label, (xmin, ymax-10), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA )
                            score_txt = f"{100 * round(score)}%"
                            cv2.putText(img_boxes, score_txt, (xmin, ymax+10), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA )
                            print(score, (ymin, xmin, ymax, xmax),label)
                            l_actual.append(label)
                            person_name.append("none")




                            
                    


                
                        img_boxes=cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB)
                        

                        
                
                        img_boxes=cv2.resize(img_boxes, (900 , 900 ))


                # Display the resulting frame
                        cv2.imshow('frame', img_boxes)
                        print("frame over")


            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            a+=1
            if a==70:
                print(a)
                break
        # After the loop release the cap object
        cap.release()
        # Destroy all the windows
        cv2.destroyAllWindows()
        return(l_actual,person_name,l_obj)

