import face_recognition
import cv2
import cvlib as cv
from datetime import datetime
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import json
import numpy as np
import os
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Camera():
    def __init__(self,number,path):

        self.number = number
        self.path = path
        self.time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.model = load_model('/home/nm/Documents/code/AI ML/CV/Face_1/Data/gender.h5')
        self.classes = ['Erkak','Ayol']
        self.names = ...
        self.encodes = ...
        self.cap = cv2.VideoCapture(number)
        
        self.frame = ...
        self.executor =ThreadPoolExecutor(max_workers=2)
        
    def save_data(self):
        data = {
                    'known_face_names':self.names,
                    'known_face_encodings':self.encodes
                }
        with open(os.path.join(self.path,f"data_{self.time}.json"),'w') as file:
            
            json.dump(data,file,cls=NumpyEncoder, indent=4)
            file.close()
    


    def load_data(self):
        filename = "/home/nm/Documents/code/AI ML/CV/Face_1/Data/data.json"
        with open(filename,'r') as file:
            data = json.load(file)
            file.close()
        self.names = np.array(data['known_face_names'])
        self.encodes = np.array(data['known_face_encodings'])
    
    
    def check_gender(self,face):
        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]

        face_crop = np.copy(self.frame[startY:endY,startX:endX])
        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            pass

        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
    
        conf = self.model.predict(face_crop,verbose=None)[0]
        

        # print(self.classes[np.argmin(conf)])
        return self.classes[np.argmin(conf)]


    def check_face(self,face_encoding,face_location):
        name = "NaN"

        face_distances = face_recognition.face_distance(self.encodes,face_encoding)
        
        matches = list(face_distances <= 0.5)
        if len(face_distances)>0:
            
            best_index = np.argmin(face_distances)
            if matches[best_index]:
                name = self.names[best_index]
            else:
                
                name = str(uuid4())
                self.names = np.append(self.names,name)
                self.encodes = np.append(self.encodes,face_encoding)
        return name    

 
    def camera(self):
        self.load_data()  
        procces_frame = True
        while True:
            ret,self.frame = self.cap.read()
            
            face , conf = cv.detect_face(self.frame)
            small_frame = cv2.resize(self.frame, (0,0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)
            
            
            if face:
                face_encodes = face_recognition.face_encodings(rgb_small_frame,face)
                face_zip = zip(face_encodes,face)

                futures = [
                    self.executor.submit(self.check_face, face_encoding,face_location) 
                    for face_encoding,face_location in face_zip
                    ]      
                face_names = [future.result() for future in futures] 
            
            procces_frame = False

            cv2.imshow(str(f"Camera {self.number}"),self.frame)
            cv2.waitKey(1)
        self.cap.release()



        

        