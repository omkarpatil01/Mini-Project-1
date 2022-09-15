import streamlit as st
import cv2
import numpy as np
import tensorflow
from PIL import Image

mod=tensorflow.keras.models.load_model('data.h5') 

cascPath="haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath) 

def get_age(distr):
     distr = distr*4
     if distr >= 0.65 and distr <= 1.4:return "0-18"
     if distr >= 1.65 and distr <= 2.4:return "19-30"
     if distr >= 2.65 and distr <= 3.4:return "31-80"
     if distr >= 3.65 and distr <= 4.4:return "80 +"
     return "Unknown"

def get_gender(prob):
     if prob < 0.5:return "Male"
     else: return "Female"

     
st.title("Age and Gender Prediction")

run = st.checkbox('Real time')
static = st.checkbox ('static image')
 
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
    
while run:
    
    _,frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = faceCascade.detectMultiScale(frame,scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+w),(255,0,0),2)
            roi=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            imgg=Image.fromarray(roi)
    
            img1=imgg.resize((64,64),Image.ANTIALIAS)
 
            img_64x64=np.array(img1)
            img_64x64=img_64x64.reshape(64,64,1)
            img_64x64 = img_64x64/255
            final_image=np.array([ img_64x64 ])

            val = mod.predict(final_image)
            age = get_age(val[0])
            gender = get_gender(val[1])
            label = "{},{}".format(gender,age)

            
            
            FRAME_WINDOW.image(frame)
            FRAME_WINDOW.image(cv2.putText(roi,label,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1))
            
              
             
if static:
    
    image_file = st.file_uploader("Upload Image :")

    if image_file is not None:
        our_image = Image.open(image_file)
    
        st.image(our_image)
        new_image = np.array(our_image.convert('RGB'))
        Img = cv2.cvtColor(new_image,1)
        Img = cv2.cvtColor(Img , cv2.COLOR_BGR2GRAY)
        new_img = np.array(our_image.convert('RGB'))
        img = cv2.cvtColor(new_img,1)
        gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

        #detect faces

        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

        #draw Rectangle

        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+w),(255,0,0),2)
            roi=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            imgg=Image.fromarray(roi)
    
            img1=imgg.resize((64,64),Image.ANTIALIAS)
 
            img_64x64=np.array(img1)
            img_64x64=img_64x64.reshape(64,64,1)
            img_64x64 = img_64x64/255
            final_image=np.array([ img_64x64 ])

            val = mod.predict(final_image)
            age = get_age(val[0])
            gender = get_gender(val[1])
            st.write("Age : {}, Gender : {}".format(age,gender))
            
        
           
        
            