import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tensorflow

@st.cache
def load_image(img):
    im=Image.open(img)
    return im


mod=tensorflow.keras.models.load_model('data.h5') 

cascPath="data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath) 
        

def detect_face(our_image):
    
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
            label ="{},{}".format(gender,age)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img,label, (x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        
        return img, faces    

        

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


def main():
    """Age and  Gender Detection App"""

    st.title("Age and Gender Detection")
    
    image_file = st.file_uploader("Upload Image :")

    if image_file is not None:
        our_image = Image.open(image_file)
    
        st.image(our_image)
        new_image = np.array(our_image.convert('RGB'))
        Img = cv2.cvtColor(new_image,1)
        Img = cv2.cvtColor(Img , cv2.COLOR_BGR2GRAY)
        
        result_img = detect_face(our_image)

        st.image(result_img)

        


        

#if __name__ == '__main__':
main()