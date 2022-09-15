import sys
import os
import numpy as np
import cv2
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import wsgiserver

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH = 'data.h5'

# Load your trained model
new_model = load_model('data.h5')
      # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

cascPath="data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

def loadImage(filepath):
    test_img = image.load_img(filepath, target_size=(64, 64))
    test_img = image.img_to_array(test_img)
    img1=test_img.resize((64,64),Image.ANTIALIAS)
    img_64x64=np.array(img1)
    print(img_64x64)
    img_64x64=img_64x64.reshape(64,64,1)
    test_img = np.expand_dims(test_img, axis = 0)
    test_img /= 255
    final_image=np.array([ test_img ])

    return final_image


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


def get_face(sample):
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, h:h+w].copy()
        roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    return roi



def model_predict(img_path):
    #global new_model
    image_1= loadImage(img_path)
    face=get_face(image_1)
    val = new_model.predict(face)
    age = get_age(val[0])
    gender = get_gender(val[1])
    #image_2=cv2.resize(image_1,(64,64))
    #image_2=image_2.reshape(64,64,1)
    #prediction_1 = new_model.predict(image_2)
    #age=get_age(prediction_1[0])
    #gender=get_gender(prediction_1[1])
    #img = image.load_img(img_path)                        
    return age, gender

'''def model_predict(img_path):
    #global new_model
    image_1= loadImage(img_path)
    frame =image_1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        
        roi = frame[y:y+h, h:h+w].copy()
        roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)

        img=Image.fromarray(roi)
    
        img1=img.resize((64,64),Image.ANTIALIAS)
 
        img_64x64=np.array(img1)
        img_64x64=img_64x64.reshape(64,64,1)
        img_64x64 = img_64x64/255
        final_image=np.array([ img_64x64 ])

        val = new_model.predict(final_image)
        age = get_age(val[0])
        gender = get_gender(val[1])
        
    #img = image.load_img(img_path)                        
    return age, gender'''


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        age, gender = model_predict(file_path)

        return ' '+ age + '    ' + ', '+ gender
    return None

if __name__ == '__main__':
    app.run(debug=True)
