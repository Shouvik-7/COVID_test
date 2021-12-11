import numpy as np
import io
import base64
from PIL import Image
import cv2
import io
import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

def get_model():
    global model
    global model2
    model = load_model('./saved-model-01-0.97.h5')
    model2 = load_model('./custom_model.h5')
    print("* Model loaded")

def preprocess_image(image,target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = (1./255) * (image)
    image = np.expand_dims(image,axis=0)
    return image


print(" * Loading keras model")
get_model()

@app.route("/predict",methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image,target_size=(224,224))
    prediction = model.predict(processed_image).tolist()
    if round(prediction[0][0])==0:
        pred='+ve'
    else:
        pred='-ve'
    
    response = {
        'prediction':{
            
            'COVID': pred
        }
    }
    return jsonify(response)    
        
@app.route("/predict2",methods=["POST"])
def predict2():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image,target_size=(224,224))
    prediction = model2.predict(processed_image).tolist()
    prediction = np.argmax(prediction, axis=1)
    true_classes = ["COVID","NORMAL","PNEUMONIA"]
    pred = true_classes[prediction[0]]
    response = {
        'prediction':{
            
            'COVID': pred
        }
    }
    return jsonify(response)
if __name__ == "__main__":
    app.run(debug=True)
