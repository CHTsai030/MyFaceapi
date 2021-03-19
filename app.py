"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""
import flask
from flask import Flask
from flask import Response
from flask import render_template
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from tensorflow import keras
from PIL import Image
import numpy as np
from flask_cors import CORS
import io
import os
from io import BytesIO
import requests
import operator
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import patches
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig
from pprint import pprint
import time
app = flask.Flask(__name__)

# Make the WSGI interface available at the top level so wfastcgi can get it.
#wsgi_app = app.wsgi_app
CORS(app)
model = None

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image, mode="caffe")

    # return the processed image
    return image
@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')
@app.route('/FaceAnalysis')
def faceanalysis():
    # rendering webpage
    return render_template('FaceAnalysis.html')
@app.route('/DetectImage')
def DetectImage():
    # rendering webpage
    return render_template('DetectImage.html')

@app.route('/predict', methods=['POST'])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    #pprint(data)
    return (data)
faces=None
@app.route('/face', methods=['post'])
def face():    
    global faces
    data = {"success": False}
    if flask.request.method == "POST":
        data = flask.request.values.get("photourl_11")
   
        headers = {
            # Request headers
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': '我的金鑰',
        }

        params = {
            # Request parameters
            'returnFaceId': 'true',
            'returnFaceLandmarks': 'false',
            'returnFaceAttributes': 'age, gender, emotion',
            'recognitionModel': 'recognition_03',
            'returnRecognitionModel': 'false',
            'detectionModel': 'detection_01',
        }
        face_api_url ='我的端點/face/v1.0/detect'
        
        image_url=data        
        body={'url':image_url}         
    
        image = Image.open(BytesIO(requests.get(image_url).content))
        fig=plt.figure(figsize=(8,8))
        ax= plt.imshow(image, alpha=0.8)

        try:
            response = requests.post(face_api_url, params = params, headers = headers, json = body)
            faces=response.json()
                         
            for face in faces:
                emotion = face["faceAttributes"]["emotion"]
                maxemotion= max(emotion.items(), key=operator.itemgetter(1))
                age =int(face["faceAttributes"]["age"])
                gender=face["faceAttributes"]["gender"]
                #print("情緒:%s, 信心指數:%f" %(maxemotion[0],maxemotion[1]))
                #=====畫框框=======
                fr = face["faceRectangle"]
                origin = (fr["left"], fr["top"])
                
                p = patches.Rectangle(origin, fr["width"], fr["height"], fill= False, linewidth=2, color='darkgreen')
                ax.axes.add_patch(p)
                plt.text(origin[0], origin[1], "%s,%s"%(age,gender),fontsize=20, weight="bold", va= "bottom", color="forestgreen")                
            _ = plt.axis("off")         
            sio = BytesIO()
            fig.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.0)
            data = base64.encodebytes(sio.getvalue()).decode()
            url = 'data:image/png;base64,' + str(data)
            #pprint(faces)
            
            return url
        except Exception as e:
            print("[Errno {0}] {1}".format(e.errno, e.strerror))
       
            
@app.route('/facetext')
def facetext():
    global faces   
    time.sleep(4)      
    return flask.jsonify(faces)
  
    

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        model = ResNet50(weights="imagenet")
        PORT = int(os.environ.get('SERVER_PORT', '8080'))
    except ValueError:
        PORT = 8080
    app.run(HOST, PORT)
