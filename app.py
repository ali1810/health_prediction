
# run it with:
# python3 app.py

#import the necessary libraries
from flask import Flask, render_template , request,redirect,send_file
import os,glob
from flask.wrappers import Request
import tensorflow as tf
#graph = tf.get_default_graph()
#import threading
from keras.models import load_model
import seaborn as sns
import cv2
import numpy as np
import pandas as pd

app = Flask(__name__)
# load the model from disk
#test()

#print(select)

#model = pickle.load(open('finalized_model_96_new.pkl', 'rb'))
#model2 = pickle.load(open('finalized_model_ethanol_98%.pkl', 'rb'))

#model=load_model('CAE_85%_lungs_model_18_08.h5')

@app.route('/')
def index():
    return render_template(
        'sub.html',
        data=[{'name':'Brain'}, {'name':'Breast'}, {'name':'Lungs'},{'name':'Skin'}])

@app.route("/test" , methods=['GET', 'POST'])
def test(): 
    #def handle_view(select):
    global select 
    global model 
    select = request.form.get('comp_select')
    #select = str(select)
    if select =='Brain':
        model=load_model('CAE_87%_brain_model_18_08.h5')
    elif select =='Breast':
        model=load_model('CAE_89%_breast_model_18_08.h5')
    elif select == 'Lungs':
        model=load_model('CAE_85%_lungs_model_18_08.h5')
    else:
        model=load_model('CAE_98_skin_model_18_08.h5')   

         #return "Thanks"
   #print(model)
    return render_template('sub.html',solvent_text = "The Selected Cancer Disease is {}".format(select)) # just to see what select isprint(test.model)
#threading.thread.start_new_thread(handle_sub_view, select)

#@app.route('/predict', methods=['POST'])
#def predict():
    #if request.method == "POST":
     #   smiles = request.form["smiles"]
        #print(smiles)
    #predOUT = predictSingle(smiles, model)
    #predOUT = predOUT +0.20

    #return render_template('sub.html', prediction_text = "The log S is {}".format(predOUT))
    #return render_template('sub.html',resu= "The log S is {}".format(predOUT))  

app.config["UPLOAD_PATH"]=  'static/uploads'
#app.config["DOWNLOAD_PATH"]='C:/Users/ali/Desktop/solub_herokuu-main/static/downloads'
@app.route('/upload_file', methods=["GET", "POST"])
def upload_file():
    if request.method == 'POST':
        if select == 'Lungs':
            model=load_model('CAE_85%_lungs_model_18_08.h5')
            dir = app.config["UPLOAD_PATH"]
            for f in os.listdir(dir):
               os.remove(os.path.join(dir, f))
            f=request.files['file_name']
            f.save(os.path.join(app.config['UPLOAD_PATH'], f.filename))
            image = cv2.imread(os.path.join(app.config['UPLOAD_PATH'], f.filename))
            img1=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
            img1 =cv2.resize(img1,(48,48)) 
            img1 = img1.astype('float32') / 255.
            img1 = np.reshape(img1, [1,48,48,3])
            preds = model.predict(img1)
            mse = np.mean((img1 - preds) ** 2)
            label = "Anomalous" if mse > 0.005264 else "normal"
            color = (255, 0, 0) if mse > 0.005264 else (0, 255, 0)
            cv2.putText(image,label, (15,  20), cv2.FONT_HERSHEY_SIMPLEX,
	        0.9,color,4)
            cv2.imwrite('static/uploads/output.png',image)  
            result = os.path.join('static/uploads/output.png') 
        elif select =='Breast':
            model=load_model('CAE_89%_breast_model_18_08.h5')    
            dir = app.config["UPLOAD_PATH"]
            for f in os.listdir(dir):
                   os.remove(os.path.join(dir, f))
            f=request.files['file_name']
            f.save(os.path.join(app.config['UPLOAD_PATH'], f.filename))
            image = cv2.imread(os.path.join(app.config['UPLOAD_PATH'], f.filename))
            img1=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
            img1 =cv2.resize(img1,(48,48)) 
            img1 = img1.astype('float32') / 255.
            img1 = np.reshape(img1, [1,48,48,3])
            preds = model.predict(img1)
            mse = np.mean((img1 - preds) ** 2)
            image1 =cv2.resize(image,(128,128)) 
            label = "Anomalous" if mse > 0.003288 else "normal"
            color = (255, 0, 0) if mse > 0.003288 else (0, 255, 0)
            cv2.putText(image1,label, (15, 20), cv2.FONT_HERSHEY_SIMPLEX,
	        0.9,color,4)
            cv2.imwrite('static/uploads/output.png',image1)
            result = os.path.join('static/uploads/output.png')
        elif select =='Brain':
            model=load_model('CAE_87%_brain_model_18_08.h5')
            dir = app.config["UPLOAD_PATH"]
            for f in os.listdir(dir):
                   os.remove(os.path.join(dir, f))
            f=request.files['file_name']
            f.save(os.path.join(app.config['UPLOAD_PATH'], f.filename))
            image = cv2.imread(os.path.join(app.config['UPLOAD_PATH'], f.filename))
            img1=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
            img1 =cv2.resize(img1,(48,48)) 
            img1 = img1.astype('float32') / 255.
            img1 = np.reshape(img1, [1,48,48,3])
            preds = model.predict(img1)
            mse = np.mean((img1 - preds) ** 2)
            label = "Anomalous" if mse > 0.01145 else "normal"
            color = (255, 0, 0) if mse > 0.01145 else (0, 255, 0)
            cv2.putText(image,label, (15,  20), cv2.FONT_HERSHEY_SIMPLEX,
	        0.9,color,4)
            cv2.imwrite('static/uploads/output.png',image)  
            result = os.path.join('static/uploads/output.png')
        else:
            model=load_model('CAE_98_skin_model_18_08.h5')
            dir = app.config["UPLOAD_PATH"]
            #for zippath in glob.iglob(os.path.join(dir, '*.png')):
            #    os.remove(zippath)
            for f in os.listdir(dir):
                   os.remove(os.path.join(dir, f))
            
            f=request.files['file_name']
            f.save(os.path.join(app.config['UPLOAD_PATH'], f.filename))
            image = cv2.imread(os.path.join(app.config['UPLOAD_PATH'], f.filename))
            img1=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
            img1 =cv2.resize(img1,(48,48)) 
            img1 = img1.astype('float32') / 255.
            img1 = np.reshape(img1, [1,48,48,3])
            preds = model.predict(img1)
            mse = np.mean((img1 - preds) ** 2)
            label = "Anomalous" if mse > 0.002203 else "normal"
            color = (0, 0, 255) if mse > 0.002203 else (0, 255, 0)
            cv2.putText(image,label, (15,  20), cv2.FONT_HERSHEY_SIMPLEX,
	        0.9,color,4)
            cv2.imwrite('static/uploads/output.png',image)  
            result = os.path.join('static/uploads/output.png')      
        return render_template('show.html', result=result)
        
    return render_template("upload_file.html", msg="Please choose a image for {}".format(select))    
if __name__ == "__main__":
    app.run(debug=True, port=7000)

