from __future__ import print_function
from flask import Flask, request, render_template,send_from_directory
from keras import regularizers
import keras.backend.tensorflow_backend as tb
from PIL import Image
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras_preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import *
from image_search_engine import image_search_engine
import numpy as np
import inspect
import os
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import shutil, os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from os import listdir
from os.path import isfile, join
import os, re, os.path

tb._SYMBOLIC_SCOPE.value = True


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
glove_model_path = "/Volumes/My Passport for Mac/model/glove.6B"
data_path = "/Volumes/My Passport for Mac/data/imagesearch/"
features_path = "/Volumes/My Passport for Mac/model/imagesearch/features"
file_mapping_path = "/Volumes/My Passport for Mac/model/imagesearch/filemapping"
custom_features_path = "/Volumes/My Passport for Mac/model/imagesearch/customfeatures"
custom_features_file_mapping_path = "/Volumes/My Passport for Mac/model/imagesearch/customfilemapping"

images, vectors, image_paths, word_vectors = load_images_vectors_paths(glove_model_path, data_path)
model = image_search_engine.load_headless_pretrained_model()
images_features, file_index = image_search_engine.load_features(features_path, file_mapping_path)
image_index = image_search_engine.index_features(images_features)



# setting up template directory
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=TEMPLATE_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app = Flask(__name__)

def clear_dir(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            os.remove(os.path.join(root, file))



def searchInternal(image_path, model):
    #x_test = kimage.resize(image, 299, 299) 
    fv = image_search_engine.get_feature_vector(model, image_path)
    results = image_search_engine.search_index_by_value(fv, image_index, file_index)
    return results

def getImagesFilePathsFromFolder(path):
    onlyfiles = [ join(path,f) for f in listdir(path) if ( isfile(join(path, f)) and (".jpg" in f) )]
    return onlyfiles


	

def search():
    fileCount = len(getImagesFilePathsFromFolder(UPLOAD_FOLDER)) 
    #print(getImagesFilePathsFromFolder(path))
    fig, ax = plt.subplots(1,fileCount, figsize=(50,50))
    img_Counter=0;
    output = {}
    for img_path in getImagesFilePathsFromFolder(UPLOAD_FOLDER):
        print(img_path)
        image_url = 'http://127.0.0.1:9000/uploads/' + img_path.split("/")[-1]
        #image_url = image_url[:-2]
        print(image_url)
        results = searchInternal(img_path, model)
        for result in results:
            img_path = result[1]
            shutil.copy(img_path, UPLOAD_FOLDER)
            print("file " + img_path + " copied successfully.")
            image_url = 'http://127.0.0.1:9000/uploads/' + img_path.split("/")[-1]
            output[image_url] = "" 
        #ax[img_Counter].set_title(breed)
        #ax[img_Counter].imshow(load_img(img_path, target_size=(299, 299)))
        #img_Counter = img_Counter + 1
    return output

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def upload_file():
    if request.method == 'POST':
        print("clearing upload dir")
        clear_dir(UPLOAD_FOLDER)
        print("done")
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            output = search()
            if not (output is None):
                for imageUrl,character in output.items():
                    print(imageUrl)
                    print(character)
            return render_template('imagesearchengine.html', character=character ,output=output)
    else:
    	return render_template('imagesearchengine.html', review="" ,output=None)




@app.route("/")
def hello():
	return TEMPLATE_DIR

@app.route('/uploads/<filename>/')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/upload",	 methods=['GET', 'POST'])
def rec():
	return upload_file()



if __name__ == "__main__":
    app.run(host='127.0.0.1',port=9000, debug=False, threaded=False)


