import os
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from image_search_engine import image_search_engine
import matplotlib.pyplot as plt
from PIL import Image
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras_preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import inspect
from argparse import ArgumentParser
import os
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

def load_images_vectors_paths(glove_model_path, data_path):
    word_vectors = image_search_engine.load_glove_vectors(glove_model_path)
    images, vectors, image_paths = load_paired_img_wrd(data_path, word_vectors)
    return images, vectors, image_paths, word_vectors


def load_paired_img_wrd(folder, word_vectors, use_word_vectors=True):
    '''
    If use_word_vectors = true, and using VGG16 with Imagenet:
    Will have 300 embedding layer at end of network
    Instead of 4096 imagenet class layer at the end of the network
    '''
    class_names = [fold for fold in os.listdir(folder) if ".DS" not in fold]
    image_list = []
    labels_list = []
    paths_list = []
    for cl in class_names:
        splits = cl.split("_")
        if use_word_vectors:
            vectors = np.array([word_vectors[split] if split in word_vectors else np.zeros(shape=300) for split in splits])
            class_vector = np.mean(vectors, axis=0)
        subfiles = [f for f in os.listdir(folder + "/" + cl) if ".DS" not in f]

        for subf in subfiles:
            #print("subf: " + subf)
            full_path = os.path.join(folder, cl, subf)
            #print("full path : " + full_path)
            img = image.load_img(full_path, target_size=(224, 224))
            #print(type(img))
            
            x_raw = image.img_to_array(img)
            #print(x_raw.shape)
            x_expand = np.expand_dims(x_raw, axis=0)
            #print("after expand")
            #print(x_expand.shape)
            x = preprocess_input(x_expand)
            #print("after preprocess")
            #print(x.shape)
            image_list.append(x)
            if use_word_vectors:
                labels_list.append(class_vector)
            paths_list.append(full_path)
    img_data = np.array(image_list)
    #print("image_data_shape")
    #print(img_data.shape)
    #print("image_data_type")
    #print(type(img_data))
    
    
    img_data = np.rollaxis(img_data, 1, 0)
    img_data = img_data[0]

    return img_data, np.array(labels_list), paths_list

def displayImagebyPath(img_path):
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    img_Counter=0;
    ax.set_title(img_path)
    ax.imshow(load_img(img_path, target_size=(224, 224)))


def displayImages(results):
    fig, ax = plt.subplots(1,len(results), figsize=(50,50))
    img_Counter=0;
    for result in results:
        img_path = result[1]
        #print(img_path)
        ax[img_Counter].set_title("")
        ax[img_Counter].imshow(load_img(img_path, target_size=(224, 224)))
        img_Counter = img_Counter + 1

def load_paired_img_wrd(folder, word_vectors, use_word_vectors=True):
    '''
    If use_word_vectors = true, and using VGG16 with Imagenet:
    Will have 300 embedding layer at end of network
    Instead of 4096 imagenet class layer at the end of the network
    '''
    class_names = [fold for fold in os.listdir(folder) if ".DS" not in fold]
    image_list = []
    labels_list = []
    paths_list = []
    for cl in class_names:
        splits = cl.split("_")
        if use_word_vectors:
            vectors = np.array([word_vectors[split] if split in word_vectors else np.zeros(shape=300) for split in splits])
            class_vector = np.mean(vectors, axis=0)
        subfiles = [f for f in os.listdir(folder + "/" + cl) if ".DS" not in f]

        for subf in subfiles:
            print("subf: " + subf)
            full_path = os.path.join(folder, cl, subf)
            print("full path : " + full_path)
            img = image.load_img(full_path, target_size=(224, 224))
            print(type(img))
            x_raw = image.img_to_array(img)
            x_expand = np.expand_dims(x_raw, axis=0)
            x = preprocess_input(x_expand)
            image_list.append(x)
            if use_word_vectors:
                labels_list.append(class_vector)
            paths_list.append(full_path)
    img_data = np.array(image_list)
    img_data = np.rollaxis(img_data, 1, 0)
    img_data = img_data[0]

    return img_data, np.array(labels_list), paths_list