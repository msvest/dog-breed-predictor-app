
### Loading Face Detector

import cv2

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('Code/Supporting Files/haarcascade_frontalface_alt.xml')

# returns "True" if face is detected in image stored at img_path
def face_detector(image):
    '''Input: image file as numpy array.

    Returns True if face is detected in the image by the OpenCV face detector.
    '''
    #this version takes in numpy array image and reads that in.
    #change was made because imread function requires filepath, not file argument.
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0




### Load Dog Detector

from tensorflow.keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

# functions to process images
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

# returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))






### Load Breed Classifier Model

import numpy as np
from glob import glob

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def extract_Xception(tensor):
    from keras.applications.xception import Xception, preprocess_input
    return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

# load list of dog names
dog_names = [item[40:-1] for item in sorted(glob("dog_breed_app/static/img/dogImages/train/*/"))]

# Xception model architecture
Xception_model = Sequential()
Xception_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
Xception_model.add(Dense(133, activation='softmax'))

#Xception_model.summary()


# Compile model
Xception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# Load best weights from prior training. See file: Training Breed Classifier.ipynb
Xception_model.load_weights('Code/Supporting Files/weights.best.Xception.hdf5')


def Xception_predict_breed(img_path):
    '''Input: path to an image.

    Returns the name of the dog breed that is the closest match to the image
    according to the trained Xception model.
    '''
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    predicted_vector = Xception_model.predict(bottleneck_feature)

    return dog_names[np.argmax(predicted_vector)]




### Final algorithms
import os
import random

def dog_breed_classifier(img, image_array):
    '''Input: image file, image file as numpy array.
    Output: [greeting, prediction statement], prediction

    Returns dog breed prediction based on the supplied image, along with a greeting.

    Greeting is determined based on whether a dog, a human, or neither is found
    in the image.
    '''
    prediction = Xception_predict_breed(img)
    clean_prediction = prediction.split('.')[1].replace('_', ' ')

    if dog_detector(img):
        return ['Hello, dog!', 'You appear to be a(n) {}.'.format(clean_prediction)], prediction

    elif face_detector(image_array):
        return ['Hello, human!', 'The dog breed you most resemble is: {}.'.format(clean_prediction)], prediction

    else:
        return ['Hmm, it looks like this image contains neither a dog nor a human.', \
                'But the dog breed this most resembles is: {}.'.format(clean_prediction)], prediction

def example_dog_image(dog_folder):
    '''Input: dog breed output of the Xception model. (Note: raw output, not cleaned one.)

    Returns path to a random dog image of that breed.
    '''
    path = 'dog_breed_app/static/img/dogImages/train'+dog_folder

    #choose one of the pictures in this directory at random
    picture = random.choice(os.listdir(path))
    path = path+'/'+picture

    #remove early part of path so that it begins with static/
    path = path[14:]

    return path
