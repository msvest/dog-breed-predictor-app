from dog_breed_app import app
from flask import render_template, request, redirect
from werkzeug.utils import secure_filename
from Code import Models as mod

import numpy as np
import os

#setting the max uploaded filesize to 10 MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['ALLOWED_IMAGE_EXTENSIONS'] = ['JPEG', 'JPG', 'PNG', 'GIF']

def allowed_image(filename):
    '''Input: filename.

    Returns boolean.

    Checks whether the filename has a period, and has one of the allowed
    extension types.
    '''
    if not '.' in filename:
        return False

    ext = filename.rsplit('.', 1)[1]

    if ext.upper() in app.config['ALLOWED_IMAGE_EXTENSIONS']:
        return True
    else:
        return False

@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        if request.files:

            image = request.files['image']

            if image.filename == '':
                print('No filename')
                return redirect(request.url)

            if allowed_image(image.filename):
                #making a numpy array of the image for compatibility with face_detector
                npimg = np.fromfile(image, np.uint8)

                greeting, prediction = mod.dog_breed_classifier(image, npimg)

                rand_dog_path = mod.example_dog_image(prediction)

                return render_template('result.html', greet1=greeting[0], greet2=greeting[1], img=rand_dog_path)

            else:
                print('That file extension is not allowed.')
                return redirect(request.url)

    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')
