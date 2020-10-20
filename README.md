# Motivation

The aim of this project is to build a web app that predicts a dog breed based on a user-supplied image. The web app will also detect whether the user-supplied image contains a human or a dog to begin with, so that the user can be greeted appropriately.

This is achieved through the use of three models:
* An OpenCV face detector is used to detect humans in the image.
* ResNet-50 (trained on ImageNet) is used to detect dogs in the image.
* An Xception model trained via transfer learning is used to predict dog breed.

The models used in this project are first explored and developed in another project, which can be [found here](https://github.com/msvest/dog-breed-predictor).

Note: This project forms part of Udacity's Data Scientist nanodegree.

# Libraries

* Python 3.7
* pandas
* scikit-learn
* keras
* opencv
* jupyter
* flask



# How to run the project

## Download missing files

A number of files are too large to upload to GitHub, which is why they have to be downloaded and added separately.

1. Download the [dataset of dog images](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). This should be unzipped and be placed inside `dog_breed_app/static/img/dogImages`.

2. Download the [Xception bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz). This should be placed in `Code/Supporting Files`.
  * Note that this is only required if you want to re-train the CNN found in file Training Breed Classifier.ipynb - it is not required for the app to function!

## Run the web app

Run the python script run.py from your terminal.

## Re-train the classifier

If you want to re-train the classifier, open and run Training Breed Classifier.ipynb. If you want to take a look at the code without running it, an html copy of the same file is also included.


# Full list of files

* **Code/**
  * **Supporting Files/**
    * **DogXceptionData.npz**: This contains bottleneck features for the Xception model. Note: needs to be downloaded!
    * **haarcascade_frontalface_alt.xml**: This file contains a pre-trained face detector from OpenCV.
    * **weights.best.Xception.hdf5**: This is the stored model weights for the final Xception model.

  * **Models.py**: This python file contains functions for loading and using the three different models used in the web app. These functions are imported by the routes.py file.

* **dog_breed_app/**
  * **static/**
    * **img/**
      * **dogImages/**: This directory contains train, validation, and test images of 133 different dog breeds for CNN training purposes. Note: needs to be downloaded!

  * **templates/**
    * **index.html**: Home page of the web app, including uploading of user images.
    * **result.html**: Page that shows the output of the classifier on the user supplied image.

  * **__init__.py**: Init file for treating dog_breed_app as a library.

  * **routes.py**: Contains all routing functions.

* **run.py**: File for running the web app.

* **Training Breed Classifier.ipynb**: Jupyter notebook that trains the Xception dog breed classifier model.

* **Training Breed Classifier.html**: An html copy of the above jupyter notebook for easier reading.


# Acknowledgements

Thanks to Udacity for the dog images dataset, the bottleneck features for training the Xception model, as well as starter code for training the various models.
