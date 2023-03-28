import os

os.environ["CUDA_VISIBLE_DEVICES"] = "  2"
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np
import matplotlib.pylab as plt
import itertools
from sklearn.utils import shuffle
import time
import tensorflow as tf

import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, learning_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.layers import MaxPooling2D

import cv2

X = np.load('data/X.npy')
Y = np.load('data/Y.npy')

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2)
xtrain = xtrain / 255.0
xtest = xtest / 255.0
numberoftrain = xtrain.shape[0]
numberoftest = xtest.shape[0]
numberOfClass = 2

ytrain = to_categorical(ytrain, numberOfClass)
ytest = to_categorical(ytest, numberOfClass)
input_shape = xtrain.shape[1:]


def resize_img(img):
    numberOfImage = img.shape[0]
    new_array = np.zeros((numberOfImage, 64, 64, 3))
    for i in range(numberOfImage):
        new_array[i] = cv2.resize(img[i, :, :, :], (64, 64))
    return new_array


xtrain = resize_img(xtrain)
xtest = resize_img(xtest)
print('resized')



def main():
    base_model = tf.keras.applications.DenseNet161(input_shape=(64, 64, 3), include_top=False,
                                                   weights='imagenet')
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(2, activation='sigmoid'))

    for layer in base_model.layers:
        layer.trainable = False
    # model summary
    model.summary()
    model.save('./model/model-ResNet50.h5')
    print('ResNet50 model saved')

    model.compile(loss="binary_crossentropy",
                  optimizer="rmsprop",
                  metrics=["accuracy"])

    hist = model.fit(xtrain, ytrain, validation_split=0.3, epochs=100, batch_size=1000)
    plt.title('MobileNet - Loss')
    plt.plot(hist.history["loss"], label="train loss")
    plt.plot(hist.history["val_loss"], label="val loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('MobileNet - Accuracy')
    plt.plot(hist.history['accuracy'], label="train acc")
    plt.plot(hist.history['val_accuracy'], label="val acc")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
