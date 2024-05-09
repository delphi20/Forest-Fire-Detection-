
# Importing Libraries
import visualkeras
import numpy as np
import pandas as pd
import gc
gc.collect()
import seaborn as sns
import random
import pathlib as pl
import glob
import cv2
from sklearn.model_selection import train_test_split

# Machine Learning
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import AveragePooling2D, Dense, BatchNormalization, Dropout, Flatten, Conv2D, MaxPooling2D, SpatialDropout2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import Recall,AUC
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Transfer Learning
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import tensorflow.keras.applications as models
from tensorflow.keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator

# Performance Metrics
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import itertools
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# GUI
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input


#Fethcing and divinding the dataset into suitable variables
train_fire_image_path = pl.Path("E:\Study\GIKI BAI Course Material\Fourth Semester BAI\AI202\Project\Code\Dataset\data\Training and Validation/fire") #change paths to your current directory
train_non_fire_path = pl.Path("E:\Study\GIKI BAI Course Material\Fourth Semester BAI\AI202\Project\Code\Dataset\data\Training and Validation/noFire")
test_fire_image_path = pl.Path("E:\Study\GIKI BAI Course Material\Fourth Semester BAI\AI202\Project\Code\Dataset\data\Testing\\testfire")
test_non_fire_path = pl.Path("E:\Study\GIKI BAI Course Material\Fourth Semester BAI\AI202\Project\Code\Dataset\data\Testing\\nofire")


# Defining train and test datasets and labels
train_data_images = {
    "Fire":list(train_fire_image_path.glob("*.jpg")),
    "Non_Fire":list(train_non_fire_path.glob("*.jpg"))
}
test_data_images = {
    "Fire":list(test_fire_image_path.glob("*.jpg")),
    "Non_Fire":list(test_non_fire_path.glob("*.jpg"))
}
train_labels = {
    "Fire":0,"Non_Fire":1
}

# Cleaning and Resizing the images in the dataset
X, y = [], []
train_count = 0
test_count = 0
img_size = 100
for label, images in train_data_images.items():
    for image in images:
        train_count = train_count+1
        img = cv2.imread(str(image)) # Reading the image
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(train_labels[label])

X_test, y_test = [], []
for label, images in test_data_images.items():
    for image in images:
        test_count = test_count+1
        img = cv2.imread(str(image)) # Reading the image
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            X_test.append(img)
            y_test.append(train_labels[label])


df_train = pd.DataFrame({"image": X,
                       "label": y})
df_test = pd.DataFrame({"image": X_test,
                       "label": y_test})

# Plotting graph to see the datasets
sns.countplot(df_train["label"], palette='RdYlBu')

sample = df_train.sample(10)

# Converting images to numpy arrays
images = [np.array(image) for image in sample.image]

# Creating a figure and an axis
fig, ax = plt.subplots(2, 5, figsize=(10, 4))

# Drawing the images on the axis
for i, image in enumerate(images):
    texto = "Fire" if sample.iloc[i].label == 0 else "No Fire"
    if i < 5:
        ax[0, i].imshow(image)
        ax[0, i].text(0.5, -0.1, texto, transform=ax[0, i].transAxes, ha="center")
        ax[0, i].axis('off')
    else:
        ax[1, i-5].imshow(image)
        ax[1, i-5].text(0.5, -0.1, texto, transform=ax[1, i-5].transAxes, ha="center")
        ax[1, i-5].axis('off')
# Showing the figure
plt.show()