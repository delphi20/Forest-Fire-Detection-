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

# Debugging
import pdb
pdb.set_trace()  # breakpoint

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
#matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# GUI
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input


#Fethching and divinding the dataset into suitable variables
train_fire_image_path = pl.Path("E:\Study\GIKI BAI Course Material\Fourth Semester BAI\AI202\Project\Code\Dataset\data\Training and Validation/fire") #change paths to your current directory
train_non_fire_path = pl.Path("E:\Study\GIKI BAI Course Material\Fourth Semester BAI\AI202\Project\Code\Dataset\data\Training and Validation/noFire")
test_fire_image_path = pl.Path("E:\Study\GIKI BAI Course Material\Fourth Semester BAI\AI202\Project\Code\Dataset\data\Testing\\testfire")
test_non_fire_path = pl.Path("E:\Study\GIKI BAI Course Material\Fourth Semester BAI\AI202\Project\Code\Dataset\data\Testing\\nofire")

pdb.set_trace()  # breakpoint

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

pdb.set_trace()  # breakpoint

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

pdb.set_trace()  # breakpoint

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

pdb.set_trace()  # breakpoint

df_train = pd.DataFrame({"image": X,
                       "label": y})
df_test = pd.DataFrame({"image": X_test,
                       "label": y_test})

pdb.set_trace()  # breakpoint

# Plotting graph to see the datasets
sns.countplot(df_train["label"], palette='RdYlBu')

pdb.set_trace()  # breakpoint

sample = df_train.sample(10)

pdb.set_trace()  # breakpoint

# Converting images to numpy arrays
images = [np.array(image) for image in sample.image]

pdb.set_trace()  # breakpoint

# Creating a figure and an axis
fig, ax = plt.subplots(2, 5, figsize=(10, 4))

pdb.set_trace()  # breakpoint

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

pdb.set_trace()  # breakpoint

#storing data as numpy arrays 
X_sample = np.array(X)
Y_sample = np.array(y)
# normalising data to fit 0 to 1 values for cnn
X_sample = X_sample.astype('float32')
X_sample /=255

X_test = np.array(X_test)
y_test = np.array(y_test)

X_test = X_test.astype('float32')
X_test /=255

pdb.set_trace()  # breakpoint

#test train splitting 
X_train, X_val, Y_train, Y_val = train_test_split(X_sample, Y_sample, train_size = 0.7, shuffle=True)

pdb.set_trace()  # breakpoint

# Number of data in each variables for train and test sets
print(f"Data Number on X_train: {len(X_train)}")
print(f"Data Number on X_val: {len(X_val)}")
print(f"Data Number on X_test: {len(X_test)}")

pdb.set_trace()  # breakpoint

# creating the model and adding layers
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (img_size, img_size, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
# adding conv2d and maxpooling layers
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
#more conv2d and dropout layers
model.add(Conv2D(64, (3, 3), activation = 'relu'))
#model.add(SpatialDropout2D(0.2))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(SpatialDropout2D(0.4))
model.add(MaxPooling2D(pool_size = (2, 2)))
# flattening the input 
model.add(Flatten())
# 
model.add(Dense(units = 256, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(units = 256, activation = 'relu'))
model.add(Dropout(0.2))
# making the data output fit to 10 classes
model.add(Dense(units = 1, activation = 'sigmoid'))

pdb.set_trace()  # breakpoint

#Showing my model:
visualkeras.layered_view(model, legend=True)

pdb.set_trace()  # breakpoint

# Compiling the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

pdb.set_trace()  # breakpoint

# Defining Early Stop condition for Epochs
early_stoppping = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 30, restore_best_weights = True)
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=5)

pdb.set_trace()  # breakpoint

# Shifting the images inroder to improve training
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the data
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=10,  # randomly rotate images in the range 10 degrees
        zoom_range = 0.1, # Randomly zoom image 10%
        width_shift_range=0.1,  # randomly shift images horizontally 10%
        height_shift_range=0.1,  # randomly shift images vertically 10%
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

pdb.set_trace()  # breakpoint

datagen.fit(X_train)

pdb.set_trace()  # breakpoint

# saving history for the model and fitting it to the data 
history = model.fit(datagen.flow(X_train, Y_train, batch_size=48), 
#history = model.fit(X_train, Y_train,   
          batch_size=48,
          epochs=50,
          validation_data=(X_val, Y_val),
          callbacks=[early_stoppping,reduce_lr_on_plateau])

pdb.set_trace()  # breakpoint

# creating the function for the loss and accuracy to plot a graph
def graphLossAcurracy(loss, val_loss, accuracy, val_accuracy):
    fig, ax = plt.subplots(1,2, figsize=(16,8))
    ax[0].plot(loss, color='b', label="Training loss")
    ax[0].plot(val_loss, color='r', label="validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(accuracy, color='b', label="Training accuracy")
    ax[1].plot(val_accuracy, color='r',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)

#calling the function 
graphLossAcurracy(history.history['loss'], history.history['val_loss'], history.history['accuracy'], history.history['val_accuracy'])

pdb.set_trace()  # breakpoint

# saving the model loss and accuracy 
gc.collect()
score = model.evaluate(X_val, Y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', round(score[1]*100, 3), "%")

pdb.set_trace()  # breakpoint

# Making Predictions and plotting Heatmap between True and Predicted Values
Y_pred = model.predict(X_val)
Y_pred = np.where(Y_pred < 0.5, 0, 1)
Y_val = Y_val.astype(int)

plt.figure(figsize = (7,7))

sns.heatmap(confusion_matrix(Y_val, Y_pred),annot = True, fmt='g')
plt.title("CONFUSION MATRIX")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()

pdb.set_trace()  # breakpoint

# Making Predictions and Showing Results using trained model
errors = []
for i in range(0, len(Y_pred)):
    if Y_pred[i] != Y_val[i]:
        errors.append(i)

pdb.set_trace()  # breakpoint

# Creating a figure and an axis
fig, ax = plt.subplots(3, 10, figsize=(20, 6))

pdb.set_trace()  # breakpoint

# Drawing the images on the axis
for i, index in enumerate(errors[:30]):  # Displaying the first 30 errors
    valor_real = "No Fire" if Y_val[index] == 0 else "Fire"
    valor_predict = "Fire" if Y_pred[index] == 0 else "No Fire"
    if i < 10:
        ax[0, i].imshow(X_val[index])
        ax[0, i].text(0.5, -0.25, f"Real: {valor_real}, \n Predicted: {valor_predict}", transform=ax[0, i].transAxes, ha="center")
        ax[0, i].axis('off')
    elif i < 20:
        ax[1, i-10].imshow(X_val[index])
        ax[1, i-10].text(0.5, -0.25, f"Real: {valor_real}, \n Predicted: {valor_predict}", transform=ax[1, i-10].transAxes, ha="center")
        ax[1, i-10].axis('off')
    else:
        ax[2, i-20].imshow(X_val[index])
        ax[2, i-20].text(0.5, -0.25, f"Real: {valor_real}, \n Predicted: {valor_predict}", transform=ax[2, i-20].transAxes, ha="center")
        ax[2, i-20].axis('off')

# Showing the figure
plt.show()

pdb.set_trace()  # breakpoint

# evaluating the model and printing test set loss and accuracy 
score = model.evaluate(X_test, y_test, verbose=0)
print('Test set loss:', score[0])
print('Test set accuracy:', round(score[1]*100, 3), "%")

pdb.set_trace()  # breakpoint

# carrying out prediction using the model and printing a confusion matrix using the values 
Y_pred = model.predict(X_test)
Y_pred = np.where(Y_pred < 0.5, 0, 1)
y_test = y_test.astype(int)

plt.figure(figsize = (7,7))

sns.heatmap(confusion_matrix(y_test, Y_pred),annot = True, fmt='g')
plt.title("CONFUSION MATRIX")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()


## Added the gui

# Saving the model to HDF5 file
model.save("forest_fire_f.h5")


model = load_model("E:\Study\GIKI BAI Course Material\Fourth Semester BAI\AI202\Project\Code\\forest_fire_f.h5")  

# Function to preprocess image
def preprocess_image(image_path, target_size):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img)
    img = img.astype("float32")
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to make predictions
def predict_image(image_path):
    img = preprocess_image(image_path, target_size=(100, 100)) 
    prediction = model.predict(img)
    return prediction

# Function to handle image selection
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        prediction = predict_image(file_path)
        if prediction[0][0] > 0.5:
            result_label.config(text="No Fire")
        else:
            result_label.config(text="Forest onFire")
        # Display selected image
        image = Image.open(file_path)
        image = image.resize((300, 300), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo

# Create main window
root = tk.Tk()
root.title("Fire Detection")

# Create widgets
select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

root.mainloop()