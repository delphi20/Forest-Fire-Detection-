
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
