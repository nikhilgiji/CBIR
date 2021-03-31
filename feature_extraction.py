import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import Model 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import load_model 
from sklearn.metrics import label_ranking_average_precision_score 

#load mist dataset 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#load model 
model = load_model('autoencoder.h5') 

#encoder layer from the trained model 
encoder = Model(inputs = model.input, outputs = model.get_layer('encoder').output) 

#array to score computed scores 
scores = [] 

#to save computation time we keep only 1000 query images from the test dataset 
n_test_samples = 1000



