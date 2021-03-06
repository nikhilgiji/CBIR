import tensorflow as tf 
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import label_ranking_average_precision_score
import time

print('Loading mnist dataset')
t0 = time.time()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)) 
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
t1 = time.time()
print('mnist dataset loaded in: ', t1-t0)

print('Loading model :')
t0 = time.time()
# Load previously trained autoencoder
autoencoder = load_model('autoencoder.h5')
t1 = time.time()
print('Model loaded in: ', t1-t0) 

#plot image 

#denoised_images = model.predict(x_test_noisy.reshape(x_test_noisy.shape[0] 1))
# test_img = x_test_noisy[0]
# resized_test_img = cv2.resize(test_img, (280, 280))
# cv2.imshow('input', resized_test_img)
# cv2.imwrite('test_image.jpg', resized_test_img)
# cv2.waitKey(0)
# output = denoised_images[0]
# resized_output = cv2.resize(output, (280, 280))
# cv2.imshow('output', resized_output) 
# cv2.imwrite('denoised_image.jpg', resized_output)
# cv2.waitKey(0)