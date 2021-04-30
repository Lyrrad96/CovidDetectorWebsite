import streamlit as st
import time
import numpy as np
import streamlit.components.v1 as components

import numpy as np
import matplotlib.pyplot as plt

import cv2
import tensorflow as tf
from keras.preprocessing import image
import os


model = tf.keras.models.load_model("/content/drive/MyDrive/Detection/model_adv.h5")

categories = ["Covid", "Normal"]  # will use this to convert prediction num to string value

uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg'])
# Using PIL
from PIL import Image
img= np.asarray(Image.open(uploaded_file))

#img = np.asarray(img)
st.image(img, caption='Uploaded Image.')

#img = image.load_img('/content/drive/MyDrive/DataSet/Val/Covid/10.1016-slash-j.anl.2020.04.002-a.png', target_size=(224, 224))

x = image.img_to_array(img)
st.write(x)
#st.image(x, caption='Uploaded Image.')
x = np.expand_dims(x, axis=0)
st.write(x)

images = np.vstack([x])
images = np.divide(images, 255)
st.image(images, caption='Uploaded Image.')

pred = model.predict(images)

#st.write(pred)

string="Result: "+categories[int(pred[0][0])]

st.success(string)
