import streamlit as st
st.set_page_config(layout="centered")
import numpy as np
import streamlit.components.v1 as components

import tensorflow as tf
from keras.preprocessing import image


HtmlFile = open("/content/drive/MyDrive/Detection/home.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
print(source_code)
components.html(source_code)

model = tf.keras.models.load_model("/content/drive/MyDrive/Detection/model.h5")

  # will use this to convert prediction num to string value

#uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg'])
# Using PIL
#from PIL import Image
#img= np.asarray(Image.open(uploaded_file))

#img = np.asarray(img)
#st.image(img, caption='Uploaded Image.')


#load image
img = image.load_img('file:///home/darryl/Desktop/VI/Prozec/CovidDataset/Test/Normal/Normal-799.png', target_size=(224, 224))

#convert to numpy array
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
images = np.divide(images, 255)
#print image
st.image(images, caption='Uploaded Image.')

#make prediction
pred = model.predict(images)

pred[0][0]=round(pred[0][0])
#declare categories
categories = ["Positive", "Negative"]
#display result
string="Result: "+categories[int(pred[0][0])]
st.write(pred[0][0])
if pred[0][0] == 0:
  st.error(string)
else:
  st.success(string)
