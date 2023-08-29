import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# loading model
model = load_model("breed_prediction.h5")

# Class Names
BREED_CLASS = ["scottish_deerhound", "maltese_dog", "bernese_mountain_dog"]

# setting titles for the app
st.title("Predict your Dog Breed")
st.markdown("Upload an image of your dog.")

# upload the image
dog_image = st.file_uploader("Choose an image...", type='jpg')
submit = st.button('Predict')

# on predict button click
if submit:

    if dog_image is not None:
        # convert an image to opencv image
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype='uint8')
        opencv_image = cv2.imdecode(file_bytes, 1)


        #Displaying the image 
        st.image(opencv_image, channels ="BGR")
        #Resizing the image
        opencv_image = cv2.resize(opencv_image,(224,224))
        #convert image to 4 dimensional
        opencv_image.shape = (1,224,224,3)
        #make predictions
        y_pred = model.predict(opencv_image)
        st.title(str("The Dog Breed is :- {}".format(BREED_CLASS[np.argmax(y_pred)])))