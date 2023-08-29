import app
from flask import request, render_template, url_for
from tensorflow import keras
from keras import models
import numpy as np
from PIL import Image
import string
import random
import os

# adding Path to the config 
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'

# Loading model
model = models.load_model('app/static/model/bird_species.h5')

# Route to home page
@app.route('/', methods=["GET", "POST"])
def index():

    # Execute if requeste is GET
    if request.method == "GET":
        full_filename = "images/white_bg.jpg"
        return render_template("index.html", full_filename=full_filename)

    # if request.method == "POST":

    #     # Generating Unique image name
    #     letters = string.ascii_lowercase
    #     name = ''.join(random.choice(letters) for i in range(10)) + '.png'
    #     print(name)
    #     # full_filename = 'uploads/' + name