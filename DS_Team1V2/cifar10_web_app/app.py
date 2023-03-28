from PIL import Image
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from imageio.v2 import imread
from skimage.transform import resize
from os.path import exists
import gdown
import sys
import os

if not exists("vgg16_cifar10_new.hdf5"):
    url = 'https://drive.google.com/file/d/1LJnhd4X4GIjoaBv40DtgxGryT33xQykT/view?usp=sharing'
    output = 'vgg16_cifar10_new.hdf5'
    gdown.download(url, output, quiet=False, fuzzy=True)

model_filename = "vgg16_cifar10_new.hdf5"
model = tf.keras.models.load_model(model_filename)

app = Flask(__name__)


@app.route('/')
def index_view():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def predict():
    file = request.files['file']
    img = Image.open(file)
    img = img.resize((80, 80))
    img = img.convert("RGB")
    input_image = np.reshape(img, (-1, 80, 80, 3))
    LABEL_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    predict_img = model.predict(input_image)
    pred_class = LABEL_NAMES[np.argmax(predict_img)]
    response = str(f'Its - {pred_class}')
    return response


if __name__ == '__main__':
    app.run(debug=True, port=8000)