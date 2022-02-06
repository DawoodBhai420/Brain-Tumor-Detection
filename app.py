from flask import Flask, render_template, request
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "D:/Brain-Tumor-Detection/static/uploaded_images/" + imagefile.filename
    imagefile.save(image_path)

    model = keras.models.load_model("model.h5")
    img = load_img(image_path, target_size=(150,150))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    pred = model.predict(images)
    pred = int(pred)

    return render_template("result.html", pred=pred)



if __name__ == "__main__" :
    app.run(debug=True)
