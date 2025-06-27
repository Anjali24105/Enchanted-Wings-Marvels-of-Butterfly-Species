from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)

# Load your trained model
model = load_model("butterfly_model.h5")

# 🔁 Label map (you can extend this to full 75 labels later)
label_map = {
    0: 'ADONIS',
    1: 'APPOLLO',
    2: 'ATALA',
    3: 'BANDED PEACOCK',
    4: 'BLACK HAIRSTREAK',
    5: 'BLUE MORPHO',
    6: 'BLUE SPOTTED CROW',
    7: 'BROWN SIPROETA',
    8: 'CABBAGE WHITE',
    9: 'CAIRNS BIRDWING',
    10: 'CHECQUERED SKIPPER',
    11: 'CLOUDED SULPHUR',
    12: 'COMMON WOOD-NYMPH',
    13: 'COPPER TAIL',
    14: 'CRIMSON PATCH',
    15: 'DANAID EGGFLY',
    16: 'EASTERN DAPPLE WHITE',
    17: 'GREAT JAY',
    18: 'GREEN CELLED CATTLEHEART',
    19: 'INDRA SWALLOW',
    20: 'IPHICLUS SISTER',
    21: 'JULIA',
    22: 'LARGE MARBLE',
    23: 'MALACHITE',
    24: 'MANGROVE SKIPPER',
    25: 'MONARCH',
    26: 'ORANGE OAKLEAF',
    27: 'PAINTED LADY',
    28: 'PAPER KITE',
    29: 'PEACOCK',
    30: 'PINE WHITE',
    31: 'PIPEVINE SWALLOW',
    32: 'PURPLISH COPPER',
    33: 'QUESTION MARK',
    34: 'RED ADMIRAL',
    35: 'RED CRACKER',
    36: 'RED POSTMAN',
    37: 'SCARCE SWALLOW',
    38: 'SOOTYWING',
    39: 'SOUTHERN DOGFACE',
    40: 'STRAITED QUEEN',
    41: 'TWO BARRED FLASHER',
    42: 'VICEROY'
}

# 🏠 Homepage
@app.route('/')
def home():
    return render_template("index.html")

# 🚀 Get Started Page
@app.route('/get_started.html')
def get_started():
    return render_template("get_started.html")

# 📤 Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    filepath = os.path.join('static', 'uploaded_image.jpg')
    file.save(filepath)

    # Load and preprocess image
    img = load_img(filepath, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = label_map.get(predicted_class, "Unknown")

    print("Predicted class index:", predicted_class)
    print("Predicted label:", predicted_label)

    return render_template("get_started.html", prediction=predicted_label, image_file=filepath)

# 🚦 Run the app
if __name__ == '__main__':
    app.run(debug=True)
