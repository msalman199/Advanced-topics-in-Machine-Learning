from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model
model = load_model('mnist_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the input data
    image = np.array(data['image'])  # The input image should be sent as a 28x28 pixel array
    image = image.reshape(1, 784)  # Reshape it to match the model's input shape
    image = image.astype('float32') / 255  # Normalize the image

    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    return jsonify({'prediction': int(predicted_class)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
