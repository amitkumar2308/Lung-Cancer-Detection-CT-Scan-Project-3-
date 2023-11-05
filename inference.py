import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Define the classes
classes = [
    "Cancerous",
    "Cancerous",
    "NO Lung Cancer/ NORMAL",
    "Cancerous"
]

def load_model_and_predict(image_path):
    # Load the pre-trained model
    model = tf.keras.models.load_model('inception_chest.h5')

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)

    # Make predictions
    predictions = model.predict(x)
    predicted_class = classes[np.argmax(predictions)]

    return predicted_class
