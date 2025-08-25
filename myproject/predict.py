import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# تحميل النموذج
model = tf.keras.models.load_model('tiny_model.keras')

def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    preds = model.predict(x)
    return "Malignant" if preds[0] > 0.5 else "Benign"
