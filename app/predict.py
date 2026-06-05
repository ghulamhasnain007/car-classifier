import numpy as np
from PIL import Image
from app.model import get_model

classes = ["700CC", "1200CC"]

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = (image_array - mean) / std
    image_array = np.transpose(image_array, (2, 0, 1))
    return np.expand_dims(image_array, axis=0).astype(np.float32)


def predict_image(image: Image.Image):
    session = get_model()
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: preprocess_image(image)})
    predicted = int(np.argmax(outputs[0], axis=1)[0])

    return classes[predicted]
