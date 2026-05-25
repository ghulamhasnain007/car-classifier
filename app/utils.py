from PIL import Image

def load_image(file_bytes):
    return Image.open(file_bytes).convert("RGB")