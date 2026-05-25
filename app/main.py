# from fastapi import FastAPI, UploadFile, File
# from app.predict import predict_image
# from PIL import Image
# import io

# app = FastAPI(title="Car Classifier API")

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

#     result = predict_image(image)

#     return result

from fastapi import FastAPI, Request
from PIL import Image
import io
from app.predict import predict_image

app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    try:
        image_bytes = await request.body()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        result = predict_image(image)

        return {
            "car_type": result
        }
    except Exception as e:
        return {
            "error": str(e)
        }
    