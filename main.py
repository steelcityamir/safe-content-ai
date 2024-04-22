"""Module providing an API for NSFW image detection."""

import io
import hashlib
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline
from transformers.pipelines import PipelineException
from PIL import Image
from cachetools import Cache
import tensorflow as tf


app = FastAPI()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Cache with no TTL
cache = Cache(maxsize=1000)

# Load the model using the transformers pipeline
model = pipeline("image-classification", model="falconsai/nsfw_image_detection")

# Detect the device used by TensorFlow
DEVICE = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
logging.info("TensorFlow version: %s", tf.__version__)
logging.info("Model is using: %s", DEVICE)

if DEVICE == "GPU":
    logging.info("GPUs available: %d", len(tf.config.list_physical_devices("GPU")))

def hash_data(data):
    """Function for hashing image data."""
    return hashlib.sha256(data).hexdigest()


@app.post("/api/v1/detect")
async def classify_image(file: UploadFile = File(...)):
    """Function analyzing image."""
    try:
        logging.info("Processing %s", file.filename)
        # Read the image file
        image_data = await file.read()
        image_hash = hash_data(image_data)

        if image_hash in cache:
            # Return cached entry
            logging.info("Returning cached entry for %s", file.filename)
            return JSONResponse(status_code=200, content=cache[image_hash])

        image = Image.open(io.BytesIO(image_data))

        # Use the model to classify the image
        results = model(image)

        # Find the prediction with the highest confidence using the max() function
        best_prediction = max(results, key=lambda x: x["score"])

        # Calculate the confidence score, rounded to the nearest tenth and as a percentage
        confidence_percentage = round(best_prediction["score"] * 100, 1)

        # Prepare the custom response data
        response_data = {
            "file_name": file.filename,
            "is_nsfw": best_prediction["label"] == "nsfw",
            "confidence_percentage": confidence_percentage,
        }

        # Populate hash
        cache[image_hash] = response_data

        return JSONResponse(status_code=200, content=response_data)

    except PipelineException as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
