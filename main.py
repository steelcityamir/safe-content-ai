"""Module providing an API for NSFW image detection."""

import io
import hashlib
import logging
import aiohttp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline
from transformers.pipelines import PipelineException
from PIL import Image
from cachetools import Cache
from models import (
    FileImageDetectionResponse,
    UrlImageDetectionResponse,
    ImageUrlsRequest,
)
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

async def download_image(image_url: str) -> bytes:
    """Download an image from a URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status, detail="Image could not be retrieved."
                )
            return await response.read()


def hash_data(data):
    """Function for hashing image data."""
    return hashlib.sha256(data).hexdigest()


@app.post("/api/v1/detect", response_model=FileImageDetectionResponse)
async def classify_image(file: UploadFile = File(None)):
    """Function analyzing image."""
    if file is None:
        raise HTTPException(
            status_code=400,
            detail="An image file must be provided.",
        )

    try:
        logging.info("Processing %s", file.filename)

        # Read the image file
        image_data = await file.read()
        image_hash = hash_data(image_data)

        if image_hash in cache:
            # Return cached entry
            logging.info("Returning cached entry for %s", file.filename)

            cached_response = cache[image_hash]
            response_data = {**cached_response, "file_name": file.filename}

            return FileImageDetectionResponse(**response_data)

        image = Image.open(io.BytesIO(image_data))

        # Use the model to classify the image
        results = model(image)

        # Find the prediction with the highest confidence using the max() function
        best_prediction = max(results, key=lambda x: x["score"])

        # Calculate the confidence score, rounded to the nearest tenth and as a percentage
        confidence_percentage = round(best_prediction["score"] * 100, 1)

        # Prepare the custom response data
        response_data = {
            "is_nsfw": best_prediction["label"] == "nsfw",
            "confidence_percentage": confidence_percentage,
        }

        # Populate hash
        cache[image_hash] = response_data.copy()

        # Add file_name to the API response
        response_data["file_name"] = file.filename

        return FileImageDetectionResponse(**response_data)

    except PipelineException as e:
        logging.error("Error processing image: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/api/v1/detect/urls", response_model=list[UrlImageDetectionResponse])
async def classify_images(request: ImageUrlsRequest):
    """Function analyzing images from URLs."""
    response_data = []

    for image_url in request.urls:
        try:
            logging.info("Downloading image from URL: %s", image_url)
            image_data = await download_image(image_url)
            image_hash = hash_data(image_data)

            if image_hash in cache:
                # Return cached entry
                logging.info("Returning cached entry for %s", image_url)

                cached_response = cache[image_hash]
                response = {**cached_response, "url": image_url}

                response_data.append(response)
                continue

            image = Image.open(io.BytesIO(image_data))

            # Use the model to classify the image
            results = model(image)

            # Find the prediction with the highest confidence using the max() function
            best_prediction = max(results, key=lambda x: x["score"])

            # Calculate the confidence score, rounded to the nearest tenth and as a percentage
            confidence_percentage = round(best_prediction["score"] * 100, 1)

            # Prepare the custom response data
            detection_result = {
                "is_nsfw": best_prediction["label"] == "nsfw",
                "confidence_percentage": confidence_percentage,
            }

            # Populate hash
            cache[image_hash] = detection_result.copy()

            # Add url to the API response
            detection_result["url"] = image_url

            response_data.append(detection_result)

        except PipelineException as e:
            logging.error("Error processing image from %s: %s", image_url, str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Error processing image from {image_url}: {str(e)}",
            )

    return JSONResponse(status_code=200, content=response_data)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
