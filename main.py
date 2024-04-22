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

app = FastAPI()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Cache with no TTL
cache = Cache(maxsize=1000)

# Load the model using the transformers pipeline
model = pipeline("image-classification", model="falconsai/nsfw_image_detection")


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


@app.post("/api/v1/detect")
async def classify_image(file: UploadFile = File(None), image_url: str = None):
    """Function analyzing image."""
    if file is None and image_url is None:
        raise HTTPException(
            status_code=400,
            detail="Either an image file or image URL must be provided.",
        )

    try:
        if file:
            logging.info("Processing %s", file.filename)
            image_data = await file.read()
            image_name = file.filename
        elif image_url:
            logging.info("Downloading image from URL")
            image_data = await download_image(image_url)
            image_name = image_url

        image_hash = hash_data(image_data)

        if image_hash in cache:
            # Return cached entry
            logging.info("Returning cached entry for %s", image_name)
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
            "source": image_name,
            "is_nsfw": best_prediction["label"] == "nsfw",
            "confidence_percentage": confidence_percentage,
        }

        # Populate hash
        cache[image_hash] = response_data

        return JSONResponse(status_code=200, content=response_data)

    except PipelineException as e:
        logging.error("Error processing image: %s", str(e))
        return JSONResponse(
            status_code=500,
            content={"message": "Error processing image", "error": str(e)},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
