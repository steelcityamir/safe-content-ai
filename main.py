from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline
from PIL import Image
from cachetools import Cache
import io, hashlib, logging

app = FastAPI()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Cache with no TTL
cache = Cache(maxsize=1000) 

# Load the model using the transformers pipeline
model = pipeline("image-classification", model="falconsai/nsfw_image_detection")

def hash_data(data):
    return hashlib.sha256(data).hexdigest()

@app.post("/api/v1/detect")
async def classify_image(file: UploadFile = File(...)):
    try:
        logging.info(f"Processing {file.filename}")
        # Read the image file
        image_data = await file.read()
        image_hash = hash_data(image_data)

        if image_hash in cache:
            # Return cached entry
            logging.info(f"Returning cached entry for {file.filename}")
            return JSONResponse(status_code=200, content=cache[image_hash])

        image = Image.open(io.BytesIO(image_data))

        # Use the model to classify the image
        results = model(image)

        # Find the prediction with the highest confidence using the max() function
        best_prediction = max(results, key=lambda x: x['score'])

        # Calculate the confidence score, rounded to the nearest tenth and as a percentage
        confidence_percentage = round(best_prediction['score'] * 100, 1)

        # Prepare the custom response data
        response_data = {
            "filename": file.filename,
            "isNsfw": best_prediction['label'] == 'nsfw',
            "confidence_percentage": confidence_percentage
        }

        # Populate hash
        cache[image_hash] = response_data

        return JSONResponse(status_code=200, content=response_data)

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
