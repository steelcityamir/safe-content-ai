# Safe Content AI
A fast reliable API for detecting NSFW images.

## Features

- Uses the [Falconsai/nsfw-image-detection](https://huggingface.co/Falconsai/nsfw_image_detection) AI model
- Caches predictions based on SHA-256 hash of image data



## Getting Started

### Run via Docker

```bash
docker run -p 8000:8000 codebyamir/safecontentai:latest
```

### Prerequisites

Ensure you have Python 3.7+ installed on your system.

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/steelcityamir/safe-content-ai.git
```

Navigate to the cloned directory:

```bash
cd safe-content-ai
```

Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

Install the required libraries using pip:

```bash
pip install -r requirements.txt
```


### Running the API

Start the API server from your command line:

```bash
uvicorn main:app --reload
```


## API usage

### POST /api/v1/detect

This endpoint allows users to upload an image file, which is then processed to determine if the content is NSFW (Not Safe For Work). The response includes whether the image is considered NSFW and the confidence level of the prediction.

#### Request

- **URL**: `/api/v1/detect`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Body**:
  - **file** (required): The image file to be classified.

#### Response

- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "filename": "string",
    "isNsfw": "boolean",
    "confidence_percentage": "number"
  }

#### Curl

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/detect" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/image.jpeg"
```



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub issue tracker for this project.

