<p align="center">
     <img src="https://github.com/steelcityamir/safe-content-ai/assets/54147931/95f56427-dd91-42f7-8d70-d0bed795e14b" alt="logo">
</p>

[![Python CI](https://github.com/steelcityamir/safe-content-ai/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/steelcityamir/safe-content-ai/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/steelcityamir/safe-content-ai/graph/badge.svg?token=RRZEJFKRG1)](https://codecov.io/gh/steelcityamir/safe-content-ai)


# Safe Content AI
A fast accurate API for detecting NSFW images.  Ideal for content moderation on digital platforms.

This project uses Python, FastAPI framework, Transformers library, and TensorFlow.  

TensorFlow will automatically detect and use the GPU if the underlying hardware supports it. 

## ⭐ Features

- Uses the [Falconsai/nsfw-image-detection](https://huggingface.co/Falconsai/nsfw_image_detection) AI model
- Caches results based on SHA-256 hash of image data


## 🐳 Quick Start using Docker

```bash
docker run -p 8000:8000 steelcityamir/safe-content-ai:latest
```

Test using curl

```bash
curl -X POST "http://127.0.0.1:8000/v1/detect" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/image.jpeg"
```

## Getting Started

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
The API server runs on port 8000.

## API usage

### POST /v1/detect

This endpoint allows users to upload an image file, which is then processed to determine if the content is NSFW (Not Safe For Work). The response includes whether the image is considered NSFW and the confidence level of the prediction.

#### Request

- **URL**: `/v1/detect`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Body**:
  - **file** (required): The image file to be classified.

#### Response

- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "file_name": "string",
    "is_nsfw": "boolean",
    "confidence_percentage": "number"
  }
  ```

#### Curl

```bash
curl -X POST "http://127.0.0.1:8000/v1/detect" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/image.jpeg"
```
### POST /v1/detect/urls

This endpoint allows users to provide image URLs, which are then processed to determine if the content is NSFW (Not Safe For Work). The response includes whether each image is considered NSFW and the confidence level of the prediction.

#### Request

- **URL**: `/v1/detect/urls`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Body**:
   ```json
  {
    "urls": [
      "https://example.com/image1.jpg",
      "https://example.com/image2.jpg",
      "https://example.com/image3.jpg",
      "https://example.com/image4.jpg",
      "https://example.com/image5.jpg"
    ]
  }
  ```

#### Response

- **Content-Type**: `application/json`
- **Body**:
  ```json
  [
    {
      "url": "string",
      "is_nsfw": "boolean",
      "confidence_percentage": "number"
    }
  ]
  ```

#### Curl

```bash
curl -X POST "http://127.0.0.1:8000/v1/detect/urls" \
     -H "Content-Type: application/json" \
     -d '{
           "urls": [
             "https://example.com/image1.jpg",
             "https://example.com/image2.jpg",
             "https://example.com/image3.jpg",
             "https://example.com/image4.jpg",
             "https://example.com/image5.jpg"
           ]
         }'
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub issue tracker for this project.

