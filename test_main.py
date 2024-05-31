"""API Tests"""

import hashlib
from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app, cache

client = TestClient(app)

FILE_NAME = "sunflower.jpg"
IMAGE_HASH = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def compute_file_hash(file_path):
    """Utility function to compute SHA-256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def test_read_main():
    """Tests that POST /v1/detect returns 200 OK with valid request body"""
    with open(FILE_NAME, "rb") as file:
        response = client.post(
            "/v1/detect",
            files={"file": (FILE_NAME, file, "image/jpeg")},
        )
    assert response.status_code == 200
    assert response.json() == {
        "file_name": FILE_NAME,
        "is_nsfw": False,
        "confidence_percentage": 100.0,
    }


def test_invalid_input():
    """Tests that POST /api/v1/detect returns 400 with empty request body"""
    response = client.post("/api/v1/detect", files={})
    assert response.status_code == 400


def test_cache_hit():
    """Tests that the endpoint returns a cached response when an image hash matches"""
    # Compute the hash of the test file
    image_hash = compute_file_hash(FILE_NAME)
    cached_response = {
        "file_name": FILE_NAME,
        "is_nsfw": False,
        "confidence_percentage": 100.0,
    }

    with patch.dict(cache, {image_hash: cached_response}), patch(
        "main.logging.info"
    ) as mock_logging:
        # Post the file to the endpoint as would happen in actual use
        with open(FILE_NAME, "rb") as file:
            response = client.post(
                "/v1/detect",
                files={"file": (FILE_NAME, file, "image/jpeg")},
            )

        # Verify that the response is from the cache
        assert response.status_code == 200
        assert response.json() == cached_response

        # Ensure logging was called correctly
        mock_logging.assert_called_with("Returning cached entry for %s", FILE_NAME)

def test_detect_urls():
    """Tests that POST /api/v1/detect/urls returns 200 OK with valid request body"""
    urls = [
        "https://raw.githubusercontent.com/steelcityamir/safe-content-ai/main/sunflower.jpg",
    ]
    response = client.post("/api/v1/detect/urls", json={"urls": urls})
    assert response.status_code == 200
    assert len(response.json()) == len(urls)
    assert response.json()[0] == {
        "url": urls[0],
        "is_nsfw": False,
        "confidence_percentage": 100.0,
    }
