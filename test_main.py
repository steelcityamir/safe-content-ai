"""API Tests"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

FILE_NAME = "sunflower.jpg"


def test_read_main():
    """Tests that POST /api/v1/detect returns 200 OK with valid request body"""
    with open(FILE_NAME, "rb") as file:
        response = client.post(
            "/api/v1/detect",
            files={"file": (FILE_NAME, file, "image/jpeg")},
        )
    assert response.status_code == 200
    assert response.json() == {
        "file_name": FILE_NAME,
        "is_nsfw": False,
        "confidence_percentage": 100.0,
    }


def test_invalid_input():
    """Tests that POST /api/v1/detect returns 422 with empty request body"""
    response = client.post("/api/v1/detect", files={})
    assert response.status_code == 422
