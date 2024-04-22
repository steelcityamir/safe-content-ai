from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

file_name = "sunflower.jpg"

def test_read_main():
    response = client.post("/api/v1/detect", files={"file": (file_name, open(file_name, "rb"), "image/jpeg")})
    assert response.status_code == 200
    assert response.json() == {"file_name": file_name, "is_nsfw": False, "confidence_percentage": 100.0}

def test_invalid_input():
    response = client.post("/api/v1/detect", files={})
    assert response.status_code == 422 

