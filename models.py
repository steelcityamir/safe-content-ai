"""Module providing base models."""

from pydantic import BaseModel


class ImageUrlsRequest(BaseModel):
    """
    Model representing the request body for the /v1/detect/urls endpoint.

    Attributes:
        urls (list[str]): List of image URLs to be processed.
    """

    urls: list[str]


class ImageDetectionResponse(BaseModel):
    """
    Base model representing the response body for image detection.

    Attributes:
        is_nsfw (bool): Whether the image is classified as NSFW.
        confidence_percentage (float): Confidence level of the NSFW classification.
    """

    is_nsfw: bool
    confidence_percentage: float


class FileImageDetectionResponse(ImageDetectionResponse):
    """
    Model extending ImageDetectionResponse with a file attribute.

    Attributes:
        file (str): The name of the file that was processed.
    """

    file_name: str


class UrlImageDetectionResponse(ImageDetectionResponse):
    """
    Model extending ImageDetectionResponse with a URL attribute.

    Attributes:
        url (str): The URL of the image that was processed.
    """

    url: str
