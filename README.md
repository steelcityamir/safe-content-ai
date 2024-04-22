# Safe Content AI
A fast reliable API for detecting NSFW images.

## Features

- Uses the [Falconsai/nsfw-image-detection](https://huggingface.co/Falconsai/nsfw_image_detection) AI model
- Caches predictions based on SHA-256 hash of image data



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


## API usage




## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub issue tracker for this project.

