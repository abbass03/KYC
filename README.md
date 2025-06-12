# KYC Verification System

A robust Know Your Customer (KYC) verification system that performs document verification, face matching, and liveness detection.

## Features

- **Document Processing**
  - ID and Passport verification
  - MRZ (Machine Readable Zone) extraction and validation
  - PDF support for document uploads
  - Automatic document rotation detection

- **Face Matching**
  - Face extraction from documents and selfies
  - Face comparison between documents and selfie
  - Parallel processing for faster face detection
  - Support for multiple face orientations

- **Liveness Detection**
  - Video-based liveness verification
  - Head movement detection
  - Smile detection
  - Anti-spoofing measures

## Prerequisites

- Python 3.8+
- OpenCV
- DeepFace
- EasyOCR
- MediaPipe
- Flask
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/abbass03/KYC.git
cd kyc-project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The system can be configured through `config.py`. Key configuration options include:

- Face matching threshold
- Liveness detection parameters
- File upload settings
- Logging configuration

## Usage

1. Start the server:
```bash
python app.py
```

2. The API will be available at `http://localhost:5000`

3. Send a POST request to `/kyc` with the following files:
   - ID document (image or PDF)
   - Passport (image or PDF)
   - Selfie image
   - Liveness video

Example request:
```python
import requests

files = {
    'id': open('id_document.jpg', 'rb'),
    'passport': open('passport.pdf', 'rb'),
    'selfie': open('selfie.jpg', 'rb'),
    'video': open('liveness.mp4', 'rb')
}

response = requests.post('http://localhost:5000/kyc', files=files)
print(response.json())
```

## API Response

The API returns a JSON response with the following structure:

```json
{
    "success": true,
    "data": {
        "passport_info": {
            "status": "SUCCESS",
            "mrz_info": {
                // MRZ extracted information
            }
        },
        "face_match": {
            "id_selfie": {
                "status": true,
                "score": 0.95
            },
            "passport_selfie": {
                "status": true,
                "score": 0.92
            },
            "video_selfie": {
                "status": true,
                "score": 0.90
            }
        },
        "liveness_detection": {
            "status": true
        },
        "overall_status": "Verified"
    }
}
```

## Project Structure 

## System Dependencies



1. **Poppler for Windows**
   - Download from: https://github.com/oschwartz10612/poppler-windows/releases/
   - Extract to a location (e.g., `C:\Program Files\poppler`)
   - Add the `bin` directory to your system PATH
   - Example: `C:\Program Files\poppler\bin`

2. **OpenCV Dependencies**
   - Visual C++ Redistributable for Visual Studio 2015-2022
   - Download from: https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist

### Linux (Ubuntu/Debian)
```bash
# Install Tesseract OCR
sudo apt-get update
sudo apt-get install tesseract-ocr

# Install Poppler
sudo apt-get install poppler-utils

# Install OpenCV dependencies
sudo apt-get install libsm6 libxext6 libxrender-dev
```

### macOS
```bash
# Using Homebrew
brew install tesseract
brew install poppler
```

## Environment Setup

1. Set environment variables for system dependencies:

### Windows
```batch
set POPPLER_PATH=C:\Program Files\poppler\bin
```



2. Verify installations:
```bash


# Check Poppler
pdfinfo -v

# Check OpenCV
python -c "import cv2; print(cv2.__version__)"
```

## Common Issues and Solutions

1. **PDF Processing Errors**
   - Error: "poppler not installed"
   - Solution: Install Poppler and ensure it's in your system PATH

2. **OpenCV Errors**
   - Error: "DLL load failed"
   - Solution: Install Visual C++ Redistributable

3. **Permission Issues**
   - Error: "Permission denied"
   - Solution: Run as administrator or check file permissions

## Development Environment Setup

1. **IDE Configuration**
   - VS Code: Install Python extension
   - PyCharm: Configure Python interpreter

2. **Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. **Testing Environment**
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest
```

## Troubleshooting

1. **PDF Processing**
   - If PDF conversion fails, check Poppler installation
   - Verify PDF file is not corrupted
   - Check file permissions

2. **OCR Issues**
   - Ensure EasyOCR is properly installed
   - Check language data files
   - Verify image quality

3. **Face Detection**
   - Check OpenCV installation
   - Verify image format and quality
   - Check system memory usage

4. **Performance Issues**
   - Monitor system resources
   - Check for memory leaks
   - Verify disk space

## Additional Resources

- [Poppler Documentation](https://poppler.freedesktop.org/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [DeepFace Documentation](https://github.com/serengil/deepface) 