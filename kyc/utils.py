import os
import logging
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS
import datetime
import imghdr
import mimetypes

logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file, filename=None):
    """Save an uploaded file securely."""
    if filename is None:
        filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        file.save(file_path)
        return file_path
    except Exception as e:
        logger.error(f"Error saving file {filename}: {str(e)}")
        raise

def save_debug_image(image, filename):
    """Save a debug image if it's valid."""
    if image is not None and isinstance(image, np.ndarray) and image.shape[-1] == 3:
        try:
            debug_path = os.path.join(UPLOAD_FOLDER, filename)
            cv2.imwrite(debug_path, image)
            return debug_path
        except Exception as e:
            logger.error(f"Error saving debug image {filename}: {str(e)}")
    return None

def normalize_image(image):
    """Normalize image to uint8 format."""
    if image is None:
        return None
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
    return image

def create_response(success, data=None, error=None, status_code=200):
    """Create a standardized API response."""
    response = {
        'success': success,
        'timestamp': datetime.datetime.now(datetime.UTC).isoformat()
    }
    if data is not None:
        response['data'] = data
    if error is not None:
        response['error'] = error
    return response, status_code

def is_image_file(filepath):
    """Check if file is a valid image (jpeg or png)."""
    return imghdr.what(filepath) in ['jpeg', 'png']

def is_video_file(filepath):
    """Check if file is a valid video by MIME type and frame count."""
    mime = mimetypes.guess_type(filepath)[0]
    if mime and mime.startswith('video'):
        # Extra check: try to open with OpenCV
        try:
            cap = cv2.VideoCapture(filepath)
            ret, _ = cap.read()
            cap.release()
            return ret  # True if at least one frame can be read
        except Exception:
            return False
    return False
