import os
import cv2
import numpy as np
import pytesseract
from passporteye import read_mrz
from deepface import DeepFace

# Set tesseract path for both pytesseract and passporteye
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\\tessdata"

# Preprocessing for MRZ (optimized, keeps color)
def preprocess_for_mrz(image_path, output_path="static/uploads/preprocessed_mrz.jpg"):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return image_path  # fallback to original

    h = image.shape[0]
    # Try a larger crop (bottom 40%)
    mrz_region = image[int(h * 0.9):, :]

    # Enhance contrast using CLAHE
    lab = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    mrz_region = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    cv2.imwrite(output_path, mrz_region)
    return output_path

# MRZ extraction from passport image
def extract_mrz(image_path):
    print(f"Pytesseract command path set to: {pytesseract.pytesseract.tesseract_cmd}")
    preprocessed = preprocess_for_mrz(image_path)
    mrz = read_mrz(preprocessed)
    return mrz.to_dict() if mrz else None

# Extract face from any document (ID or passport)
def extract_face(image_path):
    """
    Extracts a face from an image using DeepFace.
    Normalizes and converts to uint8 if needed.
    Returns the face (numpy array) or None if no face detected.
    """
    try:
        # Extract faces
        faces = DeepFace.extract_faces(img_path=image_path, detector_backend='retinaface', enforce_detection=False, expand_percentage=5)

        if faces:
            face = faces[0]["face"]
            print(f"Face extracted. Shape: {face.shape}, Dtype: {face.dtype}, Range: {np.min(face)} to {np.max(face)}")

            # Normalize and convert to uint8 if needed
            if face.dtype != np.uint8:
                face = np.clip(face, 0, 1)
                face = (face * 255).astype(np.uint8)

            return face
        else:
            print("No face detected.")
            return None

    except Exception as e:
        print(f"[Error] Face extraction failed from {image_path}: {e}")
        return None
