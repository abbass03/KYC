import os
import cv2
import numpy as np
import pytesseract
from deepface import DeepFace
from kyc.Mrz_processing import process_passport
import logging
from .utils import normalize_image
import json
import re
from PIL import Image

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.detector_backend = 'retinaface'
        self.single_doc_prompt = None  # No LLM prompt needed

    def resize_image(self, input_path, output_path, max_width=1024):
        """Resize image if it's too large."""
        img = Image.open(input_path)
        if img.width > max_width:
            img = img.resize((max_width, int(img.height * max_width / img.width)))
            img.save(output_path)
            return output_path
        return input_path

    def load_image_as_part(self, path):
        """Load image as bytes and wrap for Gemini."""
        ext = os.path.splitext(path)[1].lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg"
        with open(path, "rb") as f:
            return {
                "mime_type": mime_type,
                "data": f.read()
            }

    def extract_fields_from_image(self, image_part, prompt_template):
        # No LLM extraction, return None or placeholder
        return None

    def normalize_value(self, val, field=None):
        """Normalize field values for comparison."""
        if isinstance(val, str):
            val = [val]
        elif not isinstance(val, list):
            val = list(val) if val is not None else []

        def norm(s):
            s = str(s).strip()
            s = s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
            labels = [
                "اسم الأب", "اسم و شهرة الام", "father name", "mother full name",
                "last name", "first name", "name", "الكنية", "الاسم الأول"
            ]
            for label in labels:
                if s.replace(" ", "").startswith(label.replace(" ", "")):
                    return None
            if field == "date_of_birth":
                if len(re.findall(r"\d", s)) >= 6:
                    return self.normalize_date(s)
                else:
                    return None
            if re.search(r"[a-zA-Z\u0600-\u06FF]", s):
                return s.lower().replace(" ", "")
            return None

        return [x for x in (norm(x) for x in val if x) if x]

    def normalize_date(self, s):
        """Normalize date format."""
        s = s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
        digits = re.findall(r"\d+", s)
        if len(digits) == 3:
            if len(digits[0]) == 4:  # YYYY MM DD
                return f"{digits[0]}{digits[1].zfill(2)}{digits[2].zfill(2)}"
            elif len(digits[2]) == 4:  # DD MM YYYY or MM DD YYYY
                return f"{digits[2]}{digits[1].zfill(2)}{digits[0].zfill(2)}"
            elif all(len(d) == 2 for d in digits):  # YY MM DD
                return f"20{digits[2]}{digits[1]}{digits[0]}"
        return "".join(digits)

    # Comment out compare_documents method and any call to it or use of document_comparison

    def extract_face(self, image_path):
        """
        Extract face from any document (ID or passport).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy.ndarray: The extracted face, or None if no face is detected
        """
        try:
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )

            if faces:
                face = faces[0]["face"]
                logger.info(f"Face extracted. Shape: {face.shape}")
                return normalize_image(face)
            
            logger.warning("No face detected in document")
            return None

        except Exception as e:
            logger.error(f"Face extraction failed: {str(e)}")
            return None

    def process_passport(self, image_path, api_key=None):
        """
        Process passport image to extract and verify information.
        
        Args:
            image_path: Path to the passport image
            api_key: API key for Gemini AI
            
        Returns:
            dict: Passport processing results
        """
        try:
            return process_passport(image_path)
        except Exception as e:
            logger.error(f"Passport processing failed: {str(e)}")
            return {
                "status": "REJECTED",
                "message": f"Failed to process passport: {str(e)}"
            }