import os
import cv2
import numpy as np
import pytesseract
from passporteye import read_mrz
from deepface import DeepFace
from kyc.Mrz_processing import process_passport
import logging
from .utils import normalize_image
import google.generativeai as genai
import json
import re
from PIL import Image

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.detector_backend = 'retinaface'
        self.single_doc_prompt = """
Extract the following fields from this identity document:
- First name
- Last name
- Father's full name
- Mother's full name
- Date of birth

If a field appears in two languages (e.g., Arabic and English), return both as a list.
Respond as a JSON object with keys: first_name, last_name, father_full_name, mother_full_name, date_of_birth.
Use the exact original spelling and formatting from the document. Do not translate or clean anything.
"""

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
        """Extract fields from an image using Gemini."""
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        response = model.generate_content([prompt_template, image_part], stream=False)
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text)
        try:
            return json.loads(raw_text)
        except Exception as e:
            logger.error(f"Failed to parse JSON from Gemini response: {str(e)}")
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

    def compare_documents(self, id_path, passport_path):
        """
        Compare ID and passport documents for field matching.
        
        Args:
            id_path: Path to ID document image
            passport_path: Path to passport document image
            
        Returns:
            dict: Comparison results for each field
        """
        try:
            # Resize images if needed
            id_path = self.resize_image(id_path, id_path)
            passport_path = self.resize_image(passport_path, passport_path)

            # Load images for Gemini
            id_image_part = self.load_image_as_part(id_path)
            passport_image_part = self.load_image_as_part(passport_path)

            # Extract fields from both documents
            id_info = self.extract_fields_from_image(id_image_part, self.single_doc_prompt)
            passport_info = self.extract_fields_from_image(passport_image_part, self.single_doc_prompt)

            if id_info is None or passport_info is None:
                return {
                    'success': False,
                    'error': 'Failed to extract information from one or both documents'
                }

            # Compare fields
            fields = ["first_name", "last_name", "father_full_name", "mother_full_name", "date_of_birth"]
            comparison = {}
            
            for field in fields:
                id_val = id_info.get(field)
                pass_val = passport_info.get(field)
                id_norm = self.normalize_value(id_val, field)
                pass_norm = self.normalize_value(pass_val, field)
                
                match = (
                    any(i in pass_norm for i in id_norm) or
                    any(i in id_norm for i in pass_norm)
                ) if id_norm and pass_norm else None
                
                comparison[field] = {
                    "id": id_val,
                    "passport": pass_val,
                    "match": match
                }

            return {
                'success': True,
                'comparison': comparison
            }

        except Exception as e:
            logger.error(f"Document comparison failed: {str(e)}")
            return {
                'success': False,
                'error': f'Document comparison failed: {str(e)}'
            }

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

    def process_passport(self, image_path, api_key):
        """
        Process passport image to extract and verify information.
        
        Args:
            image_path: Path to the passport image
            api_key: API key for Gemini AI
            
        Returns:
            dict: Passport processing results
        """
        try:
            return process_passport(image_path, api_key)
        except Exception as e:
            logger.error(f"Passport processing failed: {str(e)}")
            return {
                "status": "REJECTED",
                "message": f"Failed to process passport: {str(e)}"
            }