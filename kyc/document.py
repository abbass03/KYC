import os
import cv2
import numpy as np
import pytesseract
from deepface import DeepFace
from kyc.Mrz_processing import process_passport
import logging
from .utils import normalize_image, cleanup_temp_files
import json
import re
from PIL import Image
from kyc.utils import rotate_image
import concurrent.futures
from functools import partial

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

    def _process_rotation(self, img, angle, temp_dir):
        """Process a single rotation angle."""
        try:
            rotated = rotate_image(img, angle)
            temp_path = os.path.join(temp_dir, f"rotated_{angle}.jpg")
            cv2.imwrite(temp_path, rotated)
            
            faces = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            if faces:
                face = faces[0]["face"]
                logger.info(f"Face extracted at rotation {angle}. Shape: {face.shape}")
                return normalize_image(face)
            return None
        except Exception as e:
            logger.error(f"Face extraction failed for rotation {angle}: {str(e)}")
            return None

    def extract_face(self, image_path):
        """
        Extract face from any document (ID or passport) using parallel processing.
        """
        temp_dir = os.path.join(os.path.dirname(image_path), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_files = []

        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error("Failed to read image")
                return None

            # Create a partial function with the image pre-loaded
            process_rotation = partial(self._process_rotation, img, temp_dir=temp_dir)

            # Process rotations in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_angle = {
                    executor.submit(process_rotation, angle): angle 
                    for angle in [0, 90, 180, 270]
                }
                
                # Get the first successful result
                for future in concurrent.futures.as_completed(future_to_angle):
                    result = future.result()
                    if result is not None:
                        return result

            logger.warning("No face detected in any rotation")
            return None

        except Exception as e:
            logger.error(f"Face extraction failed: {str(e)}")
            return None
        finally:
            # Clean up temporary files
            cleanup_temp_files([os.path.join(temp_dir, f) for f in os.listdir(temp_dir)])
            try:
                os.rmdir(temp_dir)
            except Exception as e:
                logger.error(f"Error removing temp directory: {str(e)}")

    def _process_passport_rotation(self, img, angle, temp_dir):
        """Process passport for a single rotation angle."""
        try:
            rotated = rotate_image(img, angle)
            temp_path = os.path.join(temp_dir, f"rotated_passport_{angle}.jpg")
            cv2.imwrite(temp_path, rotated)
            result = process_passport(temp_path)
            if result and result.get("status") == "SUCCESS":
                return result
            return None
        except Exception as e:
            logger.error(f"Passport processing failed for rotation {angle}: {str(e)}")
            return None

    def process_passport(self, image_path, api_key=None):
        """
        Process passport image to extract and verify information using parallel processing.
        """
        temp_dir = os.path.join(os.path.dirname(image_path), 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error("Failed to read passport image")
                return {
                    "status": "REJECTED",
                    "message": "Failed to read passport image"
                }

            # Create a partial function with the image pre-loaded
            process_rotation = partial(self._process_passport_rotation, img, temp_dir=temp_dir)

            # Process rotations in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_angle = {
                    executor.submit(process_rotation, angle): angle 
                    for angle in [0, 90, 180, 270]
                }
                
                # Get the first successful result
                for future in concurrent.futures.as_completed(future_to_angle):
                    result = future.result()
                    if result is not None:
                        return result

            logger.error("Passport processing failed for all rotations")
            return {
                "status": "REJECTED",
                "message": "Failed to process passport in any orientation"
            }

        except Exception as e:
            logger.error(f"Passport processing failed: {str(e)}")
            return {
                "status": "REJECTED",
                "message": f"Failed to process passport: {str(e)}"
            }
        finally:
            # Clean up temporary files
            cleanup_temp_files([os.path.join(temp_dir, f) for f in os.listdir(temp_dir)])
            try:
                os.rmdir(temp_dir)
            except Exception as e:
                logger.error(f"Error removing temp directory: {str(e)}")