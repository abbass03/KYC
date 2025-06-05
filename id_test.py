# id_test.py
import cv2
import pytesseract
import re



import os
# Set tesseract path for both pytesseract and passporteye
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\\tessdata"


import cv2
import pytesseract
# ... rest of your code ...
# 1. Image Preprocessing
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return enhanced

# 2. OCR Extraction
def ocr_text(image):
    config = "--psm 6"
    return pytesseract.image_to_string(image, config=config, lang='ara')

# 3. Clean OCR Output
def clean_text(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines

# 4. Field Extraction (Regex/Heuristics)
def extract_id_fields_arabic(clean_lines):
    result = {
        "name": None,
        "id_number": None,
        "dob": None,
        "expiry": None,
        "nationality": None
    }
    for line in clean_lines:
        if not result["name"] and "الاسم" in line:
            result["name"] = line
        if not result["nationality"] and "الجنسية" in line:
            result["nationality"] = line
        if not result["dob"] and "تاريخ الميلاد" in line:
            result["dob"] = line
        if not result["expiry"] and "تاريخ الانتهاء" in line:
            result["expiry"] = line
        if not result["id_number"] and ("رقم" in line or "هوية" in line):
            result["id_number"] = line
    return result

if __name__ == "__main__":
    # Replace with your actual ID image path
    id_image_path = r"C:\Users\User\Desktop\KYC_Project\Testing_images\id_hadi.png"


    try:
        id_preprocessed = preprocess_image(id_image_path)
        id_ocr = ocr_text(id_preprocessed)
        id_clean = clean_text(id_ocr)
        print("Cleaned OCR Lines:")
        for line in id_clean:
            print(repr(line))
        id_fields = extract_id_fields_arabic(id_clean)
        print("Extracted ID Fields:", id_fields)
    except Exception as e:
        print(f"ID extraction error: {e}")
