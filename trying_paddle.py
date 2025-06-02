import cv2
import numpy as np
from paddleocr import PaddleOCR
from passporteye.mrz.text import MRZ

def preprocess_mrz_image(image):
    """Enhanced preprocessing for MRZ region"""
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(enhanced)

    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    return binary

def extract_mrz_with_paddleocr(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print("Failed to read image")
            return None

        height, width = image.shape[:2]
        
        mrz_regions = [
            image[int(height * 0.8):, :],
            image[int(height * 0.75):, :],
            image[int(height * 0.7):, :]
        ]

        # Updated line: Removed unsupported arguments
        ocr = PaddleOCR(lang='en')
        
        best_result = None
        best_confidence = 0

        for i, region in enumerate(mrz_regions):
            print(f"Trying region {i+1}...")
            
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            result = ocr.ocr(enhanced)
            
            if result and result[0]:
                confidence = sum(line[1][1] for line in result[0]) / len(result[0])
                print(f"Region {i+1} confidence: {confidence}")
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = result[0]

        if not best_result:
            print("No text detected by PaddleOCR in any region.")
            return None

        extracted_text = '\n'.join([line[1][0] for line in best_result])
        print("\nExtracted Text:", extracted_text)
        
        try:
            mrz = MRZ(extracted_text)
            if mrz.valid:
                return {
                    'mrz_data': mrz.to_dict(),
                    'confidence': mrz.valid_score,
                    'raw_text': extracted_text
                }
        except Exception as e:
            print(f"MRZ parsing error: {e}")

        return {
            'raw_text': extracted_text,
            'confidence': best_confidence
        }

    except Exception as e:
        print(f"Error in MRZ extraction: {e}")
        return None

def validate_mrz_data(mrz_data):
    if not mrz_data:
        return False
    
    required_fields = ['number', 'type', 'country', 'names', 'surname', 'date_of_birth']
    if 'mrz_data' in mrz_data:
        for field in required_fields:
            if field not in mrz_data['mrz_data']:
                return False
    
    if mrz_data.get('confidence', 0) < 0.5:
        return False
    
    return True

# Example usage
if __name__ == "__main__":
    image_path = r'C:\Users\User\Desktop\KYC_Project\Testing_images\hadi_passport.png'
    
    mrz_data = extract_mrz_with_paddleocr(image_path)
    
    if mrz_data:
        if 'mrz_data' in mrz_data:
            print("\nMRZ Data:", mrz_data['mrz_data'])
            print("Confidence:", mrz_data['confidence'])
        else:
            print("\nRaw Text:", mrz_data['raw_text'])
            print("Confidence:", mrz_data['confidence'])
    else:
        print("Failed to extract MRZ data")
