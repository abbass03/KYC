import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
import logging
from config import API_CONFIG

logger = logging.getLogger(__name__)

class FaceMatcher:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.threshold = API_CONFIG['FACE_MATCH_THRESHOLD']
    
    def is_match(self, face1, face2, threshold=None):
        """
        Compare two face images and determine if they match using DeepFace.
        
        Args:
            face1: First face image (numpy array)
            face2: Second face image (numpy array)
            threshold: Optional threshold override (defaults to config value)
            
        Returns:
            tuple: (is_match: bool, confidence_score: float)
        """
        try:
            # Use threshold from config if not specified
            if threshold is None:
                threshold = API_CONFIG['FACE_MATCH_THRESHOLD']
            
            # Input validation
            if face1 is None or face2 is None:
                logger.error("One or both face images are None")
                return False, 0.0
                
            # Convert images to RGB if they're not already
            if len(face1.shape) == 2:  # If grayscale
                face1 = cv2.cvtColor(face1, cv2.COLOR_GRAY2RGB)
            if len(face2.shape) == 2:  # If grayscale
                face2 = cv2.cvtColor(face2, cv2.COLOR_GRAY2RGB)
            
            # Use DeepFace to verify faces
            result = DeepFace.verify(
                face1, 
                face2, 
                model_name="ArcFace",  # You can also use "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"
                detector_backend="retinaface",  # You can also use "ssd", "dlib", "mtcnn", "retinaface"
                enforce_detection=True,
                silent=True
            )
            
            # Get similarity score
            similarity_score = result['distance']  # DeepFace returns distance, convert to similarity
            is_match = result['verified']
            
            logger.info(f"Face match result: {is_match} (score: {similarity_score:.2f})")
            return is_match, similarity_score
            
        except Exception as e:
            logger.error(f"Error in face matching: {str(e)}")
            return False, 0.0