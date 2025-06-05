import logging
from deepface import DeepFace
from config import API_CONFIG
from .utils import normalize_image
import numpy as np

logger = logging.getLogger(__name__)

class FaceMatcher:
    def __init__(self, threshold=None):
        self.threshold = threshold or API_CONFIG['FACE_MATCH_THRESHOLD']

    def compare_faces(self, face1, face2):
        """
        Compare two face images and return the distance score.
        
        Args:
            face1: First face image (numpy array or path)
            face2: Second face image (numpy array or path)
            
        Returns:
            float: Distance score between faces, or None if comparison fails
        """
        try:
            # Normalize images if they are numpy arrays
            if isinstance(face1, np.ndarray):
                face1 = normalize_image(face1)
            if isinstance(face2, np.ndarray):
                face2 = normalize_image(face2)

            result = DeepFace.verify(
                face1, 
                face2, 
                enforce_detection=False,
                detector_backend='retinaface'
            )
            
            return result['distance']
        except Exception as e:
            logger.error(f"Face comparison failed: {str(e)}")
            return None

    def is_match(self, face1, face2):
        """
        Check if two faces match based on the threshold.
        
        Args:
            face1: First face image
            face2: Second face image
            
        Returns:
            tuple: (bool, float) - (is_match, distance_score)
        """
        distance = self.compare_faces(face1, face2)
        if distance is None:
            return False, -1
        return distance < self.threshold, distance
