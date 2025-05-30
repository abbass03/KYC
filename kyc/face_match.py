from deepface import DeepFace

def compare_faces(face1_path, face2_path):
    """
    Compare two face images and return True if they match, False otherwise.
    """
    result = DeepFace.verify(face1_path, face2_path, enforce_detection=False)
    return result['distance']
