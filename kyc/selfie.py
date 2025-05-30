from deepface import DeepFace
import numpy as np
import cv2

def extract_face_from_selfie(image_path):
    """
    Extract the face from a selfie image using DeepFace.
    Normalizes and converts to uint8 if needed.
    Returns the cropped face image (numpy array) or None if not found.
    """
    try:
        # Directly pass the image path to DeepFace
        faces = DeepFace.extract_faces(img_path=image_path, detector_backend='retinaface', enforce_detection=True, expand_percentage=5)

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
        print(f"[Error] Selfie face extraction failed from {image_path}: {e}")
        return None
