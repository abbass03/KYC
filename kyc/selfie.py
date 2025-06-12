from deepface import DeepFace
import numpy as np
import cv2
from kyc.utils import rotate_image, cleanup_temp_files
import concurrent.futures
from functools import partial
import os

def _process_selfie_rotation(img, angle, temp_dir):
    """Process a single rotation angle for selfie."""
    try:
        rotated = rotate_image(img, angle)
        temp_path = os.path.join(temp_dir, f"rotated_selfie_{angle}.jpg")
        cv2.imwrite(temp_path, rotated)
        
        faces = DeepFace.extract_faces(
            img_path=temp_path, 
            detector_backend='retinaface', 
            enforce_detection=True
        )
        if faces:
            face = faces[0]["face"]
            print(f"Face extracted at rotation {angle}. Shape: {face.shape}")
            
            if face.dtype != np.uint8:
                face = np.clip(face, 0, 1)
                face = (face * 255).astype(np.uint8)
            
            return face
        return None
    except Exception as e:
        print(f"[Error] Selfie face extraction failed for rotation {angle}: {e}")
        return None

def extract_face_from_selfie(image_path):
    """
    Extract the face from a selfie image using DeepFace with parallel processing.
    """
    temp_dir = os.path.join(os.path.dirname(image_path), 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"[Error] Failed to read selfie image: {image_path}")
            return None

        # Create a partial function with the image pre-loaded
        process_rotation = partial(_process_selfie_rotation, img, temp_dir=temp_dir)

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

        print("No face detected in any rotation.")
        return None

    except Exception as e:
        print(f"[Error] Selfie face extraction failed from {image_path}: {e}")
        return None
    finally:
        # Clean up temporary files
        cleanup_temp_files([os.path.join(temp_dir, f) for f in os.listdir(temp_dir)])
        try:
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"[Error] Error removing temp directory: {e}")