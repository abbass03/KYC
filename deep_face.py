from deepface import DeepFace
import cv2
import numpy as np

image_path = "Testing_images/hadi.png"

# Extract faces
faces = DeepFace.extract_faces(img_path=image_path, detector_backend='retinaface', enforce_detection=False)

if faces:
    face = faces[0]["face"]
    print(f"Face extracted. Shape: {face.shape}, Dtype: {face.dtype}, Range: {np.min(face)} to {np.max(face)}")

    # Normalize and convert to uint8 if needed
    if face.dtype != np.uint8:
        face = np.clip(face, 0, 1)
        face = (face * 255).astype(np.uint8)

    output_path = "static/uploads/extracted_face.jpg"
    cv2.imwrite(output_path, face)
    print(f"Saved to {output_path}")
else:
    print("No face detected.")
