import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

def check_liveness(video_path, min_frames_with_face=15, min_movement=8, min_smile_frames=4):
    """
    Returns True if BOTH head movement and smile are detected in a live video.
    Returns False otherwise.
    """
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(video_path)
    frames_with_face = 0
    smile_frames = 0
    nose_positions = []

    total_frames = 0

    with mp_face_mesh.FaceMesh(static_image_mode=False) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                frames_with_face += 1
                landmarks = results.multi_face_landmarks[0].landmark
                nose = landmarks[1]
                nose_positions.append((nose.x, nose.y))
                left_mouth = landmarks[61]
                right_mouth = landmarks[291]
                top_lip = landmarks[13]
                bottom_lip = landmarks[14]
                mouth_width = np.linalg.norm(np.array([left_mouth.x, left_mouth.y]) - np.array([right_mouth.x, right_mouth.y]))
                mouth_height = np.linalg.norm(np.array([top_lip.x, top_lip.y]) - np.array([bottom_lip.x, bottom_lip.y]))
                if mouth_height > 0:
                    mouth_ratio = mouth_width / mouth_height
                    if mouth_ratio > 1.5:
                        smile_frames += 1
        cap.release()

    if frames_with_face < min_frames_with_face:
        print("Liveness failed: Not enough frames with a face.")
        return False

    if len(nose_positions) > 1:
        nose_positions = np.array(nose_positions)
        movement_x = np.ptp(nose_positions[:, 0])
        movement_y = np.ptp(nose_positions[:, 1])
        # Require both X and Y movement to be above threshold, or sum
        moved_enough = (movement_x + movement_y) > (min_movement / 100)
    else:
        moved_enough = False

    smiled_enough = smile_frames >= min_smile_frames

    print("Total frames:", total_frames)
    print("Frames with face:", frames_with_face)
    print("Head movement (x, y):", movement_x, movement_y)
    print("Moved enough:", moved_enough)
    print("Smile frames:", smile_frames)
    print("Smiled enough:", smiled_enough)

    if not moved_enough:
        print("Liveness failed: Not enough head movement.")
    if not smiled_enough:
        print("Liveness failed: Not enough smile frames.")

    return bool(smiled_enough and moved_enough)

def extract_face_frame_from_video(video_path, detector_backend='retinaface'):
    """
    Extracts the first frame from the video where a face is detected.
    Returns the cropped face image (numpy array) or None if not found.
    """
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(video_path)
    face_img = None

    with mp_face_mesh.FaceMesh(static_image_mode=False) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                # Optionally, crop the face region using landmarks
                # For simplicity, return the whole frame where a face is detected
                face_img = frame
                break

        cap.release()

    return face_img

def extract_face_video(face_image):
    """
    Extracts a face from an image using DeepFace.
    Normalizes and converts to uint8 if needed.
    Returns the face (numpy array) or None if no face detected.
    """
    try:
        # Extract faces
        faces = DeepFace.extract_faces(img_path=face_image, detector_backend='retinaface', enforce_detection=False, expand_percentage=5)

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
        print(f"[Error] Face extraction failed from {face_image}: {e}")
        return None



