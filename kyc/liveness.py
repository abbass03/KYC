import logging
import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from config import API_CONFIG
from .utils import normalize_image

logger = logging.getLogger(__name__)
class LivenessDetector:
    def __init__(self, config=None):
        self.config = config or {
            'min_frames_with_face': API_CONFIG['LIVENESS_MIN_FRAMES_WITH_FACE'],
            'min_movement': API_CONFIG['LIVENESS_MIN_MOVEMENT'],
            'min_smile_frames': API_CONFIG['LIVENESS_MIN_SMILE_FRAMES']
        }
        self.mp_face_mesh = mp.solutions.face_mesh

    def check_liveness(self, video_path):
        """
        Check if the video shows a live person by detecting head movement and smile.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            bool: True if liveness is detected, False otherwise
        """
        try:
            cap = cv2.VideoCapture(video_path)
            frames_with_face = 0
            smile_frames = 0
            nose_positions = []
            total_frames = 0

            with self.mp_face_mesh.FaceMesh(static_image_mode=False) as face_mesh:
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
                        
                        # Track nose position
                        nose = landmarks[1]
                        nose_positions.append((nose.x, nose.y))
                        
                        # Check for smile
                        if self._detect_smile(landmarks):
                            smile_frames += 1
                            
            cap.release()
            
            # Log detection results
            self._log_detection_results(
                total_frames, frames_with_face, 
                nose_positions, smile_frames
            )
            
            # Check if all criteria are met
            return self._evaluate_liveness(
                frames_with_face, nose_positions, smile_frames
            )
            
        except Exception as e:
            logger.error(f"Liveness detection failed: {str(e)}")
            return False

    def _detect_smile(self, landmarks):
        """Detect smile using mouth landmarks."""
        try:
            left_mouth = landmarks[61]
            right_mouth = landmarks[291]
            top_lip = landmarks[13]
            bottom_lip = landmarks[14]
            
            mouth_width = np.linalg.norm(
                np.array([left_mouth.x, left_mouth.y]) - 
                np.array([right_mouth.x, right_mouth.y])
            )
            mouth_height = np.linalg.norm(
                np.array([top_lip.x, top_lip.y]) - 
                np.array([bottom_lip.x, bottom_lip.y])
            )
            
            if mouth_height > 0:
                mouth_ratio = mouth_width / mouth_height
                return mouth_ratio > 1.5
            return False
            
        except Exception as e:
            logger.error(f"Smile detection failed: {str(e)}")
            return False

    def _evaluate_liveness(self, frames_with_face, nose_positions, smile_frames):
        """Evaluate if the video meets liveness criteria."""
        if frames_with_face < self.config['min_frames_with_face']:
            logger.warning("Not enough frames with a face")
            return False

        if len(nose_positions) > 1:
            nose_positions = np.array(nose_positions)
            movement_x = np.ptp(nose_positions[:, 0])
            movement_y = np.ptp(nose_positions[:, 1])
            moved_enough = (movement_x + movement_y) > (self.config['min_movement'] / 100)
        else:
            moved_enough = False

        smiled_enough = smile_frames >= self.config['min_smile_frames']
        
        return bool(smiled_enough and moved_enough)

    def _log_detection_results(self, total_frames, frames_with_face, nose_positions, smile_frames):
        """Log the results of liveness detection."""
        if len(nose_positions) > 1:
            nose_positions = np.array(nose_positions)
            movement_x = np.ptp(nose_positions[:, 0])
            movement_y = np.ptp(nose_positions[:, 1])
        else:
            movement_x = movement_y = 0

        logger.info(f"Liveness detection results:")
        logger.info(f"Total frames: {total_frames}")
        logger.info(f"Frames with face: {frames_with_face}")
        logger.info(f"Head movement (x, y): {movement_x:.3f}, {movement_y:.3f}")
        logger.info(f"Smile frames: {smile_frames}")

    def extract_face_frame_from_video(self, video_path):
        """
        Extract the first frame from the video where a face is detected.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            numpy.ndarray: The frame with a detected face, or None if no face is found
        """
        try:
            cap = cv2.VideoCapture(video_path)
            face_img = None

            with self.mp_face_mesh.FaceMesh(static_image_mode=False) as face_mesh:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)

                    if results.multi_face_landmarks:
                        face_img = frame
                        break

            cap.release()
            return face_img

        except Exception as e:
            logger.error(f"Face frame extraction failed: {str(e)}")
            return None

    def extract_face_video(self, face_image):
        """
        Extract a face from an image using DeepFace.
        
        Args:
            face_image: Image containing a face
            
        Returns:
            numpy.ndarray: The extracted face, or None if no face is detected
        """
        try:
            faces = DeepFace.extract_faces(
                img_path=face_image,
                detector_backend='retinaface',
                enforce_detection=False,
                expand_percentage=5
            )

            if faces:
                face = faces[0]["face"]
                return normalize_image(face)
            
            logger.warning("No face detected in video frame")
            return None

        except Exception as e:
            logger.error(f"Face extraction failed: {str(e)}")
            return None



