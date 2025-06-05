from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from kyc import document, selfie, face_match, liveness
from werkzeug.utils import secure_filename
from kyc.liveness import extract_face_frame_from_video, extract_face_video

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/kyc', methods=['POST'])
def kyc_verification():
    # 1. Check uploaded files
    required_files = ['id', 'passport', 'selfie', 'video']
    for file_key in required_files:
        if file_key not in request.files or request.files[file_key].filename == '':
            return jsonify({"error": f"Missing or empty file: {file_key}"}), 400

    # 2. Save uploaded files
    id_file = request.files['id']
    passport_file = request.files['passport']
    selfie_file = request.files['selfie']
    video_file = request.files['video']

    id_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(id_file.filename))
    passport_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(passport_file.filename))
    selfie_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(selfie_file.filename))
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))

    try:
        id_file.save(id_path)
        passport_file.save(passport_path)
        selfie_file.save(selfie_path)
        video_file.save(video_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save uploaded files: {e}"}), 500

    # 3. Extract MRZ and faces
    id_face = document.extract_face(id_path)
    passport_face = document.extract_face(passport_path)
    selfie_face = selfie.extract_face_from_selfie(selfie_path)

    # 4. Save debug images
    def save_debug(face, filename):
        if face is not None and face.shape[-1] == 3:
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename), face)

    save_debug(id_face, 'debug_id_face.jpg')
    save_debug(passport_face, 'debug_passport_face.jpg')
    save_debug(selfie_face, 'debug_selfie_face.jpg')

    # 5. Face matching
    face_match_threshold = 0.5

    # ID vs Selfie
    id_selfie_score = -1
    id_selfie_status = False
    if id_face is not None and selfie_face is not None:
        try:
            distance = face_match.compare_faces(id_face, selfie_face)
            if distance is not None:
                id_selfie_score = distance
                id_selfie_status = distance < face_match_threshold
        except Exception as e:
            print(f"ID vs Selfie match failed: {e}")
            id_selfie_status = False
            id_selfie_score = -1

    # Passport vs Selfie
    passport_selfie_score = -1
    passport_selfie_status = False
    if passport_face is not None and selfie_face is not None:
        try:
            distance = face_match.compare_faces(passport_face, selfie_face)
            if distance is not None:
                passport_selfie_score = distance
                passport_selfie_status = distance < face_match_threshold
        except Exception as e:
            print(f"Passport vs Selfie match failed: {e}")
            passport_selfie_status = False
            passport_selfie_score = -1

    # 6. Liveness
    try:
        liveness_status = liveness.check_liveness(video_path)
    except Exception as e:
        print(f"Liveness detection failed: {e}")
        liveness_status = False

    # 7. Extract frame from video and then extract face from that frame
    video_frame = extract_face_frame_from_video(video_path)
    save_debug(video_frame, 'debug_video_face.jpg')

    video_face = None
    if video_frame is not None:
        video_face = liveness.extract_face_video(video_frame)
        save_debug(video_face, 'debug_video_face_cropped.jpg')

    # 8. Compare video face with selfie face
    video_selfie_score = -1
    video_selfie_status = False
    if video_face is not None and selfie_face is not None:
        try:
            distance = face_match.compare_faces(video_face, selfie_face)
            if distance is not None:
                video_selfie_score = distance
                video_selfie_status = distance < face_match_threshold
        except Exception as e:
            print(f"Video vs Selfie match failed: {e}")
            video_selfie_status = False
            video_selfie_score = -1

    # 9. Status
    failed_matches = []
    if not id_selfie_status:
        failed_matches.append("ID vs Selfie")
    if not passport_selfie_status:
        failed_matches.append("Passport vs Selfie")
    if not video_selfie_status:
        failed_matches.append("Video vs Selfie")

    # 10. Passport info and verification (uses updated document.py)
    passport_result = document.extract_passport_info_and_verify(
        passport_path, 
        api_key="AIzaSyCvFefseo_KYbBy0EcDfMDLKFleqeDh57Q"
    )

    # Use the new passport_result for MRZ and visual info
    response_data = {
        "passport_mrz_info": passport_result.get("mrz_info"),
        "passport_visual_info": passport_result.get("visual_info"),
        "passport_mrz_vs_visual_checks": passport_result.get("mrz_vs_visual_checks"),
        "passport_mrz_checks": passport_result.get("mrz_checks"),
        "passport_trust_score_mrz": passport_result.get("trust_score_mrz"),
        "passport_trust_score_visual": passport_result.get("trust_score_visual"),
        "passport_trust_score_combined": passport_result.get("trust_score_combined"),
        "passport_verification_result": passport_result.get("verification_result"),
        "face_match": {
            "id_selfie": {
                "status": id_selfie_status,
                "score": id_selfie_score
            },
            "passport_selfie": {
                "status": passport_selfie_status,
                "score": passport_selfie_score
            },
            "video_selfie": {
                "status": video_selfie_status,
                "score": video_selfie_score
            }
        },
        "liveness_detection": {
            "status": liveness_status
        },
        "overall_status": (
            "Verified"
            if (
                passport_result.get("verification_result", {}).get("status") == "ACCEPTED"
                and id_selfie_status
                and passport_selfie_status
                and liveness_status
                and video_selfie_status
            )
            else (
                "Rejected - MRZ Extraction Failed" if passport_result.get("mrz_info") is None
                else "Rejected - Liveness Check Failed" if not liveness_status
                else "Rejected - Face Match Failed: " + ", ".join(failed_matches) if failed_matches
                else "Rejected"
            )
        )
    }

    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(debug=True, threaded=False)
