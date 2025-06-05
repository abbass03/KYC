from dotenv import load_dotenv
load_dotenv()

import logging
import logging.config
from flask import Flask, request
from werkzeug.utils import secure_filename
import os

from config import (
    UPLOAD_FOLDER, API_CONFIG, LOG_CONFIG, 
    ALLOWED_EXTENSIONS, RESPONSE_MESSAGES, MAX_CONTENT_LENGTH
)
from kyc.face_match import FaceMatcher
from kyc.liveness import LivenessDetector
from kyc.document import DocumentProcessor
from kyc.utils import (
    allowed_file, save_uploaded_file, 
    save_debug_image, create_response
)

# Configure logging
logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Initialize KYC components
face_matcher = FaceMatcher()
liveness_detector = LivenessDetector()
document_processor = DocumentProcessor()

@app.route('/kyc', methods=['POST'])
def kyc_verification():
    """
    Main KYC verification endpoint that processes:
    1. Document verification (ID and passport)
    2. Face matching
    3. Liveness detection
    """
    try:
        # 1. Validate and save uploaded files
        files = validate_and_save_files(request)
        if not files:
            return create_response(
                False, 
                error=RESPONSE_MESSAGES['MISSING_FILE'].format('Required files'),
                status_code=400
            )

        # 2. Process documents
        document_results = process_documents(files)
        if not document_results['success']:
            return create_response(
                False,
                error=document_results['error'],
                status_code=400
            )

        # 3. Process face matching
        face_results = process_face_matching(files, document_results['faces'])
        if not face_results['success']:
            return create_response(
                False,
                error=face_results['error'],
                status_code=400
            )

        # 4. Process liveness
        liveness_results = process_liveness(files['video'])
        if not liveness_results['success']:
            return create_response(
                False,
                error=liveness_results['error'],
                status_code=400
            )

        # 5. Compile final results
        final_results = compile_results(
            document_results, 
            face_results, 
            liveness_results
        )

        return create_response(True, data=final_results)

    except Exception as e:
        logger.error(f"KYC verification failed: {str(e)}")
        return create_response(
            False,
            error=str(e),
            status_code=500
        )

def validate_and_save_files(request):
    """Validate and save uploaded files."""
    required_files = ['id', 'passport', 'selfie', 'video']
    saved_files = {}

    for file_key in required_files:
        if file_key not in request.files:
            logger.error(f"Missing file: {file_key}")
            return None
        
        file = request.files[file_key]
        if file.filename == '' or not allowed_file(file.filename):
            logger.error(f"Invalid file: {file_key}")
            return None

        try:
            saved_path = save_uploaded_file(file)
            saved_files[file_key] = saved_path
        except Exception as e:
            logger.error(f"Failed to save {file_key}: {str(e)}")
            return None

    return saved_files

def process_documents(files):
    """Process ID and passport documents."""
    try:
        # Extract faces from documents
        id_face = document_processor.extract_face(files['id'])
        passport_face = document_processor.extract_face(files['passport'])
        selfie_face = document_processor.extract_face(files['selfie'])

        # Save debug images
        save_debug_image(id_face, 'debug_id_face.jpg')
        save_debug_image(passport_face, 'debug_passport_face.jpg')
        save_debug_image(selfie_face, 'debug_selfie_face.jpg')

        # Process passport information
        passport_result = document_processor.process_passport(
            files['passport'],
            api_key=os.getenv('GOOGLE_API_KEY')
        )

        # Compare ID and passport documents
        document_comparison = document_processor.compare_documents(
            files['id'],
            files['passport']
        )

        if not document_comparison['success']:
            return {
                'success': False,
                'error': document_comparison['error']
            }

        return {
            'success': True,
            'faces': {
                'id': id_face,
                'passport': passport_face,
                'selfie': selfie_face
            },
            'passport_info': passport_result,
            'document_comparison': document_comparison['comparison']
        }

    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        return {
            'success': False,
            'error': RESPONSE_MESSAGES['EXTRACTION_ERROR'].format(str(e))
        }

def process_face_matching(files, faces):
    """Process face matching between documents and selfie."""
    try:
        # ID vs Selfie
        id_selfie_status, id_selfie_score = face_matcher.is_match(
            faces['id'], 
            faces['selfie']
        )

        # Passport vs Selfie
        passport_selfie_status, passport_selfie_score = face_matcher.is_match(
            faces['passport'], 
            faces['selfie']
        )

        return {
            'success': True,
            'matches': {
                'id_selfie': {
                    'status': id_selfie_status,
                    'score': id_selfie_score
                },
                'passport_selfie': {
                    'status': passport_selfie_status,
                    'score': passport_selfie_score
                }
            }
        }

    except Exception as e:
        logger.error(f"Face matching failed: {str(e)}")
        return {
            'success': False,
            'error': RESPONSE_MESSAGES['FACE_MATCH_FAILED'].format(str(e))
        }

def process_liveness(video_path):
    """Process liveness detection from video."""
    try:
        # Check liveness
        liveness_status = liveness_detector.check_liveness(video_path)

        # Extract face from video
        video_frame = liveness_detector.extract_face_frame_from_video(video_path)
        video_face = None
        if video_frame is not None:
            video_face = liveness_detector.extract_face_video(video_frame)
            save_debug_image(video_face, 'debug_video_face.jpg')

        return {
            'success': True,
            'status': liveness_status,
            'video_face': video_face
        }

    except Exception as e:
        logger.error(f"Liveness detection failed: {str(e)}")
        return {
            'success': False,
            'error': RESPONSE_MESSAGES['LIVENESS_FAILED']
        }

def compile_results(document_results, face_results, liveness_results):
    """Compile final verification results."""
    passport_info = document_results['passport_info']
    face_matches = face_results['matches']
    video_face = liveness_results.get('video_face')
    selfie_face = document_results['faces']['selfie']
    document_comparison = document_results['document_comparison']

    # Add video vs selfie comparison
    video_selfie_status, video_selfie_score = face_matcher.is_match(video_face, selfie_face) if (video_face is not None and selfie_face is not None) else (False, -1)

    # Check if all document fields match
    all_fields_match = all(
        field['match'] for field in document_comparison.values()
    ) if document_comparison else False

    # Determine overall status
    overall_status = "Verified" if all([
        passport_info['verification_result']['status'] == "ACCEPTED",
        face_matches['id_selfie']['status'],
        face_matches['passport_selfie']['status'],
        liveness_results['status'],
        video_selfie_status,
        all_fields_match
    ]) else "Rejected"

    return {
        "passport_info": passport_info,
        "document_comparison": document_comparison,
        "face_match": {
            **face_matches,
            "video_selfie": {
                "status": video_selfie_status,
                "score": video_selfie_score
            }
        },
        "liveness_detection": {
            "status": liveness_results['status']
        },
        "overall_status": overall_status
    }

if __name__ == '__main__':
    app.run(debug=True, threaded=False)
