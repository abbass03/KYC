from google import generativeai as genai
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_document_info(image_path: str, api_key: str, doc_type: str = "document") -> dict:
    """
    Extract information from a document (ID card, passport, etc.) using Gemini AI.
    Args:
        image_path (str): Path to the document image
        api_key (str): Google Gemini API key
        doc_type (str): Type of document ("ID card", "passport", etc.)
    Returns:
        dict: Extracted information from the document
    """
    try:
        genai.configure(api_key=api_key)
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        prompt_text = (
            f"You are an expert document information extractor. Your task is to extract all available information from this {doc_type} image.\n"
            "Important instructions:\n"
            "1. Extract information in the original language it appears on the document\n"
            "2. Look for and extract any visible information including but not limited to:\n"
            "   - Full Name\n"
            "   - Date of Birth\n"
            "   - Document/ID/Passport Number\n"
            "   - Expiry Date\n"
            "   - Nationality\n"
            "   - Gender\n"
            "   - Address\n"
            "   - Issuing Authority\n"
            "   - Any other visible information\n"
            "3. If you find additional fields not listed above, include them in the response\n"
            "4. If any information is not visible or unclear, mark it as 'Not Available'\n"
            "5. Format the response as a JSON object with all extracted fields\n"
            "6. Preserve the original language of the extracted text\n"
            "7. If the document is in a non-Latin script, provide the text in its original script"
        )
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            contents=[{
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}}
                ]
            }]
        )
        
        # Parse the response and convert to dictionary
        try:
            response_text = response.text.strip()
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            doc_info = json.loads(response_text)
            return doc_info
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Error in document information extraction: {str(e)}")
        return None

def normalize_key(key):
    """
    Normalize keys to a standard set for comparison.
    """
    key = key.lower().replace(" ", "").replace("_", "")
    if "name" in key:
        return "full_name"
    if "birth" in key:
        return "date_of_birth"
    if "id" in key and "number" in key:
        return "id_number"
    if "passport" in key and "number" in key:
        return "passport_number"
    if "nationality" in key:
        return "nationality"
    if "gender" in key or "sex" in key:
        return "gender"
    if "expiry" in key or "expire" in key:
        return "expiry_date"
    return key

def compare_documents(id_info, passport_info):
    """
    Compare key fields between ID card and passport.
    Returns a dict with comparison results.
    """
    # Define which fields to compare
    fields_to_compare = ["full_name", "date_of_birth", "nationality", "gender", "expiry_date"]
    # Normalize both dictionaries
    norm_id_info = {normalize_key(k): v for k, v in id_info.items()}
    norm_passport_info = {normalize_key(k): v for k, v in passport_info.items()}

    comparison = {}
    for field in fields_to_compare:
        id_value = norm_id_info.get(field, "Not Available")
        passport_value = norm_passport_info.get(field, "Not Available")
        # Simple comparison: case-insensitive, ignore whitespace
        match = (
            id_value != "Not Available" and
            passport_value != "Not Available" and
            str(id_value).strip().lower() == str(passport_value).strip().lower()
        )
        comparison[field] = {
            "id_card": id_value,
            "passport": passport_value,
            "match": match
        }
    return comparison

def main():
    api_key = "AIzaSyCvFefseo_KYbBy0EcDfMDLKFleqeDh57Q"  # Your Gemini API key

    # Paths to your images
    id_image_path = r"C:\Users\User\Desktop\KYC_Project\Testing_images\id_hadi.png"
    passport_image_path = r"C:\Users\User\Desktop\KYC_Project\Testing_images\hadi_passport.png"

    # Extract info from ID card
    id_info = extract_document_info(id_image_path, api_key, doc_type="ID card")
    # Extract info from passport
    passport_info = extract_document_info(passport_image_path, api_key, doc_type="passport")

    # Print the extracted information
    print("\nExtracted ID Card Information:")
    print("-" * 30)
    if id_info:
        for key, value in id_info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
    else:
        print("No information extracted from ID card.")
    print("-" * 30)

    print("\nExtracted Passport Information:")
    print("-" * 30)
    if passport_info:
        for key, value in passport_info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
    else:
        print("No information extracted from passport.")
    print("-" * 30)

    # Compare the two documents
    if id_info and passport_info:
        comparison = compare_documents(id_info, passport_info)
        print("\nComparison Results:")
        print("-" * 30)
        for field, result in comparison.items():
            print(f"{field.replace('_', ' ').title()}:")
            print(f"  ID Card:   {result['id_card']}")
            print(f"  Passport:  {result['passport']}")
            print(f"  Match:     {'✅' if result['match'] else '❌'}")
            print("-" * 30)
    else:
        print("Comparison not possible due to missing information.")

    # Return both results and comparison if needed
    return {
        "id_card_info": id_info,
        "passport_info": passport_info,
        "comparison": comparison if id_info and passport_info else None
    }

if __name__ == "__main__":
    main()
