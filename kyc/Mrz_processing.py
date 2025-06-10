import easyocr
import cv2
import re
from datetime import datetime
import logging
import json
import string
import unicodedata

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ICAO country codes (partial list, add more as needed)
ICAO_COUNTRY_CODES = {
    'AFG', 'ALB', 'DZA', 'AND', 'AGO', 'ATG', 'ARG', 'ARM', 'AUS', 'AUT', 'AZE',
    'BHS', 'BHR', 'BGD', 'BRB', 'BLR', 'BEL', 'BLZ', 'BEN', 'BTN', 'BOL', 'BIH',
    'BWA', 'BRA', 'BRN', 'BGR', 'BFA', 'BDI', 'KHM', 'CMR', 'CAN', 'CPV', 'CAF',
    'TCD', 'CHL', 'CHN', 'COL', 'COM', 'COG', 'COD', 'CRI', 'CIV', 'HRV', 'CUB',
    'CYP', 'CZE', 'DNK', 'DJI', 'DMA', 'DOM', 'ECU', 'EGY', 'SLV', 'GNQ', 'ERI',
    'EST', 'SWZ', 'ETH', 'FJI', 'FIN', 'FRA', 'GAB', 'GMB', 'GEO', 'DEU', 'GHA',
    'GRC', 'GRD', 'GTM', 'GIN', 'GNB', 'GUY', 'HTI', 'HND', 'HUN', 'ISL', 'IND',
    'IDN', 'IRN', 'IRQ', 'IRL', 'ISR', 'ITA', 'JAM', 'JPN', 'JOR', 'KAZ', 'KEN',
    'KIR', 'PRK', 'KOR', 'KWT', 'KGZ', 'LAO', 'LVA', 'LBN', 'LSO', 'LBR', 'LBY',
    'LIE', 'LTU', 'LUX', 'MDG', 'MWI', 'MYS', 'MDV', 'MLI', 'MLT', 'MHL', 'MRT',
    'MUS', 'MEX', 'FSM', 'MDA', 'MCO', 'MNG', 'MNE', 'MAR', 'MOZ', 'MMR', 'NAM',
    'NRU', 'NPL', 'NLD', 'NZL', 'NIC', 'NER', 'NGA', 'MKD', 'NOR', 'OMN', 'PAK',
    'PLW', 'PAN', 'PNG', 'PRY', 'PER', 'PHL', 'POL', 'PRT', 'QAT', 'ROU', 'RUS',
    'RWA', 'KNA', 'LCA', 'VCT', 'WSM', 'SMR', 'STP', 'SAU', 'SEN', 'SRB', 'SYC',
    'SLE', 'SGP', 'SVK', 'SVN', 'SLB', 'SOM', 'ZAF', 'SSD', 'ESP', 'LKA', 'SDN',
    'SUR', 'SWE', 'CHE', 'SYR', 'TWN', 'TJK', 'TZA', 'THA', 'TLS', 'TGO', 'TON',
    'TTO', 'TUN', 'TUR', 'TKM', 'TUV', 'UGA', 'UKR', 'ARE', 'GBR', 'USA', 'URY',
    'UZB', 'VUT', 'VAT', 'VEN', 'VNM', 'YEM', 'ZMB', 'ZWE'
}

# Field mapping from MRZ keys to possible visual field names
VISUAL_FIELD_MAP = {
    'country': ['Country'],
    'surname': ['Surname', 'Last Name'],
    'given_names': ['Given Names', 'First Name'],
    'passport_number': ['Passport Number', 'Document Number'],
    'nationality': ['Nationality'],
    'birth_date': ['Date of Birth', 'Birth Date'],
    'sex': ['Sex', 'Gender'],
    'expiry_date': ['Date of Expiry', 'Expiry Date'],
    'personal_number': ['Personal Number', 'ID Number'],
    # Add more as needed
}

NATIONALITY_MAP = {
    'AFG': 'Afghan',
    'ALB': 'Albanian',
    'DZA': 'Algerian',
    'AND': 'Andorran',
    'AGO': 'Angolan',
    'ATG': 'Antiguan or Barbudan',
    'ARG': 'Argentine',
    'ARM': 'Armenian',
    'AUS': 'Australian',
    'AUT': 'Austrian',
    'AZE': 'Azerbaijani',
    'BHS': 'Bahamian',
    'BHR': 'Bahraini',
    'BGD': 'Bangladeshi',
    'BRB': 'Barbadian',
    'BLR': 'Belarusian',
    'BEL': 'Belgian',
    'BLZ': 'Belizean',
    'BEN': 'Beninese',
    'BTN': 'Bhutanese',
    'BOL': 'Bolivian',
    'BIH': 'Bosnian or Herzegovinian',
    'BWA': 'Botswanan',
    'BRA': 'Brazilian',
    'BRN': 'Bruneian',
    'BGR': 'Bulgarian',
    'BFA': 'Burkinabé',
    'BDI': 'Burundian',
    'KHM': 'Cambodian',
    'CMR': 'Cameroonian',
    'CAN': 'Canadian',
    'CPV': 'Cape Verdean',
    'CAF': 'Central African',
    'TCD': 'Chadian',
    'CHL': 'Chilean',
    'CHN': 'Chinese',
    'COL': 'Colombian',
    'COM': 'Comorian',
    'COG': 'Congolese',
    'COD': 'Congolese (Democratic Republic)',
    'CRI': 'Costa Rican',
    'CIV': 'Ivorian',
    'HRV': 'Croatian',
    'CUB': 'Cuban',
    'CYP': 'Cypriot',
    'CZE': 'Czech',
    'DNK': 'Danish',
    'DJI': 'Djiboutian',
    'DMA': 'Dominican',
    'DOM': 'Dominican',
    'ECU': 'Ecuadorian',
    'EGY': 'Egyptian',
    'SLV': 'Salvadoran',
    'GNQ': 'Equatorial Guinean',
    'ERI': 'Eritrean',
    'EST': 'Estonian',
    'SWZ': 'Eswatini',
    'ETH': 'Ethiopian',
    'FJI': 'Fijian',
    'FIN': 'Finnish',
    'FRA': 'French',
    'GAB': 'Gabonese',
    'GMB': 'Gambian',
    'GEO': 'Georgian',
    'DEU': 'German',
    'GHA': 'Ghanaian',
    'GRC': 'Greek',
    'GRD': 'Grenadian',
    'GTM': 'Guatemalan',
    'GIN': 'Guinean',
    'GNB': 'Bissau-Guinean',
    'GUY': 'Guyanese',
    'HTI': 'Haitian',
    'HND': 'Honduran',
    'HUN': 'Hungarian',
    'ISL': 'Icelandic',
    'IND': 'Indian',
    'IDN': 'Indonesian',
    'IRN': 'Iranian',
    'IRQ': 'Iraqi',
    'IRL': 'Irish',
    'ISR': 'Israeli',
    'ITA': 'Italian',
    'JAM': 'Jamaican',
    'JPN': 'Japanese',
    'JOR': 'Jordanian',
    'KAZ': 'Kazakhstani',
    'KEN': 'Kenyan',
    'KIR': 'Kiribati',
    'PRK': 'North Korean',
    'KOR': 'South Korean',
    'KWT': 'Kuwaiti',
    'KGZ': 'Kyrgyzstani',
    'LAO': 'Laotian',
    'LVA': 'Latvian',
    'LBN': 'Lebanese',
    'LSO': 'Basotho',
    'LBR': 'Liberian',
    'LBY': 'Libyan',
    'LIE': 'Liechtensteiner',
    'LTU': 'Lithuanian',
    'LUX': 'Luxembourgish',
    'MDG': 'Malagasy',
    'MWI': 'Malawian',
    'MYS': 'Malaysian',
    'MDV': 'Maldivian',
    'MLI': 'Malian',
    'MLT': 'Maltese',
    'MHL': 'Marshallese',
    'MRT': 'Mauritanian',
    'MUS': 'Mauritian',
    'MEX': 'Mexican',
    'FSM': 'Micronesian',
    'MDA': 'Moldovan',
    'MCO': 'Monégasque',
    'MNG': 'Mongolian',
    'MNE': 'Montenegrin',
    'MAR': 'Moroccan',
    'MOZ': 'Mozambican',
    'MMR': 'Burmese',
    'NAM': 'Namibian',
    'NRU': 'Nauruan',
    'NPL': 'Nepalese',
    'NLD': 'Dutch',
    'NZL': 'New Zealand',
    'NIC': 'Nicaraguan',
    'NER': 'Nigerien',
    'NGA': 'Nigerian',
    'MKD': 'Macedonian',
    'NOR': 'Norwegian',
    'OMN': 'Omani',
    'PAK': 'Pakistani',
    'PLW': 'Palauan',
    'PAN': 'Panamanian',
    'PNG': 'Papua New Guinean',
    'PRY': 'Paraguayan',
    'PER': 'Peruvian',
    'PHL': 'Filipino',
    'POL': 'Polish',
    'PRT': 'Portuguese',
    'QAT': 'Qatari',
    'ROU': 'Romanian',
    'RUS': 'Russian',
    'RWA': 'Rwandan',
    'KNA': 'Kittitian or Nevisian',
    'LCA': 'Saint Lucian',
    'VCT': 'Saint Vincentian',
    'WSM': 'Samoan',
    'SMR': 'San Marinese',
    'STP': 'São Toméan',
    'SAU': 'Saudi Arabian',
    'SEN': 'Senegalese',
    'SRB': 'Serbian',
    'SYC': 'Seychellois',
    'SLE': 'Sierra Leonean',
    'SGP': 'Singaporean',
    'SVK': 'Slovak',
    'SVN': 'Slovenian',
    'SLB': 'Solomon Islander',
    'SOM': 'Somali',
    'ZAF': 'South African',
    'SSD': 'South Sudanese',
    'ESP': 'Spanish',
    'LKA': 'Sri Lankan',
    'SDN': 'Sudanese',
    'SUR': 'Surinamese',
    'SWE': 'Swedish',
    'CHE': 'Swiss',
    'SYR': 'Syrian',
    'TWN': 'Taiwanese',
    'TJK': 'Tajikistani',
    'TZA': 'Tanzanian',
    'THA': 'Thai',
    'TLS': 'Timorese',
    'TGO': 'Togolese',
    'TON': 'Tongan',
    'TTO': 'Trinidadian or Tobagonian',
    'TUN': 'Tunisian',
    'TUR': 'Turkish',
    'TKM': 'Turkmen',
    'TUV': 'Tuvaluan',
    'UGA': 'Ugandan',
    'UKR': 'Ukrainian',
    'ARE': 'Emirati',
    'GBR': 'British',
    'USA': 'American',
    'URY': 'Uruguayan',
    'UZB': 'Uzbekistani',
    'VUT': 'Vanuatuan',
    'VAT': 'Vatican',
    'VEN': 'Venezuelan',
    'VNM': 'Vietnamese',
    'YEM': 'Yemeni',
    'ZMB': 'Zambian',
    'ZWE': 'Zimbabwean'
}


FIELDS_TO_COMPARE = [
    # ('document_type', ['Type', 'Document Type', 'Passport Type']),
    ('given_names', ['Given Names', 'First Name']),
    ('surname', ['Surname', 'Last Name']),
    ('nationality', ['Nationality']),
    ('passport_number', ['Passport Number', 'Document Number']),
    ('birth_date', ['Date of Birth', 'Birth Date']),
    ('sex', ['Sex', 'Gender']),
]

def is_valid_country_code(code):
    return code in ICAO_COUNTRY_CODES

def validate_date_format(date_str):
    try:
        if len(date_str) != 6:
            return False
        year = int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        # Leap year and month/day checks
        current_year = datetime.now().year % 100
        full_year = 2000 + year if year <= current_year else 1900 + year
        datetime(full_year, month, day)
        return True
    except Exception:
        return False

def validate_passport_number(number):
    # Basic passport number validation
    if not number or len(number) < 6:
        return False
    # Check if it contains only alphanumeric characters
    return bool(re.match(r'^[A-Z0-9]+$', number))

def compute_icao_check_digit(data):
    weights = [7, 3, 1]
    total = 0
    for i, char in enumerate(data):
        if char.isdigit():
            value = int(char)
        elif char.isalpha():
            value = ord(char.upper()) - 55  # A=10, B=11, ..., Z=35
        elif char == '<':
            value = 0
        else:
            value = 0
        total += value * weights[i % 3]
    return str(total % 10)

def validate_check_digit(data, check_digit):
    return compute_icao_check_digit(data) == check_digit

def validate_composite_check_td3(l2):
    composite_data = (
        l2[0:10] +  # passport number + check digit
        l2[13:20] + # birth date + check digit
        l2[21:28] + # expiry date + check digit
        l2[28:43]   # personal number + check digit
    )
    composite_check_digit = l2[43]
    return validate_check_digit(composite_data, composite_check_digit)

def normalize_key(key):
    """Normalize keys for robust comparison (case-insensitive, ignore spaces/underscores)."""
    return re.sub(r'[\s_]+', '', key).lower()

def normalize_field(value):
    if not isinstance(value, str):
        return value
    value = unicodedata.normalize('NFKC', value)
    value = value.replace('\xa0', '')
    value = re.sub(r'\s+', '', value)
    value = value.translate(str.maketrans('', '', string.punctuation))
    return value.upper()

def normalize_date(value):
    """
    Converts various date formats to YYMMDD string for comparison.
    Handles:
      - '19 AUG 73'
      - '19/08/1973'
      - '1973-08-19'
      - '730819'
      - etc.
    """
    if not isinstance(value, str):
        return value
    value = value.strip().replace('/', ' ').replace('-', ' ').replace('.', ' ')
    # If already in YYMMDD
    if re.fullmatch(r"\d{6}", value):
        return value
    # If in DD MMM YY or DD MMM YYYY
    match = re.match(r"(\d{1,2})\s*([A-Za-z]{3,})\s*(\d{2,4})", value)
    if match:
        day, month_str, year = match.groups()
        try:
            month = datetime.strptime(month_str[:3], "%b").month
            year = int(year)
            # Convert 4-digit year to 2-digit
            if year > 100:
                year = year % 100
            return f"{year:02d}{month:02d}{int(day):02d}"
        except Exception:
            pass
    # Try parsing common formats
    for fmt in ("%d %m %Y", "%d %m %y", "%Y %m %d", "%y %m %d"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.strftime("%y%m%d")
        except Exception:
            continue
    return value.upper().replace(" ", "")

def normalize_nationality(value):
    val = normalize_field(value)
    # Try to map code to name or name to code
    for code, name in NATIONALITY_MAP.items():
        if val in [code, normalize_field(name)]:
            return code
    return val

def compare_selected_fields(mrz_info, visual_info):
    checks = {}
    visual_info_norm = {normalize_key(k): v for k, v in visual_info.items()}

    # Pre-fetch both visual surname and given names
    visual_surname = ''
    visual_given_names = ''
    for candidate in ['Surname', 'Last Name']:
        norm_candidate = normalize_key(candidate)
        if norm_candidate in visual_info_norm:
            visual_surname = visual_info_norm[norm_candidate]
            break
    for candidate in ['Given Names', 'First Name']:
        norm_candidate = normalize_key(candidate)
        if norm_candidate in visual_info_norm:
            visual_given_names = visual_info_norm[norm_candidate]
            break

    for mrz_key, visual_candidates in FIELDS_TO_COMPARE:
        visual_val = ''
        for candidate in visual_candidates:
            norm_candidate = normalize_key(candidate)
            if norm_candidate in visual_info_norm:
                visual_val = visual_info_norm[norm_candidate]
                break
        mrz_val = mrz_info.get(mrz_key, '')

        # Special handling for surname/given_names swap
        if mrz_key == 'surname':
            v_val = normalize_field(visual_val)
            m_val = normalize_field(mrz_val)
            alt_v_val = normalize_field(visual_given_names)
            checks[mrz_key] = (v_val == m_val) or (alt_v_val == m_val)
            logger.warning(f"🧐 Comparing 'surname': Visual='{visual_val}'/'{visual_given_names}' → '{v_val}'/'{alt_v_val}' | MRZ='{mrz_val}' → '{m_val}'")
            continue
        if mrz_key == 'given_names':
            v_val = normalize_field(visual_val)
            m_val = normalize_field(mrz_val)
            alt_v_val = normalize_field(visual_surname)
            checks[mrz_key] = (v_val == m_val) or (alt_v_val == m_val)
            logger.warning(f"🧐 Comparing 'given_names': Visual='{visual_val}'/'{visual_surname}' → '{v_val}'/'{alt_v_val}' | MRZ='{mrz_val}' → '{m_val}'")
            continue

        # Custom nationality logic
        if mrz_key == 'nationality':
            mrz_code = mrz_val.upper()
            visual_nat = str(visual_val).upper()
            # Use your mapping
            mapped = NATIONALITY_MAP.get(mrz_code, mrz_code)
            # Only check if mapped value is in visual field
            checks[mrz_key] = mapped.upper() in visual_nat
            logger.warning(f"🧐 Custom nationality check: MRZ='{mrz_code}' mapped='{mapped}' in Visual='{visual_nat}'? {checks[mrz_key]}")
            continue

        # Date normalization
        if mrz_key in ['birth_date']:
            v_val = normalize_date(visual_val)
            m_val = normalize_date(mrz_val)
        else:
            v_val = normalize_field(visual_val)
            m_val = normalize_field(mrz_val)

        logger.warning(f"🧐 Comparing '{mrz_key}': Visual='{visual_val}' → '{v_val}' | MRZ='{mrz_val}' → '{m_val}'")
        checks[mrz_key] = (v_val == m_val)
    return checks

def calculate_trust_score(results):
    # results: dict with boolean values for each check
    score = sum(results.values())
    max_score = len(results)
    return score / max_score  # Returns a value between 0 and 1

def preprocess_mrz_region(image_path):
    img = cv2.imread(image_path)
    if img is None:
        logger.error("❌ Could not read the image for preprocessing.")
        return image_path
    h = img.shape[0]
    mrz = img[int(h*0.8):, :]
   ## mrz = cv2.cvtColor(mrz, cv2.COLOR_BGR2GRAY)
   ## mrz = cv2.equalizeHist(mrz)
   ## mrz = cv2.adaptiveThreshold(mrz, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
   ## mrz = cv2.medianBlur(mrz, 3)
    temp_path = "preprocessed_mrz.jpg"
    cv2.imwrite(temp_path, mrz)
    return temp_path

def extract_mrz_easyocr(image_path):
    preprocessed_path = preprocess_mrz_region(image_path)
    reader = easyocr.Reader(['en'])
    image = cv2.imread(preprocessed_path)
    if image is None:
        logger.error("❌ Could not read the image.")
        return None
    results = reader.readtext(image, detail=0, paragraph=True)
    if results:
        logger.info("✅ MRZ text found by EasyOCR")
        mrz_text = "\n".join(results)
        return mrz_text
    else:
        logger.error("❌ No MRZ found in the image.")
        return None

def split_mrz_line(mrz):
    mrz = mrz.replace(' ', '').replace('\n', '')
    if len(mrz) >= 88:
        return mrz[:44], mrz[44:88]
    idx = mrz.rfind('<<<<', 30, 60)
    if idx != -1:
        line1 = mrz[:idx+4]
        line2 = mrz[idx+4:]
        line1 = line1.ljust(44, '<')[:44]
        line2 = line2.ljust(44, '<')[:44]
        return line1, line2
    line1 = mrz[:44].ljust(44, '<')[:44]
    line2 = mrz[44:88].ljust(44, '<')[:44]
    return line1, line2

def extract_mrz_lines_from_text(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    mrz_candidates = []
    for line in lines:
        if line.startswith('P<') or (len(line) >= 30 and '<' in line):
            mrz_candidates.append(line.replace(' ', '').replace('\n', ''))
    if len(mrz_candidates) == 1:
        l = mrz_candidates[0]
        if len(l) >= 60:
            line1, line2 = split_mrz_line(l)
            return line1 + '\n' + line2
    if len(mrz_candidates) == 2:
        line1 = mrz_candidates[0].ljust(44, '<')[:44]
        line2 = mrz_candidates[1].ljust(44, '<')[:44]
        return line1 + '\n' + line2
    for i in range(len(mrz_candidates)-1):
        if len(mrz_candidates[i]) == 44 and len(mrz_candidates[i+1]) == 44:
            return mrz_candidates[i] + '\n' + mrz_candidates[i+1]
    if mrz_candidates:
        line1 = mrz_candidates[0].ljust(44, '<')[:44]
        line2 = mrz_candidates[1].ljust(44, '<')[:44] if len(mrz_candidates) > 1 else '<'*44
        return line1 + '\n' + line2
    return ''

def manual_parse_mrz(mrz_string):
    try:
        lines = [line.strip() for line in mrz_string.splitlines() if line.strip()]
        if len(lines) == 2:
            if len(lines[0]) == 44 and len(lines[1]) == 44:
                l1, l2 = lines
                l1 = l1.ljust(44, '<')
                l2 = l2.ljust(44, '<')
                if l1[0] != 'P':
                    logger.warning("Invalid document: Not a passport (first character should be 'P')")
                    return None
                passport_number = l2[0:9].replace("<", "")
                birth_date = l2[13:19]
                expiry_date = l2[21:27]
                if not validate_date_format(birth_date):
                    logger.warning("Invalid birth date format")
                    return None
                if not validate_date_format(expiry_date):
                    logger.warning("Invalid expiry date format")
                    return None
                if not validate_passport_number(passport_number):
                    logger.warning("Invalid passport number format")
                    return None
                personal_number_raw = l2[28:42]
                personal_number_check_digit = l2[42]
                if personal_number_raw.replace('<', '') == '':
                    personal_number_check_result = True
                else:
                    personal_number_check_result = validate_check_digit(personal_number_raw, personal_number_check_digit)
                checks = {
                    'birth_date_format': validate_date_format(birth_date),
                    'expiry_date_format': validate_date_format(expiry_date),
                    'passport_number_format': validate_passport_number(passport_number),
                    'passport_number_check': validate_check_digit(l2[0:9], l2[9]),
                    'birth_date_check': validate_check_digit(l2[13:19], l2[19]),
                    'expiry_date_check': validate_check_digit(l2[21:27], l2[27]),
                    'personal_number_check': personal_number_check_result,
                    'composite_check': validate_composite_check_td3(l2),
                    'country_code_check': is_valid_country_code(l1[2:5])
                }
                logger.info(f"Passport MRZ validation results: {checks}")
                return {
                    "mrz_type": "TD3",
                    "document_type": l1[0],
                    "country": l1[2:5],
                    "surname": l1[5:l1.find("<<")].replace("<", " ").strip(),
                    "given_names": l1[l1.find("<<")+2:].replace("<", " ").strip(),
                    "passport_number": passport_number,
                    "passport_number_check": l2[9],
                    "nationality": l2[10:13],
                    "birth_date": birth_date,
                    "birth_date_check": l2[19],
                    "sex": l2[20],
                    "expiry_date": expiry_date,
                    "expiry_date_check": l2[27],
                    "personal_number": l2[28:42].replace("<", ""),
                    "personal_number_check": l2[42],
                    "composite_check": l2[43],
                    "validation_checks": checks
                }
            else:
                logger.warning(f"Invalid MRZ line lengths for 2-line MRZ: {len(lines[0])}, {len(lines[1])}")
                return None
        else:
            logger.warning(f"Invalid number of MRZ lines: {len(lines)}")
            return None
    except Exception as e:
        logger.error(f"Error in passport MRZ parsing: {str(e)}")
        return None

def process_passport(image_path, api_key=None):
    """
    Extracts MRZ info from a passport image using EasyOCR, parses and validates it, and returns the result.
    """
    mrz_string = extract_mrz_easyocr(image_path)
    if not mrz_string:
        return {
            "status": "REJECTED",
            "message": "Failed to extract MRZ using EasyOCR"
        }
    cleaned_mrz = extract_mrz_lines_from_text(mrz_string)
    mrz_info = manual_parse_mrz(cleaned_mrz)
    if not mrz_info:
        return {
            "status": "REJECTED",
            "message": "Failed to parse passport MRZ information"
        }
    return {
        "mrz_info": mrz_info,
        "status": "SUCCESS",
        "message": "Successfully extracted and parsed MRZ"
    }

# --- Composite check digit validation for TD1 and TD2 ---
def validate_composite_check_td1(l1, l2):
    composite_data = (
        l1[5:14] + l1[14] +  # document number + check
        l2[0:6] + l2[6] +    # birth date + check
        l2[8:14] + l2[14] +  # expiry date + check
        l2[18:29] + l2[29]   # optional data 2 + check
    )
    composite_check_digit = l2[29]
    return validate_check_digit(composite_data, composite_check_digit)

def validate_composite_check_td2(l2):
    composite_data = (
        l2[0:9] + l2[9] +    # passport number + check
        l2[13:19] + l2[19] + # birth date + check
        l2[21:27] + l2[27] + # expiry date + check
        l2[28:35] + l2[35]   # optional data + check
    )
    composite_check_digit = l2[35]
    return validate_check_digit(composite_data, composite_check_digit)

