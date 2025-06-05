from passporteye import read_mrz
from google import generativeai as genai
import re
from datetime import datetime
import logging
import json
import string
import unicodedata
import cv2


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

def is_mrz_valid_in_image(image_path):
    try:
        mrz_obj = read_mrz(image_path)
        if mrz_obj is not None and mrz_obj.valid:
            logger.info("Valid MRZ found in image.")
            return True, mrz_obj
        else:
            logger.warning("No valid MRZ found in image.")
            return False, None
    except Exception as e:
        logger.error(f"Error in MRZ validation: {str(e)}")
        return False, None

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

def retrieve_passport_mrz(image_path: str, api_key: str) -> str:
    try:
        genai.configure(api_key=api_key)
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt_text = (
            "You are an expert system designed to extract the Machine Readable Zone (MRZ) from a passport image. "
            "Return only the MRZ lines, each on a new line, with no extra text or explanation. "
            "Ensure the output is in the correct format with proper spacing and characters."
        )
        
        response = model.generate_content(
            contents=[{
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}}
                ]
            }]
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error in LLM MRZ extraction: {str(e)}")
        return None

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

def manual_parse_mrz(mrz_string):
    try:
        lines = [line.strip() for line in mrz_string.splitlines() if line.strip()]
        if len(lines) == 2:
            if len(lines[0]) == 44 and len(lines[1]) == 44:  # TD3 (passport)
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
            elif len(lines[0]) == 36 and len(lines[1]) == 36:  # TD2 (rare, some passports/visas)
                l1, l2 = lines
                l1 = l1.ljust(36, '<')
                l2 = l2.ljust(36, '<')
                passport_number = l2[0:9].replace("<", "")
                birth_date = l2[13:19]
                expiry_date = l2[21:27]
                if not validate_date_format(birth_date):
                    logger.warning("Invalid birth date format (TD2)")
                    return None
                if not validate_date_format(expiry_date):
                    logger.warning("Invalid expiry date format (TD2)")
                    return None
                if not validate_passport_number(passport_number):
                    logger.warning("Invalid passport number format (TD2)")
                    return None
                optional_data_raw = l2[28:35]
                optional_data_check_digit = l2[35]
                if optional_data_raw.replace('<', '') == '':
                    optional_data_check_result = True
                else:
                    optional_data_check_result = validate_check_digit(optional_data_raw, optional_data_check_digit)
                checks = {
                    'birth_date_format': validate_date_format(birth_date),
                    'expiry_date_format': validate_date_format(expiry_date),
                    'passport_number_format': validate_passport_number(passport_number),
                    'passport_number_check': validate_check_digit(l2[0:9], l2[9]),
                    'birth_date_check': validate_check_digit(l2[13:19], l2[19]),
                    'expiry_date_check': validate_check_digit(l2[21:27], l2[27]),
                    'optional_data_check': optional_data_check_result,
                    'composite_check': validate_composite_check_td2(l2),
                    'country_code_check': is_valid_country_code(l1[2:5])
                }
                logger.info(f"TD2 MRZ validation results: {checks}")
                return {
                    "mrz_type": "TD2",
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
                    "optional_data": l2[28:35].replace("<", ""),
                    "optional_data_check": l2[35],
                    "composite_check": l2[35],
                    "validation_checks": checks
                }
            else:
                logger.warning(f"Invalid MRZ line lengths for 2-line MRZ: {len(lines[0])}, {len(lines[1])}")
                return None
        elif len(lines) == 3:  # TD1 (ID card/passport)
            l1, l2, l3 = lines
            l1 = l1.ljust(30, '<')
            l2 = l2.ljust(30, '<')
            l3 = l3.ljust(30, '<')
            document_number = l1[5:14].replace("<", "")
            birth_date = l2[0:6]
            expiry_date = l2[8:14]
            if not validate_date_format(birth_date):
                logger.warning("Invalid birth date format (TD1)")
                return None
            if not validate_date_format(expiry_date):
                logger.warning("Invalid expiry date format (TD1)")
                return None
            if not validate_passport_number(document_number):
                logger.warning("Invalid document number format (TD1)")
                return None
            optional_data2_raw = l2[18:29]
            optional_data2_check_digit = l2[29]
            if optional_data2_raw.replace('<', '') == '':
                optional_data2_check_result = True
            else:
                optional_data2_check_result = validate_check_digit(optional_data2_raw, optional_data2_check_digit)
            checks = {
                'birth_date_format': validate_date_format(birth_date),
                'expiry_date_format': validate_date_format(expiry_date),
                'document_number_format': validate_passport_number(document_number),
                'document_number_check': validate_check_digit(l1[5:14], l1[14]),
                'birth_date_check': validate_check_digit(l2[0:6], l2[6]),
                'expiry_date_check': validate_check_digit(l2[8:14], l2[14]),
                'optional_data2_check': optional_data2_check_result,
                'composite_check': validate_composite_check_td1(l1, l2),
                'country_code_check': is_valid_country_code(l1[2:5])
            }
            logger.info(f"TD1 MRZ validation results: {checks}")
            return {
                "mrz_type": "TD1",
                "document_type": l1[0],
                "country": l1[2:5],
                "document_number": document_number,
                "document_number_check": l1[14],
                "optional_data1": l1[15:30].replace("<", ""),
                "birth_date": birth_date,
                "birth_date_check": l2[6],
                "sex": l2[7],
                "expiry_date": expiry_date,
                "expiry_date_check": l2[14],
                "nationality": l2[15:18],
                "optional_data2": l2[18:29].replace("<", ""),
                "optional_data2_check": l2[29],
                "composite_check": l2[29],
                "surname": l3.split("<<")[0].replace("<", " ").strip(),
                "given_names": " ".join(l3.split("<<")[1:]).replace("<", " ").strip() if "<<" in l3 else "",
                "validation_checks": checks
            }
        else:
            logger.warning(f"Invalid number of MRZ lines: {len(lines)} (not TD3, TD2, or TD1)")
            return None
    except Exception as e:
        logger.error(f"Error in passport MRZ parsing: {str(e)}")
        return None

def determine_verification_result(trust_score, threshold=0.9):
    """
    Determine whether to accept or reject the passport based on trust score.
    Args:
        trust_score (float): Score between 0 and 1
        threshold (float): Minimum score required for acceptance (default 0.8)
    Returns:
        dict: Verification result with status and details
    """
    if trust_score >= threshold:
        return {
            "status": "ACCEPTED",
            "trust_score": trust_score,
            "message": "Document verification passed"
        }
    else:
        return {
            "status": "REJECTED",
            "trust_score": trust_score,
            "message": "Document verification failed - trust score below threshold"
        }

def extract_passport_visual_info(image_path: str, api_key: str) -> dict:
    """
    Extract visible (printed) info from the passport page using Gemini AI.
    """
    try:
        genai.configure(api_key=api_key)
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        prompt_text = (
            "You are an expert at reading passports. Extract all visible information from the main (photo) page of this passport, "
            "including but not limited to: Full Name, Given Names, Surname, Nationality, Date of Birth, Place of Birth, Sex, Passport Number, "
            "Date of Issue, Date of Expiry, Issuing Authority. "
            "Return the information as a JSON object with English field names. "
            "If any field is not visible, mark it as 'Not Available'."
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
        response_text = response.text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(response_text)
    except Exception as e:
        logger.error(f"Error extracting visual passport info: {str(e)}")
        return None

def preprocess_for_mrz(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return image_path  # fallback

    h = image.shape[0]
    # Crop bottom 20% (adjust as needed)
    mrz_region = image[int(h * 0.8):, :]
    mrz_region = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2GRAY)
    mrz_region = cv2.adaptiveThreshold(mrz_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    temp_path = "static/uploads/preprocessed_mrz.jpg"
    cv2.imwrite(temp_path, mrz_region)
    return temp_path

def process_passport(image_path, api_key):
    """
    Extracts MRZ and visual info from a passport image, compares them, and returns the result.
    Uses LLM to extract MRZ, then parses and validates it, and compares with visual info.
    Reports MRZ and visual-vs-MRZ trust scores separately.
    """
    # 1. Use LLM to extract MRZ from the image
    mrz_string = retrieve_passport_mrz(image_path, api_key)
    if not mrz_string:
        return {
            "status": "REJECTED",
            "message": "Failed to extract MRZ using LLM"
        }
    mrz_info = manual_parse_mrz(mrz_string)
    if not mrz_info:
        return {
            "status": "REJECTED",
            "message": "Failed to parse passport MRZ information from LLM"
        }

    # 2. Extract visual info and compare
    visual_info = extract_passport_visual_info(image_path, api_key)
    if not visual_info:
        return {
            "status": "REJECTED",
            "message": "Failed to extract visual information from passport"
        }
        
    mrz_vs_visual_checks = compare_selected_fields(mrz_info, visual_info)
    mrz_checks = dict(mrz_info['validation_checks'])
    trust_score_mrz = calculate_trust_score(mrz_checks)
    trust_score_visual = calculate_trust_score(mrz_vs_visual_checks)
    trust_score_combined = calculate_trust_score({**mrz_checks, **mrz_vs_visual_checks})
    verification_result = determine_verification_result(trust_score_combined)

    return {
        "mrz_info": mrz_info,
        "visual_info": visual_info,
        "mrz_vs_visual_checks": mrz_vs_visual_checks,
        "mrz_checks": mrz_checks,
        "trust_score_mrz": trust_score_mrz,
        "trust_score_visual": trust_score_visual,
        "trust_score_combined": trust_score_combined,
        "verification_result": verification_result
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

