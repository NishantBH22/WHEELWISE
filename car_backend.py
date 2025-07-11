import re
import pandas as pd
from sqlalchemy import create_engine
from fuzzywuzzy import fuzz
import random

def get_synonyms_dictionary():
    """Comprehensive synonyms dictionary"""
    return {
        'gas': 'regular unleaded',
        'gasoline': 'regular unleaded',
        'petrol': 'regular unleaded',
        'unleaded': 'regular unleaded',
        'regular': 'regular unleaded',
        'premium unleaded': 'premium unleaded (required)',
        'premium gas': 'premium unleaded (required)',
        'high octane': 'premium unleaded (required)',
        'cng': 'natural gas',
        'compressed natural gas': 'natural gas',
        'e85': 'flex-fuel (unleaded/e85)',
        'ethanol': 'flex-fuel (unleaded/e85)',
        'flex': 'flex-fuel (unleaded/e85)',
        'flexfuel': 'flex-fuel (unleaded/e85)',
        'hybrid gas': 'flex-fuel (unleaded/e85)',
        'ev': 'electric',
        'battery': 'electric',
        'electric vehicle': 'electric',
        'bev': 'electric',
        'man': 'manual',
        'stick': 'manual',
        'stick shift': 'manual',
        'manual transmission': 'manual',
        'mt': 'manual',
        'standard': 'manual',
        'auto': 'automatic',
        'automatic transmission': 'automatic',
        'at': 'automatic',
        'cvt': 'automatic',
        'continuously variable': 'automatic',
        'dual clutch': 'automated_manual',
        'dct': 'automated_manual',
        'semi automatic': 'automated_manual',
        'paddle shift': 'automated_manual',
        'fwd': 'front wheel drive',
        'front wheel': 'front wheel drive',
        'front': 'front wheel drive',
        'rwd': 'rear wheel drive',
        'rear wheel': 'rear wheel drive',
        'rear': 'rear wheel drive',
        'awd': 'all wheel drive',
        'all wheel': 'all wheel drive',
        '4wd': 'four wheel drive',
        '4x4': 'four wheel drive',
        'four wheel': 'four wheel drive',
        '4 wheel drive': 'four wheel drive',
        'sedan': 'sedan',
        'hatch': 'hatchback',
        'hatchback': 'hatchback',
        'coupe': 'coupe',
        'suv': 'suv',
        'truck': 'pickup',
        'pickup truck': 'pickup',
        'van': 'minivan',
        'minivan': 'minivan',
        'wagon': 'wagon',
        'convertible': 'convertible',
        'cabriolet': 'convertible',
        'cabrio': 'convertible',
        'roadster': 'convertible',
        'big': 'large',
        'large': 'large',
        'full size': 'large',
        'fullsize': 'large',
        'mid': 'midsize',
        'midsize': 'midsize',
        'medium': 'midsize',
        'mid-size': 'midsize',
        'small': 'compact',
        'compact': 'compact',
        'subcompact': 'compact',
        'mini': 'compact',
        'sport': 'performance',
        'sports': 'performance',
        'sporty': 'performance',
        'fast': 'performance',
        'high performance': 'high-performance',
        'race': 'performance',
        'racing': 'performance',
        'luxurious': 'luxury',
        'upscale': 'luxury',
        'high end': 'luxury',
        'economy': 'crossover',
        'eco': 'hybrid',
        'green': 'hybrid',
        'environmental': 'hybrid',
        'fuel efficient': 'good mpg',
        'good mileage': 'good mpg',
        'tuner': 'factory tuner',
        'modified': 'factory tuner',
        'custom': 'factory tuner',
        'highway mpg': 'highwaympg',
        'city mpg': 'citympg',
        'fuel economy': 'highwaympg',
        'mileage': 'highwaympg',
        'mpg': 'highwaympg',
        'highway mileage': 'highwaympg',
        'city mileage': 'citympg',
        'popular': 'popularity',
        'brand popularity': 'popularity',
        'social score': 'popularity',
        'doors': 'door',
        'door': 'door',
        '2 door': '2dr',
        '4 door': '4dr',
        'two door': '2dr',
        'four door': '4dr',
        'cylinder': 'cyl',
        'cylinders': 'cyl',
        'horsepower': 'hp',
        'power': 'hp',
        'price': 'msrp',
        'cost': 'msrp',
        'cheap': 'low price',
        'expensive': 'high price',
        'affordable': 'low price',
        'budget': 'low price',
    }


def apply_synonyms(text, synonyms_dict):
    """Apply context-aware synonym replacement"""
    text_lower = text.lower()
    working_text = text_lower
    fuel_compounds = {
        'premium unleaded': 'premium unleaded (required)',
        'premium gas': 'premium unleaded (required)',
        'high octane': 'premium unleaded (required)',
        'regular unleaded': 'regular unleaded',
        'compressed natural gas': 'natural gas',
        'flex fuel': 'flex-fuel (unleaded/e85)',
        'electric vehicle': 'electric',
    }
    for compound, replacement in fuel_compounds.items():
        if compound in working_text:
            working_text = working_text.replace(compound, replacement)
            print(f"Applied compound fuel mapping: '{compound}' -> '{replacement}'")
    hp_pattern = re.compile(r'(\d+)\s*(hp|horsepower)\b', re.IGNORECASE)
    working_text = hp_pattern.sub(r'\1 horsepower', working_text)
    processed_fuel_terms = any(fuel in working_text for fuel in fuel_compounds.values())
    remaining_synonyms = {k: v for k, v in synonyms_dict.items() if k not in ['hp', 'horsepower']}
    if processed_fuel_terms:
        remaining_synonyms = {k: v for k, v in remaining_synonyms.items() if k not in ['premium', 'unleaded', 'gas']}
        print("Skipping individual fuel synonyms since compound was processed")
    sorted_synonyms = sorted(remaining_synonyms.items(), key=lambda x: len(x[0]), reverse=True)
    for synonym, replacement in sorted_synonyms:
        pattern = r'\b' + re.escape(synonym) + r'\b'
        if re.search(pattern, working_text):
            if synonym == 'premium' and any(fuel_word in working_text for fuel_word in ['unleaded', 'gas', 'fuel']):
                continue
            working_text = re.sub(pattern, replacement, working_text)
    return working_text


def enhanced_categorical_matching(text, unique_values, synonyms_dict):
    """Enhanced categorical matching with synonym support"""
    processed_text = apply_synonyms(text, synonyms_dict)
    print(f"Processed text for categorical matching: {processed_text}")
    categorical_matches = {}
    categorical_mappings = {
        'transmissiontype': unique_values.get('transmissiontype', []),
        'drivenwheels': unique_values.get('drivenwheels', []),
        'vehiclestyle': unique_values.get('vehiclestyle', []),
        'vehiclesize': unique_values.get('vehiclesize', []),
        'enginefueltype': unique_values.get('enginefueltype', []),
        'marketcategory': unique_values.get('marketcategory', [])
    }
    for spec_key, possible_values in categorical_mappings.items():
        best_match = None
        best_score = 0
        for value in possible_values:
            value_lower = value.lower()
            if value_lower in processed_text:
                categorical_matches[spec_key] = value
                best_match = value
                print(f"Found exact match for {spec_key}: {value}")
                break
            for word in processed_text.split():
                if word in value_lower or value_lower in word:
                    score = fuzz.ratio(word, value_lower)
                    if score > best_score and score > 80:
                        best_score = score
                        best_match = value
        if best_match and spec_key not in categorical_matches:
            categorical_matches[spec_key] = best_match
            print(f"Found fuzzy match for {spec_key}: {best_match} (score: {best_score})")
    if 'premium unleaded (required)' in processed_text:
        categorical_matches['enginefueltype'] = 'premium unleaded (required)'
        print("Explicitly set fuel type to premium unleaded (required)")
    if 'hatch' in processed_text or 'hatchback' in processed_text:
        doors_match = re.search(r'(\d)\s*(?:door|dr)', processed_text)
        if doors_match:
            doors = doors_match.group(1)
            if doors == '2':
                categorical_matches['vehiclestyle'] = '2dr hatchback'
            elif doors == '4':
                categorical_matches['vehiclestyle'] = '4dr hatchback'
        elif 'vehiclestyle' not in categorical_matches:
            categorical_matches['vehiclestyle'] = '4dr hatchback'
    if 'suv' in processed_text:
        doors_match = re.search(r'(\d)\s*(?:door|dr)', processed_text)
        if doors_match:
            doors = doors_match.group(1)
            if doors == '2':
                categorical_matches['vehiclestyle'] = '2dr suv'
            elif doors == '4':
                categorical_matches['vehiclestyle'] = '4dr suv'
        elif 'vehiclestyle' not in categorical_matches:
            categorical_matches['vehiclestyle'] = '4dr suv'
    if any(term in processed_text for term in ['truck', 'pickup']) and 'horsepower' not in processed_text:
        if 'crew cab' in processed_text or 'crew' in processed_text:
            categorical_matches['vehiclestyle'] = 'crew cab pickup'
        elif 'extended cab' in processed_text or 'extended' in processed_text:
            categorical_matches['vehiclestyle'] = 'extended cab pickup'
        elif 'vehiclestyle' not in categorical_matches:
            categorical_matches['vehiclestyle'] = 'crew cab pickup'
    return categorical_matches


def load_model_mappings():
    """Load model mappings for special cases"""
    return {
        'spider': '124 spider',
        'mustang': 'mustang',
        'camaro': 'camaro',
        'camry': 'camry',
        'corolla': 'corolla',
        'prius': 'prius',
        'accord': 'accord',
        'civic': 'civic',
        '3 series': '3 series',
        '1 series': '1 series',
        '5 series': '5 series',
        '7 series': '7 series',
        '350-class': '350-class',
        '300-class': '300-class',
        'e-class': 'e-class',
        's-class': 's-class',
        '300': '300',
        'aveo': 'aveo',
        '3': '3',
        '370z': '370z',
        '300m': '300m',
    }


def get_all_unique_values(engine):
    """Get unique values from categorical columns"""
    unique_values = {}
    categorical_columns = [
        'EngineFuelType', 'TransmissionType', 'DrivenWheels',
        'MarketCategory', 'VehicleSize', 'VehicleStyle'
    ]
    for column in categorical_columns:
        try:
            query = f"SELECT DISTINCT {column} FROM Cars WHERE {column} IS NOT NULL AND {column} != ''"
            values = pd.read_sql(query, engine)[column].tolist()
            unique_values[column.lower()] = [str(v).strip().lower() for v in values if str(v).strip()]
            print(f"Loaded {len(unique_values[column.lower()])} unique values for {column}")
        except Exception as e:
            print(f"Warning: Could not get values for {column}: {e}")
            unique_values[column.lower()] = []
    return unique_values


def calculate_price_score(target_price, actual_price, tolerance=0.20):
    """Calculate price score with tolerance"""
    if target_price is None or pd.isna(actual_price):
        return 0
    price_diff = abs(actual_price - target_price)
    max_diff = target_price * tolerance
    if price_diff <= max_diff:
        return 100 * (1 - price_diff / max_diff)
    else:
        penalty_factor = min(2.0, price_diff / target_price)
        return max(0, 50 - penalty_factor * 25)


def calculate_match_score(row, parsed_specs, max_scores):
    """Calculate match score with enhanced differentiation and capped at 100"""
    score = 0
    total_weight = 0
    penalties = 0
    specific_query = parsed_specs['make'] or parsed_specs['model']
    if parsed_specs.get('make'):
        make_similarity = fuzz.ratio(str(row['make']).lower(), parsed_specs['make'].lower())
        make_match = make_similarity > 75
        if make_match:
            score += min(100, make_similarity) * 0.20
            total_weight += 0.20
        elif specific_query:
            return 0
    if parsed_specs.get('model'):
        db_model = str(row['model']).lower().strip()
        search_model = parsed_specs['model'].lower().strip()
        if db_model == search_model:
            score += 100 * 0.25
            total_weight += 0.25
        elif db_model.startswith(search_model + ' ') or db_model.startswith(search_model):
            score += 95 * 0.25
            total_weight += 0.25
        elif specific_query:
            return 0
    if parsed_specs.get('msrp') and pd.notna(row['msrp']):
        price_score = calculate_price_score(parsed_specs['msrp'], row['msrp'])
        score += price_score * 0.15
        total_weight += 0.15
    elif parsed_specs.get('msrp_min') and parsed_specs.get('msrp_max') and pd.notna(row['msrp']):
        if parsed_specs['msrp_min'] <= row['msrp'] <= parsed_specs['msrp_max']:
            score += 100 * 0.15
        else:
            price_diff = min(abs(row['msrp'] - parsed_specs['msrp_min']), abs(row['msrp'] - parsed_specs['msrp_max']))
            max_diff = max(parsed_specs['msrp_max'] - parsed_specs['msrp_min'], 1)
            score += max(0, 100 * (1 - price_diff / max_diff)) * 0.15
        total_weight += 0.15
    if parsed_specs.get('year') is not None:
        year_diff = abs(int(row['year']) - parsed_specs['year'])
        if year_diff == 0:
            score += 100 * 0.10
        elif year_diff == 1:
            score += 90 * 0.10
        else:
            score += max(0, 60 - year_diff * 10) * 0.10
        total_weight += 0.10
    elif parsed_specs.get('year_min') is not None and parsed_specs.get('year_max') is not None:
        if parsed_specs['year_min'] <= int(row['year']) <= parsed_specs['year_max']:
            score += 100 * 0.10
        else:
            year_diff = min(abs(int(row['year']) - parsed_specs['year_min']),
                            abs(int(row['year']) - parsed_specs['year_max']))
            score += max(0, 100 - year_diff * 20) * 0.10
        total_weight += 0.10
    if parsed_specs.get('enginehp') and pd.notna(row.get('enginehp')):
        hp_diff = abs(row['enginehp'] - parsed_specs['enginehp'])
        target_hp = parsed_specs['enginehp']
        if target_hp <= 150:
            tolerance = 15
        elif target_hp <= 250:
            tolerance = 25
        else:
            tolerance = 35
        if hp_diff <= tolerance:
            score += (100 - (hp_diff / tolerance) * 80) * 0.35  # Increased weight
            total_weight += 0.35
        else:
            penalty_factor = min(2.0, hp_diff / tolerance)
            penalties += 25 * penalty_factor  # Increased penalty
    categorical_weights = {
        'transmissiontype': 0.20,  # Increased weight
        'drivenwheels': 0.06,
        'vehiclestyle': 0.06,
        'vehiclesize': 0.05,
        'enginefueltype': 0.04,
        'marketcategory': 0.04
    }
    for feature, weight in categorical_weights.items():
        if parsed_specs.get(feature) and pd.notna(row.get(feature)):
            db_value = str(row[feature]).lower().strip()
            search_value = parsed_specs[feature].lower().strip()
            if db_value == search_value:
                score += 100 * weight
                total_weight += weight
            elif feature == 'marketcategory':
                db_categories = [cat.strip() for cat in db_value.split(',')]
                if search_value in db_categories or any(fuzz.ratio(search_value, cat) > 85 for cat in db_categories):
                    score += 90 * weight
                    total_weight += weight
                else:
                    penalties += 25  # Increased penalty
            elif fuzz.ratio(db_value, search_value) > 85:
                score += (fuzz.ratio(db_value, search_value) / 100) * 90 * weight
                total_weight += weight
            else:
                penalties += 25  # Increased penalty
    numerical_features = {
        'enginecylinders': 0.04,
        'numberofdoors': 0.03,
        'highwaympg': 0.05,  # Adjusted weight
        'citympg': 0.05,  # Adjusted weight
        'popularity': 0.02
    }
    for feature, weight in numerical_features.items():
        if parsed_specs.get(feature) and pd.notna(row.get(feature)):
            target_val = parsed_specs[feature]
            actual_val = row[feature]
            if feature == 'enginecylinders':
                if actual_val == target_val:
                    score += 100 * weight
                    total_weight += weight
                elif abs(actual_val - target_val) <= 1:
                    score += 75 * weight
                    total_weight += weight
                else:
                    penalties += 25
            elif feature == 'numberofdoors':
                if actual_val == target_val:
                    score += 100 * weight
                    total_weight += weight
                else:
                    penalties += 25
            elif feature in ['highwaympg', 'citympg']:
                diff_pct = abs(actual_val - target_val) / max(target_val, 1)
                if diff_pct <= 0.25:
                    mpg_score = 100 * (1 - diff_pct / 0.25)
                    score += mpg_score * weight
                    total_weight += weight
                else:
                    penalties += 25  # Increased penalty
            elif feature == 'popularity':
                diff_pct = abs(actual_val - target_val) / max(max_scores[feature], 1)
                if diff_pct <= 0.25:
                    pop_score = 100 * (1 - diff_pct / 0.25)
                    score += pop_score * weight
                    total_weight += weight
                else:
                    penalties += 25
    if total_weight > 0:
        base_score = score / total_weight
        final_score = max(0, min(100, base_score - penalties))  # Cap at 100
        final_score += random.uniform(0, 0.1)  # Reduced tie-breaker
        final_score = min(final_score, 100)  # Ensure final score doesn't exceed 100
    else:
        final_score = max(0, 50 - penalties) + random.uniform(0, 0.1)
        final_score = min(final_score, 100)  # Ensure final score doesn't exceed 100
    return round(final_score, 2)


def parse_user_input(text, makes, models, model_mappings, unique_values):
    """Improved parsing with corrected model handling and invalid input rejection"""
    specs = {
        'make': None,
        'model': None,
        'year': None,
        'year_min': None,
        'year_max': None,
        'enginefueltype': None,
        'enginehp': None,
        'enginecylinders': None,
        'transmissiontype': None,
        'drivenwheels': None,
        'numberofdoors': None,
        'marketcategory': None,
        'vehiclestyle': None,
        'vehiclesize': None,
        'highwaympg': None,
        'citympg': None,
        'popularity': None,
        'msrp': None,
        'msrp_min': None,
        'msrp_max': None
    }
    if '_' in text:
        parts = text.split('_')
        text = ' '.join(parts)
    elif ';' in text:
        parts = text.split(';')
        text = ' '.join(parts)
    synonyms_dict = get_synonyms_dictionary()
    text_lower = text.lower()
    original_text = text
    used_numbers = set()
    invalid_terms = {
        'cylinder', 'cyl', 'door', 'dr', 'luxury', 'performance', 'hybrid', 'crossover',
        'diesel', 'electric', 'can', 'suggest', 'car', 'you', 'quit', 'exit', 'about',
        'around', 'with', 'that', 'is', 'has', 'a'
    }

    # Categorical matching first
    processed_text = apply_synonyms(text, synonyms_dict)
    categorical_matches = enhanced_categorical_matching(processed_text, unique_values, synonyms_dict)
    for key, value in categorical_matches.items():
        specs[key] = value
        text = re.sub(r'\b' + re.escape(value.lower()) + r'\b', '', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'\s+', ' ', text)
        print(f"Removed categorical term from text: {value}")

    # Year range or single year parsing
    year_range_match = re.search(r'\b((?:19|20)\d{2})\s*-\s*((?:19|20)\d{2})\b', text_lower)
    if year_range_match:
        try:
            year_min = int(year_range_match.group(1))
            year_max = int(year_range_match.group(2))
            if 1900 <= year_min <= 2025 and 1900 <= year_max <= 2025 and year_min <= year_max:
                specs['year_min'] = year_min
                specs['year_max'] = year_max
                used_numbers.add(year_min)
                used_numbers.add(year_max)
                text = text.replace(year_range_match.group(0), '').strip()
                print(f"Extracted year range: {year_min}-{year_max}")
            else:
                print(f"Invalid year range: {year_min}-{year_max}")
                return specs, True
        except (ValueError, TypeError) as e:
            print(f"Error parsing year range: {e}")
            return specs, True
    else:
        year_match = re.search(r'\b((?:19|20)\d{2})\b', text_lower)
        if year_match:
            try:
                year_value = int(year_match.group(1))
                if 1900 <= year_value <= 2025:
                    specs['year'] = year_value
                    used_numbers.add(year_value)
                    text = text.replace(year_match.group(1), '').strip()
                    print(f"Extracted year: {year_value}")
                else:
                    print(f"Invalid year: {year_value}")
                    return specs, True
            except (ValueError, TypeError) as e:
                print(f"Error parsing year: {e}")
                return specs, True

    # MSRP range or single MSRP parsing
    msrp_range_match = re.search(r'(?:msrp|price|cost|\$|budget)\s*[\$:]?\s*(\d{4,6})\s*-\s*(\d{4,6})\b', text_lower,
                                 re.IGNORECASE)
    if msrp_range_match:
        try:
            msrp_min = int(msrp_range_match.group(1))
            msrp_max = int(msrp_range_match.group(2))
            if 1000 <= msrp_min <= 200000 and 1000 <= msrp_max <= 200000 and msrp_min <= msrp_max and msrp_min not in used_numbers and msrp_max not in used_numbers:
                specs['msrp_min'] = msrp_min
                specs['msrp_max'] = msrp_max
                used_numbers.add(msrp_min)
                used_numbers.add(msrp_max)
                text = text.replace(msrp_range_match.group(0), '').strip()
                print(f"Extracted MSRP range: ${msrp_min}-${msrp_max}")
            else:
                print(f"Invalid MSRP range: ${msrp_min}-${msrp_max}")
                return specs, True
        except (ValueError, TypeError) as e:
            print(f"Error parsing MSRP range: {e}")
            return specs, True
    else:
        msrp_patterns = [
            r'msrp\s*[\$:]?\s*(\d{4,6})\b',
            r'price\s*[\$:]?\s*(\d{4,6})\b',
            r'\$(\d{4,6})\b',
            r'(\d{4,6})\s*dollars?\b',
            r'budget\s*[\$:]?\s*(\d{4,6})\b',
            r'cost\s*[\$:]?\s*(\d{4,6})\b',
            r'around\s*\$?(\d{4,6})\b',
            r'about\s*\$?(\d{4,6})\b',
            r'under\s*\$?(\d{4,6})\b',
        ]
        msrp_extracted = False
        for pattern in msrp_patterns:
            price_match = re.search(pattern, text_lower, re.IGNORECASE)
            if price_match:
                try:
                    price_value = int(price_match.group(1))
                    if 1000 <= price_value <= 200000 and price_value not in used_numbers:
                        specs['msrp'] = price_value
                        used_numbers.add(price_value)
                        text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
                        text = re.sub(r'\s+', ' ', text)
                        msrp_extracted = True
                        print(f"Extracted MSRP: ${price_value}")
                        break
                    else:
                        print(f"Invalid MSRP: ${price_value}")
                        return specs, True
                except (ValueError, TypeError) as e:
                    print(f"Error parsing MSRP: {e}")
                    return specs, True

    # Negative MSRP validation
    negative_price_match = re.search(r'(?:msrp|price|cost|\$|budget)\s*[\$:]?\s*-(\d+)', text_lower, re.IGNORECASE)
    if negative_price_match:
        print("Invalid input: MSRP cannot be negative.")
        return specs, True

    # MPG validation
    if 'mpg' in text_lower:
        mpg_match = re.search(r'(\d{1,3})\s*mpg', text_lower, re.IGNORECASE)
        if mpg_match and (int(mpg_match.group(1)) < 5 or int(mpg_match.group(1)) > 60):
            print("Invalid input: MPG value seems unrealistic (must be between 5 and 60).")
            return specs, True

    hp_patterns = [
        r'(\d{2,4})\s*(?:hp|horsepower)\b',
        r'(\d{2,4})\s*horse\s*power\b',
        r'about\s*(\d{2,4})\s*(?:hp|horsepower)',
        r'around\s*(\d{2,4})\s*(?:hp|horsepower)',
        r'with\s*(\d{2,4})\s*(?:hp|horsepower)',
        r'(\d{2,4})\s*power\b',
        r'(?:,\s*)?(\d{2,4})\s*hp\b',
        r'hp\s*(\d{2,4})\b',
    ]
    hp_extracted = False
    for pattern in hp_patterns:
        hp_match = re.search(pattern, text_lower, re.IGNORECASE)
        if hp_match:
            try:
                hp_value = int(hp_match.group(1))
                if 50 <= hp_value <= 1000 and hp_value not in used_numbers:
                    specs['enginehp'] = hp_value
                    used_numbers.add(hp_value)
                    text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
                    text = re.sub(r'\s+', ' ', text)
                    hp_extracted = True
                    print(f"Extracted HP: {hp_value}")
                    break
            except (ValueError, TypeError) as e:
                print(f"Error parsing HP: {e}")

    mpg_patterns = [
        r'(\d{1,3})-(\d{1,3})\s*mpg\b',
        r'highway\s*(?:mpg|mileage)\s*(\d{1,3})\b',
        r'(\d{1,3})\s*highway\s*(?:mpg|mileage)\b',
        r'city\s*(?:mpg|mileage)\s*(\d{1,3})\b',
        r'(\d{1,3})\s*city\s*(?:mpg|mileage)\b',
        r'(\d{1,3})\s*mpg\b(?!\s*(city|highway))',
    ]
    for pattern in mpg_patterns:
        mpg_match = re.search(pattern, text_lower, re.IGNORECASE)
        if mpg_match:
            try:
                if pattern.endswith(r'mpg\b'):
                    mpg_min = int(mpg_match.group(1))
                    mpg_max = int(mpg_match.group(2))
                    if 5 <= mpg_min <= 60 and 5 <= mpg_max <= 60 and mpg_min <= mpg_max:
                        specs['citympg'] = mpg_min
                        specs['highwaympg'] = mpg_max
                        used_numbers.add(mpg_min)
                        used_numbers.add(mpg_max)
                        print(f"Extracted MPG range: {mpg_min}-{mpg_max} (interpreted as city/highway or flexible)")
                    else:
                        print(f"Invalid MPG range: {mpg_min}-{mpg_max}")
                        return specs, True
                elif 'highway' in pattern:
                    mpg_value = int(mpg_match.group(1))
                    if 5 <= mpg_value <= 60 and mpg_value not in used_numbers:
                        specs['highwaympg'] = mpg_value
                        used_numbers.add(mpg_value)
                        print(f"Extracted highway MPG: {mpg_value}")
                    else:
                        print(f"Invalid highway MPG: {mpg_value}")
                        return specs, True
                elif 'city' in pattern:
                    mpg_value = int(mpg_match.group(1))
                    if 5 <= mpg_value <= 60 and mpg_value not in used_numbers:
                        specs['citympg'] = mpg_value
                        used_numbers.add(mpg_value)
                        print(f"Extracted city MPG: {mpg_value}")
                    else:
                        print(f"Invalid city MPG: {mpg_value}")
                        return specs, True
                elif pattern.endswith(r'mpg\b(?!\s*(city|highway))'):
                    mpg_value = int(mpg_match.group(1))
                    if 5 <= mpg_value <= 60 and mpg_value not in used_numbers:
                        specs['highwaympg'] = mpg_value
                        used_numbers.add(mpg_value)
                        print(f"Extracted ambiguous MPG: {mpg_value} (interpreted as highway)")
                        print(
                            "Note: Ambiguous MPG input; assuming highway MPG. Specify 'city' or 'highway' for clarity.")
                    else:
                        print(f"Invalid MPG: {mpg_value}")
                        return specs, True
                text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
                text = re.sub(r'\s+', ' ', text)
                break
            except (ValueError, TypeError) as e:
                print(f"Error parsing MPG: {e}")
                return specs, True

    popularity_patterns = [
        r'popularity\s*(\d{1,5})\b',
        r'popular\s*score\s*(\d{1,5})\b',
        r'social\s*score\s*(\d{1,5})\b',
        r'brand\s*popularity\s*(\d{1,5})\b',
        r'(\d{1,5})\s*popularity\b',
    ]
    for pattern in popularity_patterns:
        pop_match = re.search(pattern, text_lower, re.IGNORECASE)
        if pop_match:
            try:
                pop_value = int(pop_match.group(1))
                if 1 <= pop_value <= 6000 and pop_value not in used_numbers:
                    specs['popularity'] = pop_value
                    used_numbers.add(pop_value)
                    text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
                    text = re.sub(r'\s+', ' ', text)
                    print(f"Extracted popularity: {pop_value}")
                    break
            except (ValueError, TypeError) as e:
                print(f"Error parsing popularity: {e}")

    best_make = None
    best_make_score = 0
    for make in makes:
        if len(make) < 3:
            continue
        make_lower = make.lower()
        if make_lower in text_lower:
            best_make = make
            break
        score = fuzz.partial_ratio(text_lower, make_lower)
        if score > best_make_score and score > 80:
            best_make_score = score
            best_make = make
    if best_make:
        specs['make'] = best_make.lower()
        text = re.sub(re.escape(best_make), '', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'\s+', ' ', text)

    model_found = False
    vehicle_styles = unique_values.get('vehiclestyle', [])
    market_categories = unique_values.get('marketcategory', [])
    model_keyword_pattern = r'\b(\d{1,4})\s+model\b'
    model_keyword_match = re.search(model_keyword_pattern, text_lower, re.IGNORECASE)
    if model_keyword_match:
        potential_model = model_keyword_match.group(1)
        if potential_model in model_mappings:
            specs['model'] = model_mappings[potential_model]
            model_found = True
            text = re.sub(model_keyword_pattern, '', text, flags=re.IGNORECASE).strip()
            text = re.sub(r'\s+', ' ', text)
            print(f"Found model via 'model' keyword: {specs['model']}")
        else:
            print(f"Number {potential_model} followed by 'model' but not in model_mappings")

    if not model_found:
        for keyword, model in model_mappings.items():
            if keyword == text_lower.strip():
                if keyword.isdigit() and (hp_extracted or msrp_extracted):
                    print(f"Skipping numeric model '{keyword}' because HP or MSRP was extracted")
                    continue
                specs['model'] = model
                model_found = True
                text = re.sub(re.escape(keyword), '', text, flags=re.IGNORECASE).strip()
                text = re.sub(r'\s+', ' ', text)
                print(f"Found mapped model: {model}")
                break
            elif re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                if keyword.isdigit() and (hp_extracted or msrp_extracted):
                    print(f"Skipping numeric model '{keyword}' because HP or MSRP was extracted")
                    continue
                specs['model'] = model
                model_found = True
                text = re.sub(re.escape(keyword), '', text, flags=re.IGNORECASE).strip()
                text = re.sub(r'\s+', ' ', text)
                print(f"Found mapped model: {model}")
                break

    if not model_found:
        model_patterns = [
            r'\b([a-z0-9][a-z0-9\- ]*[a-z0-9])\b',  # Require start/end with alphanumeric, allow spaces
        ]
        for pattern in model_patterns:
            model_match = re.search(pattern, text, re.IGNORECASE)
            if model_match:
                potential_model = model_match.group(1).strip()
                if potential_model.isdigit() and (hp_extracted or msrp_extracted):
                    print(f"Skipping numeric model pattern '{potential_model}' because HP or MSRP was extracted")
                    continue
                if potential_model.lower() in [vs.lower() for vs in vehicle_styles]:
                    print(f"Skipping '{potential_model}' as it matches vehicle style")
                    continue
                if potential_model.lower() in [cat.lower() for cat in
                                               market_categories] or potential_model.lower() in invalid_terms:
                    print(f"Skipping '{potential_model}' as it matches market category or invalid term")
                    continue
                if re.search(r'\b(cylinder|door|dr)\b', potential_model.lower()):
                    print(f"Skipping '{potential_model}' as it contains attribute terms")
                    continue
                if any(fuzz.ratio(potential_model.lower(), m.lower()) > 90 for m in models):
                    specs['model'] = potential_model.lower()
                    text = text.replace(model_match.group(1), '').strip()
                    text = re.sub(r'\s+', ' ', text)
                    print(f"Found model pattern: {potential_model}")
                    break
                else:
                    print(f"Skipping '{potential_model}' as it does not match any known model")

    cyl_match = re.search(r'(\d{1,2})\s*(cylinder|cyl)', text_lower)
    if cyl_match:
        try:
            specs['enginecylinders'] = int(cyl_match.group(1))
            print(f"Extracted cylinders: {specs['enginecylinders']}")
            text = re.sub(r'\d{1,2}\s*(cylinder|cyl)', '', text, flags=re.IGNORECASE).strip()
        except (ValueError, TypeError) as e:
            print(f"Error parsing cylinders: {e}")

    door_patterns = [
        r'(\d)\s*door',
        r'(\d)-door'
    ]
    for pattern in door_patterns:
        door_match = re.search(pattern, text_lower)
        if door_match:
            try:
                specs['numberofdoors'] = int(door_match.group(1))
                print(f"Extracted number of doors: {specs['numberofdoors']}")
                text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
                break
            except (ValueError, TypeError) as e:
                print(f"Error parsing doors: {e}")

    processed_text = apply_synonyms(text, synonyms_dict)
    print(f"Original: {original_text}")
    print(f"After processing: {text}")
    print(f"After synonyms: {processed_text}")
    categorical_matches = enhanced_categorical_matching(processed_text, unique_values, synonyms_dict)
    for key, value in categorical_matches.items():
        if not specs.get(key):
            specs[key] = value

    return specs, False


def apply_filters(cars, parsed_specs, models, relaxed=False):
    """Apply filters with enhanced logging and relaxed constraints"""
    filtered_cars = cars.copy()
    model_filter_applied = False
    print(f"\nApplying filters (relaxed={relaxed}):")
    print(f"Initial number of cars: {len(filtered_cars)}")

    if parsed_specs.get('msrp') is not None:
        msrp_tolerance = parsed_specs['msrp'] * (0.20 if not relaxed else 0.30)
        filtered_cars = filtered_cars[
            (filtered_cars['msrp'] >= parsed_specs['msrp'] - msrp_tolerance) &
            (filtered_cars['msrp'] <= parsed_specs['msrp'] + msrp_tolerance)
            ]
        print(
            f"Applied MSRP filter: ${parsed_specs['msrp'] - msrp_tolerance:.0f} - ${parsed_specs['msrp'] + msrp_tolerance:.0f}, remaining: {len(filtered_cars)}")
    elif parsed_specs.get('msrp_min') is not None and parsed_specs.get('msrp_max') is not None:
        msrp_min = parsed_specs['msrp_min'] * (0.7 if relaxed else 1.0)
        msrp_max = parsed_specs['msrp_max'] * (1.3 if relaxed else 1.0)
        filtered_cars = filtered_cars[
            (filtered_cars['msrp'] >= msrp_min) &
            (filtered_cars['msrp'] <= msrp_max)
            ]
        print(f"Applied MSRP range filter: ${msrp_min:.0f} - ${msrp_max:.0f}, remaining: {len(filtered_cars)}")
        print(f"MSRP values in filtered data: {filtered_cars['msrp'].tolist()[:10]}...")  # Log first 10 MSRP values
    if parsed_specs.get('enginehp') is not None:
        target_hp = parsed_specs['enginehp']
        if target_hp <= 150:
            hp_tolerance = 15
        elif target_hp <= 250:
            hp_tolerance = 25
        else:
            hp_tolerance = 35
        filtered_cars = filtered_cars[
            (filtered_cars['enginehp'] >= target_hp - hp_tolerance) &
            (filtered_cars['enginehp'] <= target_hp + hp_tolerance)
            ]
        print(
            f"Applied HP filter: {target_hp - hp_tolerance} - {target_hp + hp_tolerance} HP, remaining: {len(filtered_cars)}")
    if parsed_specs.get('make') is not None:
        filtered_cars = filtered_cars[filtered_cars['make'].str.lower() == parsed_specs['make'].lower()]
        print(f"Applied make filter: {parsed_specs['make']}, remaining: {len(filtered_cars)}")
    if parsed_specs.get('model') is not None:
        if any(fuzz.ratio(parsed_specs['model'].lower(), m.lower()) > 90 for m in models):
            filtered_cars_temp = filtered_cars[
                filtered_cars['model'].str.lower().str.contains(parsed_specs['model'].lower(), na=False) |
                (filtered_cars['model'].str.lower() == parsed_specs['model'].lower())
                ]
            if filtered_cars_temp.empty:
                print(f"No matches with model filter: {parsed_specs['model']}. Skipping model filter.")
            else:
                filtered_cars = filtered_cars_temp
                print(f"Applied model filter: {parsed_specs['model']}, remaining: {len(filtered_cars)}")
                model_filter_applied = True
        else:
            print(f"Invalid model: {parsed_specs['model']}. Skipping model filter.")
    if parsed_specs.get('year') is not None:
        filtered_cars = filtered_cars[
            (filtered_cars['year'] >= parsed_specs['year'] - 1) &
            (filtered_cars['year'] <= parsed_specs['year'] + 1)
            ]
        print(
            f"Applied year filter: {parsed_specs['year'] - 1} - {parsed_specs['year'] + 1}, remaining: {len(filtered_cars)}")
    elif parsed_specs.get('year_min') is not None and parsed_specs.get('year_max') is not None:
        filtered_cars = filtered_cars[
            (filtered_cars['year'] >= parsed_specs['year_min']) &
            (filtered_cars['year'] <= parsed_specs['year_max'])
            ]
        print(
            f"Applied year range filter: {parsed_specs['year_min']} - {parsed_specs['year_max']}, remaining: {len(filtered_cars)}")
    categorical_features = [
        'enginefueltype', 'transmissiontype', 'drivenwheels',
        'vehiclestyle', 'vehiclesize', 'marketcategory'
    ]
    for feature in categorical_features:
        if parsed_specs.get(feature) is not None:
            if feature == 'marketcategory':
                filtered_cars = filtered_cars[
                    filtered_cars[feature].str.lower().str.contains(parsed_specs[feature].lower(), na=False)
                ]
                print(f"Applied {feature} filter: contains {parsed_specs[feature]}, remaining: {len(filtered_cars)}")
            else:
                filtered_cars = filtered_cars[
                    filtered_cars[feature].str.lower() == parsed_specs[feature].lower()
                    ]
                print(f"Applied {feature} filter: {parsed_specs[feature]}, remaining: {len(filtered_cars)}")
    numerical_features = ['enginecylinders', 'numberofdoors', 'highwaympg', 'citympg', 'popularity']
    for feature in numerical_features:
        if parsed_specs.get(feature) is not None:
            if feature in ['highwaympg', 'citympg']:
                mpg_tolerance = parsed_specs[feature] * (0.25 if not relaxed else 0.35)
                filtered_cars = filtered_cars[
                    (filtered_cars[feature] >= parsed_specs[feature] - mpg_tolerance) &
                    (filtered_cars[feature] <= parsed_specs[feature] + mpg_tolerance)
                    ]
                print(
                    f"Applied {feature} filter: {parsed_specs[feature] - mpg_tolerance:.1f} - {parsed_specs[feature] + mpg_tolerance:.1f}, remaining: {len(filtered_cars)}")
            elif feature == 'popularity':
                pop_tolerance = parsed_specs[feature] * 0.25
                filtered_cars = filtered_cars[
                    (filtered_cars[feature] >= parsed_specs[feature] - pop_tolerance) &
                    (filtered_cars[feature] <= parsed_specs[feature] + pop_tolerance)
                    ]
                print(
                    f"Applied {feature} filter: {parsed_specs[feature] - pop_tolerance:.0f} - {parsed_specs[feature] + pop_tolerance:.0f}, remaining: {len(filtered_cars)}")
            else:
                filtered_cars = filtered_cars[filtered_cars[feature] == parsed_specs[feature]]
                print(f"Applied {feature} filter: {parsed_specs[feature]}, remaining: {len(filtered_cars)}")
    if parsed_specs.get('citympg') and parsed_specs.get('highwaympg'):
        mpg_min = min(parsed_specs['citympg'], parsed_specs['highwaympg'])
        mpg_max = max(parsed_specs['citympg'], parsed_specs['highwaympg'])
        mpg_tolerance_min = mpg_min * (0.25 if not relaxed else 0.35)
        mpg_tolerance_max = mpg_max * (0.25 if not relaxed else 0.35)
        filtered_cars = filtered_cars[
            ((filtered_cars['citympg'] >= mpg_min - mpg_tolerance_min) &
             (filtered_cars['citympg'] <= mpg_max + mpg_tolerance_max)) |
            ((filtered_cars['highwaympg'] >= mpg_min - mpg_tolerance_min) &
             (filtered_cars['highwaympg'] <= mpg_max + mpg_tolerance_max))
            ]
        print(
            f"Applied flexible MPG filter: {mpg_min - mpg_tolerance_min:.1f} - {mpg_max + mpg_tolerance_max:.1f} (city or highway), remaining: {len(filtered_cars)}")
    if filtered_cars.empty and not relaxed:
        print("No matches with strict filters. Retrying with relaxed constraints.")
        filtered_cars = cars.copy()
        return apply_filters(cars, parsed_specs, models, relaxed=True)
    if filtered_cars.empty and model_filter_applied:
        print("No matches with model filter. Retrying without model filter.")
        filtered_cars = cars.copy()
        parsed_specs_temp = parsed_specs.copy()
        parsed_specs_temp['model'] = None
        filtered_cars = apply_filters(cars, parsed_specs_temp, models, relaxed=relaxed)
    if filtered_cars.empty:
        print("Final filtered cars empty. Available MSRP values after make/year/marketcategory filters:")
        temp_cars = cars.copy()
        if parsed_specs.get('make'):
            temp_cars = temp_cars[temp_cars['make'].str.lower() == parsed_specs['make'].lower()]
        if parsed_specs.get('year_min') and parsed_specs.get('year_max'):
            temp_cars = temp_cars[
                (temp_cars['year'] >= parsed_specs['year_min']) & (temp_cars['year'] <= parsed_specs['year_max'])]
        if parsed_specs.get('marketcategory'):
            temp_cars = temp_cars[
                temp_cars['marketcategory'].str.lower().str.contains(parsed_specs['marketcategory'].lower(), na=False)]
        print(f"Available MSRP values: {temp_cars['msrp'].tolist()[:10]}...")
    return filtered_cars


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 20)
    try:
        engine = create_engine("mssql+pyodbc://sa:22bct0290@localhost/CarSpecs?driver=ODBC+Driver+17+for+SQL+Server")
        model_mappings = load_model_mappings()
        makes = pd.read_sql("SELECT DISTINCT Make FROM Cars", engine)['Make'].tolist()
        models = pd.read_sql("SELECT DISTINCT Model FROM Cars", engine)['Model'].tolist()
        unique_values = get_all_unique_values(engine)
        max_query = """
                    SELECT MAX(EngineHP)        as enginehp,
                           MAX(EngineCylinders) as enginecylinders,
                           MAX(NumberOfDoors)   as numberofdoors,
                           MAX(HighwayMPG)      as highwaympg,
                           MAX(CityMPG)         as citympg,
                           MAX(MSRP)            as msrp,
                           MAX(Popularity)      as popularity
                    FROM Cars
                    """
        max_scores = pd.read_sql(max_query, engine).iloc[0].to_dict()
        while True:
            text = input("\nEnter your car requirement (or 'quit' to exit): ").strip()
            if text.lower() == 'quit':
                break
            print(f"Processing: {text}")
            parsed_specs, invalid_input = parse_user_input(text, makes, models, model_mappings, unique_values)
            if invalid_input:
                print("Query rejected due to invalid input. Please check your specifications (e.g., MPG, MSRP, year).")
                continue
            print("\nParsed Specifications:")
            non_null_specs = {k: v for k, v in parsed_specs.items() if v is not None}
            if not non_null_specs:
                print("No specifications could be parsed from your input.")
                continue
            for key, value in non_null_specs.items():
                print(f"  {key}: {value}")
            cars = pd.read_sql("SELECT * FROM Cars", engine)
            cars.columns = cars.columns.str.strip().str.lower()
            filtered_cars = apply_filters(cars, parsed_specs, models)
            if filtered_cars.empty:
                print("\nNo cars matched your input after filtering. Try relaxing your constraints.")
                continue
            filtered_cars['matchscore'] = filtered_cars.apply(
                lambda row: calculate_match_score(row, parsed_specs, max_scores),
                axis=1
            )
            min_score = 25
            if parsed_specs['make'] or parsed_specs['model']:
                min_score = 60
            elif parsed_specs.get('enginehp') and (
                    parsed_specs.get('msrp') or (parsed_specs.get('msrp_min') and parsed_specs.get('msrp_max'))):
                min_score = 40
            elif parsed_specs.get('enginehp') or parsed_specs.get('msrp') or (
                    parsed_specs.get('msrp_min') and parsed_specs.get('msrp_max')):
                min_score = 30
            top_matches = filtered_cars[filtered_cars['matchscore'] >= min_score].sort_values(by='matchscore',
                                                                                              ascending=False)
            if top_matches.empty:
                print("\nNo cars matched your input after scoring. Try being more specific or rephrasing.")
            else:
                print(f"\nTop Matches (min score: {min_score}):")
                print(top_matches[['make', 'model', 'year', 'enginefueltype', 'enginehp',
                                   'transmissiontype', 'drivenwheels', 'numberofdoors',
                                   'marketcategory', 'vehiclesize', 'vehiclestyle',
                                   'highwaympg', 'citympg', 'popularity', 'msrp', 'matchscore']].head(10))
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()