from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from car_backend import parse_user_input, apply_filters, calculate_match_score, load_model_mappings, get_all_unique_values
from sqlalchemy import create_engine
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Allow cross-origin requests from the frontend

# Database connection
try:
    engine = create_engine("mssql+pyodbc://sa:22bct0290@localhost/CarSpecs?driver=ODBC+Driver+17+for+SQL+Server")
except Exception as e:
    print(f"Database connection error: {e}")
    raise

# Load data
try:
    model_mappings = load_model_mappings()
    makes = pd.read_sql("SELECT DISTINCT Make FROM Cars", engine)['Make'].tolist()
    models = pd.read_sql("SELECT DISTINCT Model FROM Cars", engine)['Model'].tolist()
    unique_values = get_all_unique_values(engine)
    cars = pd.read_sql("SELECT * FROM Cars", engine)
    cars.columns = cars.columns.str.strip().str.lower()
    max_query = """
        SELECT MAX(EngineHP) as enginehp,
               MAX(EngineCylinders) as enginecylinders,
               MAX(NumberOfDoors) as numberofdoors,
               MAX(HighwayMPG) as highwaympg,
               MAX(CityMPG) as citympg,
               MAX(MSRP) as msrp,
               MAX(Popularity) as popularity
        FROM Cars
    """
    max_scores = pd.read_sql(max_query, engine).iloc[0].to_dict()
except Exception as e:
    print(f"Error loading data: {e}")
    raise

@app.route('/')
def serve_frontend():
    return send_file('index.html')

@app.route('/api/find-cars', methods=['POST'])
def find_cars():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        if not query:
            return jsonify({'error': 'Please enter a valid car description.'}), 400

        parsed_specs, invalid_input = parse_user_input(query, makes, models, model_mappings, unique_values)
        if invalid_input:
            return jsonify({'error': 'Invalid input. Please check your specifications (e.g., MPG, MSRP, year).'}), 400

        if not any(parsed_specs.values()):
            return jsonify({'error': 'No specifications could be parsed from your input.'}), 400

        # Log Parsed Specifications
        print("\nParsed Specifications:")
        non_null_specs = {k: v for k, v in parsed_specs.items() if v is not None}
        for key, value in non_null_specs.items():
            print(f"  {key}: {value}")

        filtered_cars = apply_filters(cars, parsed_specs, models)
        if filtered_cars.empty:
            return jsonify({'error': 'No cars matched your input after filtering. Try relaxing your constraints.'}), 404

        filtered_cars['matchscore'] = filtered_cars.apply(
            lambda row: calculate_match_score(row, parsed_specs, max_scores), axis=1
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

        top_matches = filtered_cars[filtered_cars['matchscore'] >= min_score].sort_values(
            by='matchscore', ascending=False
        ).head(10)

        # Ensure all columns are included, including popularity
        display_columns = [
            'make', 'model', 'year', 'enginefueltype', 'enginehp', 'transmissiontype',
            'drivenwheels', 'numberofdoors', 'marketcategory', 'vehiclesize',
            'vehiclestyle', 'highwaympg', 'citympg', 'popularity', 'msrp', 'matchscore'
        ]
        result = top_matches[display_columns].to_dict(orient='records')
        for row in result:
            for col in display_columns:
                if pd.isna(row[col]):
                    row[col] = None
                elif col in ['enginehp', 'numberofdoors', 'year', 'highwaympg', 'citympg', 'popularity', 'msrp', 'matchscore']:
                    row[col] = float(row[col]) if pd.notna(row[col]) else None
                else:
                    row[col] = str(row[col]) if pd.notna(row[col]) else None

        return jsonify(result)

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)