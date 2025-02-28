import os

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import requests
from flask_cors import CORS
import re
import json

from models import db, User, Injury
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
from langchain.chains import LLMChain
import math

import pandas as pd
from langchain.prompts import PromptTemplate


load_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')

db.init_app(app)
jwt = JWTManager(app)
CORS(app)  # Enable CORS for all routes

openai_api_key = os.getenv("OPENAI_API_KEY")
# Initialize the LLM
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, openai_api_key=openai_api_key)
retriever = None


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.json
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'User already exists.'}), 400

        user = User(username=data['username'])
        user.set_password(data['password'])
        db.session.add(user)
        db.session.commit()
        return jsonify({'message': 'User registered successfully.'}), 201

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.json
        user = User.query.filter_by(username=data['username']).first()

        if user and user.check_password(data['password']):
            token = create_access_token(identity=str(user.id))  # Convert user ID to string
            return jsonify({"token": token}), 200

        return jsonify({"error": "Invalid username or password"}), 401

    return render_template('login.html')

@app.route('/add-injury', methods=['GET'])
def add_injury_page():
    """Renders the injury report form."""
    return render_template('add_injury.html')


@app.route('/api/add-injury', methods=['POST'])
@jwt_required()
def report_injury():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400
    user_id = int(get_jwt_identity())  # Convert back to integer if user_id is stored as int

    required_fields = ["description", "latitude", "longitude"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 422

     # Additional checks for field types
    try:
        injury = Injury(
            user_id=user_id,
            description=str(data['description']),  # Ensure description is a string
            severity="unknown",
            latitude=float(data['latitude']),  # Ensure latitude is a float
            longitude=float(data['longitude'])  # Ensure longitude is a float
        )
    except ValueError as e:
        return jsonify({"error": f"Invalid data type: {str(e)}"}), 422

    db.session.add(injury)
    db.session.commit()

    return jsonify({'message': 'Injury reported successfully.'}), 201


@app.route('/injuries', methods=['GET'])
@jwt_required(optional=True)
def list_of_injuries_page():
    """Renders the injury report form."""
    return render_template('injuries.html')

@app.route('/api/injuries', methods=['GET'])
@jwt_required()
def get_injuries():
    user_id = get_jwt_identity()
    injuries = Injury.query.filter_by(user_id=user_id).all()
    return jsonify([
        {
            'description': i.description,
            'severity': i.severity,
            'latitude': i.latitude,
            'longitude': i.longitude,
            'reported_at': i.reported_at
        } for i in injuries
    ])

@app.route('/report', methods=['GET'])
@jwt_required(optional=True)
def query_page():
    # user_id = get_jwt_identity()
    # print(f'user_id = {user_id}')
    # if not user_id:
    #     return render_template('unauthorized.html'), 401  # Optional: Render a template prompting login
    return render_template('query_page.html')


def initialize_retriever():
    global retriever
    injury_data = pd.read_sql("SELECT description, latitude, longitude, reported_at FROM injuries", db.engine)

    if injury_data.empty:
        print("⚠️ No injury data found.")
        return

    loader = DataFrameLoader(injury_data, page_content_column='description')
    documents = loader.load()

    embeddings = OpenAIEmbeddings()

    # Check if the FAISS index file exists
    if os.path.exists("faiss_index"):
        print("✅ Loading existing FAISS index...")
        retriever = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True).as_retriever()
    else:
        print("🚀 Creating new FAISS index...")
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local("faiss_index")
        retriever = vector_store.as_retriever()

# 📍 Convert location to coordinates
def get_coordinates(location: str):
    import difflib
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    import time

    geolocator = Nominatim(user_agent="injury_app")

    # Safe geocoding with retry
    def safe_geocode(query, max_retries=2):
        for attempt in range(max_retries):
            try:
                return geolocator.geocode(query)
            except (GeocoderTimedOut, GeocoderServiceError):
                time.sleep(1) if attempt < max_retries - 1 else None
        return None

    # 1. Try direct geocoding
    if loc := safe_geocode(location):
        return loc.latitude, loc.longitude

    # 2. Try with country hints
    countries = ["Україна", "Ukraine"]
    for country in countries:
        if loc := safe_geocode(f"{location}, {country}"):
            return loc.latitude, loc.longitude

    # 3. Special case for Ukrainian cities with spelling variations
    variants = {
        "Лисечанськ": ["Лисичанськ", "Lysychansk", "Lisichansk"],
        "Лисичанськ": ["Лисечанськ", "Lysechansk"],
        "Сєвєродонецьк": ["Severodonetsk", "Syeverodonetsk"],
        "Бахмут": ["Bakhmut", "Artyomovsk"],
        "Слов'янськ": ["Sloviansk", "Slavyansk"],
        # Add more as needed
    }

    # Try specific variants
    for original, alternates in variants.items():
        if location in [original] + alternates:
            for name in [original] + alternates:
                for country in countries:
                    if loc := safe_geocode(f"{name}, {country}"):
                        return loc.latitude, loc.longitude

    # 4. Try fuzzy matching
    known_locations = list(variants.keys()) + [item for sublist in variants.values() for item in sublist]
    best_matches = difflib.get_close_matches(location, known_locations, cutoff=0.7)

    for match in best_matches:
        for country in countries:
            if loc := safe_geocode(f"{match}, {country}"):
                return loc.latitude, loc.longitude

    raise ValueError(f"Location not found: {location}")

# 📏 Haversine formula for distance calculation
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# 🧠 Extract query parameters from user input using LLM
def extract_query_parameters(user_query: str):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
        Ти асистент для вилучення інформації із запиту. Завжди відповідай у форматі:
        Місце: <місце або 'Невідомо'>; Радіус: <число> км; Часовий проміжок: <період або 'Невідомо'>.

        Приклади:
        - Запит: 'Які травми були біля Бахмуту на минулому тижні?'
          Відповідь: Місце: Бахмут; Радіус: 10 км; Часовий проміжок: минулого тижня.
        - Запит: 'Скільки було поранень у Харкові за останні 3 дні?'
          Відповідь: Місце: Харків; Радіус: 15 км; Часовий проміжок: останні 3 дні.
        - Запит: 'Які найчастіші травми минулого місяця?'
          Відповідь: Місце: Невідомо; Радіус: 10 км; Часовий проміжок: минулого місяця.
        """),
        ("user", "{query}")
    ])

    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    chain = prompt_template | llm
    response = chain.invoke({"query": user_query}).content

    # 🔎 Parse response format: "Місце: Бахмут; Радіус: 15 км; Часовий проміжок: минулого тижня"
    location = response.split("Місце:")[1].split(';')[0].strip() if "Місце:" in response else None
    radius_str = response.split("Радіус:")[1].split(';')[0].strip() if "Радіус:" in response else "10 км"
    time_frame = response.split("Часовий проміжок:")[1].strip() if "Часовий проміжок:" in response else "минулого тижня"

    radius_km = int(radius_str.replace("км", "").strip())
    return location, radius_km, time_frame

def get_most_frequent_injuries(time_frame: str):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7) if time_frame == "минулого тижня" else None

    result = pd.read_sql(f"""
        SELECT description, COUNT(*) as frequency 
        FROM injuries 
        WHERE reported_at BETWEEN '{start_date}' AND '{end_date}' 
        GROUP BY description ORDER BY frequency DESC LIMIT 5
    """, db.engine)

    return result.to_dict(orient='records')


def parse_time_frame(time_frame: str):
    end_date = datetime.now()

    # Handle non-numeric time frames
    if "вчора" in time_frame:
        start_date = end_date - timedelta(days=1)
    elif "позавчора" in time_frame:
        start_date = end_date - timedelta(days=2)
    elif "на цьому тижні" in time_frame or "в цьому тижні" in time_frame or "цей тиждень" in time_frame:
        # Start from Monday of the current week
        start_date = end_date - timedelta(days=end_date.weekday())  # Move to Monday
    elif "в цьому місяці" in time_frame or "на цьому місяці" in time_frame or "цей місяць" in time_frame:
        # Start from the 1st day of the current month
        start_date = datetime(end_date.year, end_date.month, 1)
    elif "минулого тижня" in time_frame or "на минулому тижні" in time_frame or "останній тиждень" in time_frame:
        # Get start of the previous week (Monday)
        today = end_date.weekday()  # 0 is Monday, 6 is Sunday
        days_since_last_monday = today + 7  # Go back to Monday of previous week
        start_date = end_date - timedelta(days=days_since_last_monday)
        end_date = start_date + timedelta(days=7)  # End of that week (Sunday)
    elif "минулого місяця" in time_frame or "минулий місяць" in time_frame:
        # Get previous month (approximate)
        current_month = end_date.month
        current_year = end_date.year
        if current_month == 1:
            previous_month = 12
            previous_year = current_year - 1
        else:
            previous_month = current_month - 1
            previous_year = current_year
        start_date = datetime(previous_year, previous_month, 1)
        # End is last day of that month (approximate)
        if previous_month == 12:
            end_date = datetime(previous_year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(previous_year, previous_month + 1, 1) - timedelta(days=1)
    else:
        # Improved regex: captures digits and units, ignoring trailing punctuation
        match = re.search(r'(\d+)\s*(дні|день|дня|тижні|тиждень|тижня|місяці|місяць|місяця)[.,]?', time_frame)

        if match:
            value, unit = int(match.group(1)), match.group(2)

            if 'день' in unit or 'дні' in unit or 'дня' in unit:
                start_date = end_date - timedelta(days=value)
            elif 'тиж' in unit:
                start_date = end_date - timedelta(weeks=value)
            elif 'місяц' in unit:
                start_date = end_date - timedelta(days=30 * value)  # Approximate month
            else:
                start_date = end_date - timedelta(days=7)  # Default fallback
        else:
            # Fallback for common phrases without numbers
            if 'минулого тижня' in time_frame:
                start_date = end_date - timedelta(days=7)
            elif 'минулого місяця' in time_frame:
                start_date = end_date - timedelta(days=30)
            else:
                start_date = end_date - timedelta(days=7)  # Default fallback

    return start_date, end_date


def extract_query_parameters_with_langchain(query):
    """Use LangChain to extract query parameters from natural language query"""
    template = """
    Проаналізуй запит українською мовою щодо травм і витягни з нього параметри.

    Запит: {query}

    Витягни наступні параметри у форматі JSON:
    1. location: місце, де шукаються травми (або null якщо не вказано)
    2. radius_km: радіус пошуку в кілометрах (використовуй 10 як значення за замовчуванням)
    3. time_frame: часовий період (наприклад, "минулого тижня", "за останні 3 дні", тощо)
    4. injury_type: тип травми, якщо вказано (наприклад, "лицьові поранення", "травми хребта")
    5. sort_by: як сортувати результати (наприклад, "за частотою", "за датою")
    6. limit: обмеження кількості результатів (використовуй 100 як значення за замовчуванням)
    7. query_type: тип запиту ("count" для підрахунку, "list" для списку травм, "frequent" для найчастіших)

    Формат JSON:
    {{
      "location": "...",
      "radius_km": ...,
      "time_frame": "...",
      "injury_type": "...",
      "sort_by": "...",
      "limit": ...,
      "query_type": "..."
    }}
    """

    prompt = PromptTemplate(
        input_variables=["query"],
        template=template,
    )

    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    chain = prompt | llm
    result = chain.invoke({"query": query}).content

    try:
        # Clean up the result to ensure it's valid JSON
        result = re.sub(r'```json', '', result)
        result = re.sub(r'```', '', result)
        params = json.loads(result.strip())

        # Set default values for missing parameters
        params.setdefault('location', None)
        params.setdefault('radius_km', 10)
        params.setdefault('time_frame', 'за останній тиждень')
        params.setdefault('injury_type', None)
        params.setdefault('sort_by', 'за датою')
        params.setdefault('limit', 100)
        params.setdefault('query_type', 'list')

        # 🔥 If injury_type is missing, extract manually using regex
        if not params['injury_type']:
            params['injury_type'] = extract_injury_type(query)
            print(f"⚡ Manually Extracted injury_type: {params['injury_type']}")

        return params
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM output: {e}")
        print(f"Raw output: {result}")
        # Return default parameters if parsing fails
        return {
            'location': None,
            'radius_km': 10,
            'time_frame': 'за останній тиждень',
            'injury_type': None,
            'sort_by': 'за датою',
            'limit': 100,
            'query_type': 'list'
        }


def extract_injury_type(query):
    """Extract injury type manually with improved Ukrainian language support"""
    # Add word stemming or handle variations
    variations = {
        "контузія": ["контузія", "контузії", "контузією", "контузій"],
        "лицьові поранення": ["лицьові поранення", "лицьового поранення", "лицьовими пораненнями"]
        # Add other injury types with their variations
    }

    # Check for any variation in the query
    for main_type, variants in variations.items():
        for variant in variants:
            if re.search(fr"(?<![а-яіїєґА-ЯІЇЄҐ]){variant}(?![а-яіїєґА-ЯІЇЄҐ])", query, re.IGNORECASE):
                return main_type

    # Fallback to the original INJURY_TYPES list
    for injury in INJURY_TYPES:
        if re.search(fr"(?<![а-яіїєґА-ЯІЇЄҐ]){injury}(?![а-яіїєґА-ЯІЇЄҐ])", query, re.IGNORECASE):
            return injury

    return None  # Default if nothing is found


def build_sql_query(params):
    """Build the SQL query based on the extracted parameters"""
    # Parse the time frame
    start_date, end_date = parse_time_frame(params['time_frame'])

    # Start building the base query
    query = """
        SELECT description, latitude, longitude, reported_at 
        FROM injuries 
        WHERE reported_at BETWEEN %(start_date)s AND %(end_date)s
    """

    query_params = {
        'start_date': start_date,
        'end_date': end_date
    }

    params['start_date'] = start_date
    params['end_date'] = end_date

    # Add injury type filter if specified
    if params['injury_type']:

        query += " AND LOWER(description) ILIKE %(injury_type)s"
        query_params['injury_type'] = f"%{params['injury_type'].lower()}%"

    # Add location filter if specified
    location_filter = ""
    if params['location']:
        try:
            lat, lon = get_coordinates(params['location'])
            # We'll filter by location in Python code after fetching the results
            # Store the coordinates in params for later use
            params['latitude'] = lat
            params['longitude'] = lon
            query_params['latitude'] = lat
            query_params['longitude'] = lon
        except ValueError:
            # If location can't be geocoded, just ignore this filter
            pass

    # Add sorting
    if params['sort_by'] == 'за датою':
        query += " ORDER BY reported_at DESC"

    # Add limit
    query += " LIMIT %(limit)s"
    query_params['limit'] = params['limit']

    return query, query_params


def generate_answer(original_query, params, results_df):
    """Generate a natural language answer based on the query and results"""

    # Apply location filtering if needed
    if params.get('latitude') and params.get('longitude'):
        # Filter by radius in Python
        def is_within_radius(row):
            return haversine(params['latitude'], params['longitude'],
                             row['latitude'], row['longitude']) <= params['radius_km']

        filtered_df = results_df[results_df.apply(is_within_radius, axis=1)]
    else:
        filtered_df = results_df

    # Count results
    count = len(filtered_df)

    # No results case
    if count == 0:
        if params['location']:
            answer = f"Не знайдено травм{' типу ' + params['injury_type'] if params['injury_type'] else ''} у радіусі {params['radius_km']} км від {params['location']} з {params['start_date'].strftime('%Y-%m-%d')} по {params['end_date'].strftime('%Y-%m-%d')}."
            filtered_df = pd.DataFrame()
            return answer, filtered_df
        else:
            return f"Не знайдено травм{' типу ' + params['injury_type'] if params['injury_type'] else ''} за вказаний період."

    # Format count with proper Ukrainian grammar
    if count == 1:
        count_text = "1 травма"
    elif 2 <= count <= 4:
        count_text = f"{count} травми"
    else:
        count_text = f"{count} травм"

    # Generate basic answer
    if params['query_type'] == 'count':
        if params['location']:
            answer = f"Знайдено {count_text}{' типу ' + params['injury_type'] if params['injury_type'] else ''} у радіусі {params['radius_km']} км від {params['location']}."
        else:
            answer = f"Знайдено {count_text}{' типу ' + params['injury_type'] if params['injury_type'] else ''} з {params['start_date'].strftime('%Y-%m-%d')} по {params['end_date'].strftime('%Y-%m-%d')}."


    elif params['query_type'] == 'frequent':
        # Count occurrences of each injury type and get the top 3
        injury_counts = filtered_df['description'].value_counts()

        # Exclude injuries that appear only once
        injury_counts = injury_counts[injury_counts > 1].head(3)

        if injury_counts.empty:
            answer = "Не знайдено достатньо даних для визначення найчастіших травм."
            return answer, pd.DataFrame()

        # Get the most common injuries and their counts
        top_injuries = [(injury, count) for injury, count in zip(injury_counts.index, injury_counts.values)]

        # Format the answer with each injury on a new line
        injuries_text = "\n".join([f"{injury} ({count} випадків)" for injury, count in top_injuries])
        answer = f"Найчастіші травми у районі {params['location']} з {params['start_date'].strftime('%Y-%m-%d')} по {params['end_date'].strftime('%Y-%m-%d')}:\n{injuries_text}"

        # Filter DataFrame to only include rows with these top injuries, sorted by description
        filtered_df = filtered_df[filtered_df['description'].isin(injury_counts.index)].sort_values(by='description')
        return answer, filtered_df

    else:  # 'list' type query
        if params['location']:
            answer = f"Знайдено {count_text}{' типу ' + params['injury_type'] if params['injury_type'] else ''} у радіусі {params['radius_km']} км від {params['location']} з {params['start_date'].strftime('%Y-%m-%d')} по {params['end_date'].strftime('%Y-%m-%d')}."
        else:
            answer = f"Знайдено {count_text}{' типу ' + params['injury_type'] if params['injury_type'] else ''} з {params['start_date'].strftime('%Y-%m-%d')} по {params['end_date'].strftime('%Y-%m-%d')}."

    return answer, filtered_df

@app.route('/api/query', methods=['POST'])
@jwt_required()
def query_injuries():
    data = request.get_json()
    user_query = data.get('query')

    print(f'user_query = {user_query}')

    if not user_query:
        return jsonify({"error": "Query is required."}), 400

    # Extract query parameters using LangChain
    query_params = extract_query_parameters_with_langchain(user_query)
    print(f'Extracted query parameters: {query_params}')

    # Build the SQL query based on the extracted parameters
    sql_query, params = build_sql_query(query_params)
    print(f'Generated SQL: {sql_query}')

    # Execute the query
    try:
        injury_data = pd.read_sql(sql_query, db.engine, params=params)
        print(f'injury data = {injury_data}')
    except Exception as e:
        print(f"Database error: {str(e)}")
        return jsonify({"error": f"Помилка бази даних: {str(e)}"}), 500

    if injury_data.empty:
        return jsonify({"answer": "Немає доступних даних.", "injuries": []}), 200

    # Generate the answer text using LangChain
    answer, injury_data = generate_answer(user_query, query_params, injury_data)
    injuries_list = injury_data.to_dict(orient='records')

    return jsonify({
        "answer": answer,
        "injuries": injuries_list
    })

@app.route('/api/verify-token', methods=['GET'])
@jwt_required()
def verify_token():
    return jsonify({"message": "Token is valid"}), 200


# Call the function at app startup
with app.app_context():
    initialize_retriever()

if __name__ == '__main__':
    app.run(debug=os.getenv('DEBUG', 'False').lower() in ['true', '1', 'yes'], port=5001, use_reloader=False)
