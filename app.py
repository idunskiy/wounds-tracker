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

from utils import initialize_retriever, get_coordinates, haversine, extract_query_parameters, get_most_frequent_injuries, parse_time_frame, extract_query_parameters_with_langchain, extract_injury_type, build_sql_query, generate_answer

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
