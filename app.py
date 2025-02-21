import os
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

# Initialize Gemini LLM with API Key
def initialize_llm():
    gemini_api_key =os.getenv("GEMINI_API_KEY")  # Use environment variable for security
    model_name = "gemini-1.5-pro-latest"
    llm = ChatGoogleGenerativeAI(api_key=gemini_api_key, model=model_name)
    return llm

prompt_template = PromptTemplate(
    input_variables=["user_query"],
    template="""
    You are an experienced financial planner. Your task is to provide clear and comprehensive advice on financial investments.
    Only answer queries strictly related to finance, investments, or financial planning. If the query is unrelated, respond with:
    'This query is not related to finance. Please ask questions about financial investments or planning.'

    Query: {user_query}

    Provide actionable steps, potential risks, and benefits if applicable.
    
    IMPORTANT: Format your response in plain text only. Do not use markdown.
    """
)

def create_financial_planner_chain(llm):
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain

llm = initialize_llm()
financial_planner_chain = create_financial_planner_chain(llm)


"""@app.route('/')
def index():
    return render_template('index.html')"""


@app.route('/ask', methods=['POST'])
def handle_query():
    data = request.json
    user_query = data.get("user_query", "")
    
    if not user_query:
        return jsonify({"error": "No query provided."}), 400

    response = financial_planner_chain.run({"user_query": user_query})
    
    return jsonify({"response": response})

# New route to fetch YouTube video links
youtubeApi=os.getenv("YOUTUBE_API_KEY") # Use environment variable for security
baseUrl=os.getenv("BASE_URL")

@app.route('/get_videos', methods=['POST'])
def get_video_links():
    data = request.json
    question = data.get("question", "")
    
    if not question:
        return jsonify({"error": "No query provided."}), 400

    params = {
        'part': 'snippet',
        'q': f'{question} telugu',  # Add Telugu filter
        'key': youtubeApi,
        'type': 'video',
        'maxResults': 5,
        'relevanceLanguage': 'te',  # Telugu language code
        'regionCode': 'IN'  # India region
    }

    response = requests.get(baseUrl, params=params)
    data = response.json()

    if 'error' in data:
        return jsonify({"error": data['error']['message']}), 400

    video_links = [f'https://www.youtube.com/watch?v={item["id"]["videoId"]}' 
                 for item in data.get('items', [])]
    
    return jsonify({"videos": video_links})

if __name__ == "__main__":
    app.run(debug=True)