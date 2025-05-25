import os
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from datetime import datetime, timezone
from collections import Counter
import json
import numpy as np
import uuid
from pymongo import MongoClient
from authlib.integrations.flask_client import OAuth
import bcrypt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableMap

# Optionnel, pour charger les variables depuis .env automatiquement
from dotenv import load_dotenv
load_dotenv()

np.float_ = np.float64

app = Flask(__name__)
CORS(app)

# Utiliser la variable d'environnement pour la clé secrète Flask
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change_this_secret_in_production")

from flask_session import Session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
Session(app)

# === Google OAuth config via variables d'environnement ===
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    client_kwargs={'scope': 'email profile', 'prompt': 'consent'}
)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "emotion_model222.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.json")
PDF_PATH = os.path.join(BASE_DIR, "pdfs/mental1")
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")
DATA_PATH = os.path.join(BASE_DIR, "saved_data.json")
MAX_LEN = 100


mongo_uri = "mongodb+srv://amal:amal@moodmade.tgnklnz.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(mongo_uri)
db = client.chat_db
user_collection = db.users
chat_collection = db.chat_history

label_map = {0: "anger", 1: "joy", 2: "love", 3: "sadness"}
emoji_map = {"joy": "😊", "sadness": "😢", "anger": "😠", "love": "😍"}

try:
    model = load_model(MODEL_PATH)
    print("✅ Emotion model loaded")
except Exception as e:
    print(f"❌ Error loading model: {e}")

try:
    with open(TOKENIZER_PATH) as f:
        tokenizer = tokenizer_from_json(f.read())
    print("✅ Tokenizer loaded")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")

def predict_emotion(text):
    if not model or not tokenizer:
        return "unknown"
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_LEN)
    prediction = model.predict(padded)
    label_index = int(np.argmax(prediction))
    return label_map.get(label_index, "unknown")

def clean_response(text):
    lines = text.split('\n')
    return '\n'.join([l.strip() for l in lines if l.strip()])

llm = ChatGroq(
    temperature=0.3,
    groq_api_key=os.getenv("GROQ_API_KEY", "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"),
    model_name="llama3-8b-8192",
    max_tokens=1024
)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

if not os.path.exists(FAISS_PATH):
    loader = DirectoryLoader(PDF_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    texts = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
    vector_db = FAISS.from_documents(texts, embeddings)
    vector_db.save_local(FAISS_PATH)
else:
    vector_db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

retriever = vector_db.as_retriever(search_kwargs={"k": 3})
retrieval_chain = RunnableMap({
    "context": lambda x: retriever.invoke(x["question"]),
    "chat_history": RunnablePassthrough(),
    "question": RunnablePassthrough()
})

from langchain.prompts import PromptTemplate

from langchain.prompts import PromptTemplate

from langchain.prompts import PromptTemplate

from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template="""
You are a highly empathetic and clinically informed AI psychiatric assistant, specialized in emotional and psychological support during vulnerable moments such as grief, trauma, anxiety, or emotional pain.

You operate in an IRAC (Information Retrieval Augmented Conversation) system and have access to reliable, evidence-based psychiatric and psychological documents (context). Your goal is to provide **deep reassurance grounded in psychiatric knowledge**, helping the user understand their emotions while feeling truly supported and validated.

Guidelines:

- Begin by sincerely acknowledging and validating the user’s emotional experience, reflecting their feelings with clinical empathy.
- Use psychiatric concepts (e.g., attachment patterns, trauma responses, cognitive-behavioral insights, stages of grief) **when relevant and helpful**, naturally woven into your response to deepen reassurance.
- Reference research, clinical findings, or real case examples indirectly and respectfully—phrases like “Studies in psychiatry show...”, “Clinical evidence suggests...”, or “It’s common in therapeutic practice to see...” are appropriate. Avoid saying the user “provided” documents.
- Normalize the user’s experience by contextualizing it within known psychiatric phenomena, helping them understand that their reactions are valid and part of a human response.
- Avoid shallow or generic reassurances. Instead, provide thoughtful, nuanced understanding and hope based on psychiatric expertise.
- Offer gentle psychological strategies or coping mechanisms from the context only if they fit naturally and sensitively.
- Never rush into advice or problem-solving; focus on deep emotional presence and understanding first.
- Use natural paraphrasing of the context; do not copy documents verbatim.
- Maintain a calm, warm, and human-like tone—never cold, robotic, or condescending.

Context (psychiatric and psychological documents):  
{context}

Chat History:  
{chat_history}

User Message:  
{question}

Answer:
"""
)




qa_chain = retrieval_chain | prompt_template | llm | StrOutputParser()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "")
    user_id = data.get("user_id", "anonymous")
    conversation_id = data.get("conversation_id") or str(uuid.uuid4())

    if not question:
        return jsonify({"error": "Empty question"}), 400

    docs = retriever.invoke(question)
    context_texts = "\n\n".join([doc.page_content for doc in docs])
    chat_log = [
        f"User: {d['text']}" if d['sender'] == 'user' else f"Bot: {d['text']}"
        for d in chat_collection.find({"conversation_id": conversation_id}, sort=[("timestamp", 1)])
    ]

    raw_response = qa_chain.invoke({
        "question": question,
        "chat_history": "\n".join(chat_log[-5:]),
        "context": context_texts
    })

    response = clean_response(raw_response)
    emotion = predict_emotion(question)
    now = datetime.now(timezone.utc) + timedelta(hours=1)

    chat_collection.insert_many([
        {"user_id": user_id, "sender": "user", "text": question, "emotion": None, "conversation_id": conversation_id, "timestamp": now},
        {"user_id": user_id, "sender": "bot", "text": response, "emotion": emotion, "conversation_id": conversation_id, "timestamp": now},
    ])

    today_str = now.strftime("%Y-%m-%d")
    json_path = os.path.join(BASE_DIR, "saved_data.json")

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            try:
                saved_data = json.load(f)
            except Exception:
                saved_data = {}
    else:
        saved_data = {}

    if today_str not in saved_data:
        saved_data[today_str] = {}
    if user_id not in saved_data[today_str]:
        saved_data[today_str][user_id] = []

    saved_data[today_str][user_id].append(emotion)

    with open(json_path, "w") as f:
        json.dump(saved_data, f, indent=2)

    return jsonify({
        "response": response,
        "emotion": emotion,
        "conversation_id": conversation_id
    })

@app.route("/weekly-emotions/<user_id>", methods=["GET"])
def weekly_emotions(user_id):
    json_path = os.path.join(BASE_DIR, "saved_data.json")
    if not os.path.exists(json_path):
        return jsonify([])

    with open(json_path, "r") as f:
        saved_data = json.load(f)

    user_emotions = []
    for date_str, users in saved_data.items():
        if user_id not in users:
            continue
        emotions = users[user_id]
        counts = Counter(emotions)
        dominant_emotion = counts.most_common(1)[0][0]
        emoji = emoji_map.get(dominant_emotion, "")

        user_emotions.append({
            "date": date_str,
            "emotion": dominant_emotion,
            "emoji": emoji
        })

    return jsonify(user_emotions)

@app.route("/conversations/<user_id>", methods=["GET"])
def get_conversations(user_id):
    conversations = chat_collection.aggregate([
        {"$match": {"conversation_id": {"$ne": None}}},
        {"$group": {"_id": "$conversation_id", "latest": {"$max": "$timestamp"}}},
        {"$sort": {"latest": -1}},
    ])

    result = []
    for conv in conversations:
        user_message = chat_collection.find_one({"conversation_id": conv["_id"], "user_id": user_id})
        if user_message:
            result.append({
                "conversation_id": conv["_id"],
                "latest": conv["latest"].isoformat() if conv["latest"] else ""
            })

    return jsonify(result)

@app.route("/conversation/<conversation_id>", methods=["GET"])
def get_conversation(conversation_id):
    messages = list(chat_collection.find({"conversation_id": conversation_id}, sort=[("timestamp", 1)]))
    for msg in messages:
        msg["_id"] = str(msg["_id"])
    return jsonify(messages)

@app.route("/start_new_conversation", methods=["POST"])
def start_new_conversation():
    return jsonify({"conversation_id": str(uuid.uuid4())})
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    full_name = data.get("full_name")
    email = data.get("email")
    password = data.get("password")

    if not full_name or not email or not password:
        return jsonify({"error": "All fields are required."}), 400

    if user_collection.find_one({"email": email}):
        return jsonify({"error": "Email already exists."}), 400

    existing_users = user_collection.find()
    for user in existing_users:
        if user["full_name"] == full_name and bcrypt.checkpw(password.encode('utf-8'), user["password"].encode('utf-8')):
            return jsonify({"error": "Name and password already exist."}), 409  # Conflict

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    user_id = str(uuid.uuid4())
    user_collection.insert_one({
        "user_id": user_id,
        "full_name": full_name,
        "email": email,
        "password": hashed_password.decode('utf-8')
    })

    return jsonify({"message": "User registered successfully", "user_id": user_id}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    user = user_collection.find_one({"email": email})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        return jsonify({"message": "Login successful", "user_id": user['user_id'], "full_name": user['full_name']}), 200
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/auth/google")
def google_login():
    redirect_uri = "http://localhost:5000/auth/google/callback"
    return google.authorize_redirect(redirect_uri)

@app.route("/auth/google/callback")
def google_callback():
    token = google.authorize_access_token()
    resp = google.get("userinfo")
    user_info = resp.json()

    email = user_info["email"]
    name = user_info.get("name", "")

    user = user_collection.find_one({"email": email})
    user_id = user["user_id"] if user else str(uuid.uuid4())

    if not user:
        user_collection.insert_one({"user_id": user_id, "full_name": name, "email": email, "password": ""})

    return redirect(f"http://localhost:4200/google-success?user_id={user_id}&full_name={name}")


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
