# ChatGroq Flask Chatbot

This is a Flask-based chatbot application featuring emotion detection, PDF document search, and user management with MongoDB integration. The chatbot leverages a TensorFlow emotion detection model and Groq API for advanced language processing.

---

## Folder Structure and Required Files

To ensure the application works properly, please follow this folder structure:

- **Model file:** Place the emotion detection model file named `emotion_model222.keras` inside the `model/` folder.
- This file should be uploaded from this link https://drive.google.com/drive/folders/1CEEQXcDGUGU3Vq8nnB8m9rl5si3D4pOT?usp=drive_link
- **Tokenizer file:** Make sure `tokenizer.json` is located in the root folder or the expected path as defined in the code.
## MongoDB Setup (Mandatory)

1. **Create a MongoDB account:**  
   Use a cloud service like [MongoDB Atlas](https://www.mongodb.com/atlas) to create a free account and set up your cluster.

2. **Create a database and user:**  
   Set up a database and generate a database user with access privileges.

3. **Obtain your connection string:**  

4. **Replace the connection string in the code:**  
Open `app.py` (or the relevant configuration file) and replace the placeholder MongoDB URI with your own.  
This step is critical for your app to store and retrieve user data and chat history.

---

## Environment Variables and API Keys

Before running the application, replace these variables with your actual credentials:

```python
GROQ_API_KEY = "your_groq_api_key_here"
MONGO_URI = "your_mongodb_connection_string_here"
FLASK_SECRET_KEY = "your_flask_secret_key_here"
GOOGLE_CLIENT_ID = "your_google_client_id_here"
GOOGLE_CLIENT_SECRET = "your_google_client_secret_here"


Then try:
----pip install -r requirements.txt
----cd mental_health_suivi
----python app.py
   
