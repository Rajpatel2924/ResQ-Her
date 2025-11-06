import os
import pickle
import logging
from bson import Binary
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import List, Dict, Any

# Import local utility functions
from backend.utils.embedding import generate_text_embedding
from backend.logger import CustomFormatter # Assuming this file exists and works

# NOTE: load_dotenv() is called only once in main.py, not here.
load_dotenv() 
MONGO_ENDPOINT = os.getenv("MONGO_ENDPOINT")

db_client = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
try:
    handler.setFormatter(CustomFormatter())
except NameError:
    pass 
logger.addHandler(handler)


def get_database():
    """Connects to MongoDB and reuses the cached connection."""
    global db_client
    if db_client is None:
        try:
            db_client = MongoClient(MONGO_ENDPOINT)
            logger.info("Connected to the database")
        except Exception as e:
            logger.error(f"Error connecting to the database: {e}")
            return None
    return db_client["SheBuilds"]


def insert_data_into_db(
    embedding_model: Any, # New required argument
    name, location, contact_info, severity, culprit, relationship_to_culprit, other_info
) -> Any:
    """Inserts a document into the 'complains2' collection with embeddings."""
    db = get_database()
    if db is None:
        print("Database connection is not available.")
        return None

    collection = db["complains2"]
    document = {
        "name": name, "location": location, "contact_info": contact_info, 
        "severity": severity, "culprit": culprit, 
        "relationship_to_culprit": relationship_to_culprit, 
        "other_info": other_info, "status": "Pending",
    }
    
    # FIXED: Must pass the model instance here
    culprit_embedding = generate_text_embedding(embedding_model, culprit) 
    document["culprit_embedding"] = culprit_embedding
    
    try:
        result = collection.insert_one(document)
        return result.inserted_id
    except Exception as e:
        print("Error inserting data:", e)
        return None


# Function to upload embeddings to MongoDB
def upload_embeddings_to_mongo(file_contents: List[tuple], embedding_model: Any):
    """Uploads RAG documents with embeddings to MongoDB."""
    db = get_database()
    collection = db["doc_embedding"]
    
    for filename, content in file_contents:
        # FIXED: Must pass the model instance here
        embedding = generate_text_embedding(embedding_model, content) 

        doc = {
            "filename": filename,
            "embedding": Binary(pickle.dumps(embedding)), 
            "content": content[:500], 
        }

        collection.insert_one(doc)
        print(f"Uploaded {filename} to MongoDB.")