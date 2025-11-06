# ==============================================================================
# COMBINED & FREE-ONLY FASTAPI APPLICATION (main.py)
# ==============================================================================

# Built-in libraries
import base64
import json
import logging
import os
from io import BytesIO

# External dependencies (Free/Open-Source/Self-Hosted)
from bson import ObjectId
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv

# Database Imports (Free/Self-Hosted)
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure as MongoConnectionError, ServerSelectionTimeoutError 

# LLM and Vector Imports (Open-Source/Self-Hosted)
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

# Placeholder for Mock Data (Replace with your actual mock_legal_data.py)
try:
    from mock_legal_data import LEGAL_DATA
except ImportError:
    LEGAL_DATA = ["Law on domestic violence.", "Information on filing a police report.", "Legal aid contact details."]
    print("WARNING: mock_legal_data.py not found. Using minimal placeholder data.")


# --- Project-Specific Internal Imports ---
# NOTE: These modules must now be compatible with the free, model-passing structure.
from backend.db import get_database, upload_embeddings_to_mongo # Added upload_embeddings_to_mongo import
from backend.logger import CustomFormatter 
from backend.schema import FileContent, PostInfo 
from backend.utils.common import (load_image_from_url_or_file,
                                  read_files_from_directory,
                                  serialize_object_id)
from backend.utils.embedding import find_top_matches, generate_text_embedding # These functions now take 'embedding_model'
from backend.utils.regex_ptr import extract_info
from backend.utils.steganography import (decode_text_from_image,
                                         encode_text_in_image)


# ==============================================================================
# 1. INITIALIZATION AND CONFIGURATION
# ==============================================================================
load_dotenv()

# Logger Setup (from the first file)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
try:
    handler.setFormatter(CustomFormatter())
except NameError:
    # Fallback if CustomFormatter is not defined/imported correctly
    pass
logger.addHandler(handler)

app = FastAPI(title="ResQ-Her Backend API (Free/Open-Source)")

# CORS Middleware (Combined from both files)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_ENDPOINT = os.getenv("MONGO_ENDPOINT")

# Global Database and Model Clients
mongo_client: MongoClient = None
db = None
embedding_model: SentenceTransformer = None # The model for RAG


# ==============================================================================
# 2. DATABASE AND MODEL CONNECTION HANDLERS
# ==============================================================================

@app.on_event("startup")
def startup_db_client():
    """Connects to MongoDB and loads the embedding model (SentenceTransformer)."""
    global mongo_client, db, embedding_model

    # 2a. MongoDB Connection (Using logic from the second file for robustness)
    if not MONGO_ENDPOINT:
        logger.error("FATAL ERROR: MONGO_ENDPOINT not found. Database will be inaccessible.")
    else:
        try:
            # Use the environment variable to connect
            mongo_client = MongoClient(MONGO_ENDPOINT)
            db = mongo_client.get_database("resqher_db") 
            db.command('ping')
            logger.info("Successfully connected to MongoDB Atlas!")
        except (MongoConnectionError, ServerSelectionTimeoutError, Exception) as e:
            logger.error(f"FATAL ERROR: Could not connect to MongoDB Atlas. Error: {e}")
            mongo_client = None
            db = None

    # 2b. Load Embedding Model (Open-Source for Zero Cost)
    try:
        model_name = 'all-MiniLM-L6-v2'
        embedding_model = SentenceTransformer(model_name)
        logger.info(f"Successfully loaded embedding model: {model_name}")
    except Exception as e:
        logger.error(f"FATAL ERROR: Could not load SentenceTransformer model. Error: {e}")
        embedding_model = None


@app.on_event("shutdown")
def shutdown_db_client():
    """Closes the MongoDB connection."""
    global mongo_client
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB client closed.")


# ==============================================================================
# 3. DATA MODELS (From both files)
# ==============================================================================
# PostInfo is imported from backend.schema 

class UserInput(BaseModel):
    keywords: str

class ExpandedMessage(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    expanded_text: str
    # Pydantic fix for 'model_' protected namespace (optional, but good practice)
    # from pydantic import ConfigDict
    # model_config = ConfigDict(protected_namespaces=()) 
    model_used: str 

class LawBotRequest(BaseModel):
    question: str

class LawBotResponse(BaseModel):
    answer: str
    sources: List[str]


# ==============================================================================
# 4. UTILITY FUNCTIONS (Adapted for free/local execution)
# ==============================================================================

def mock_llm_expand_text(input_text: str) -> str:
    """MOCK function to replace proprietary LLM expansion."""
    return (
        "URGENT REQUEST FOR ASSISTANCE. Based on the provided summary, a situation "
        "requiring immediate attention has been reported. The details are: "
        f"{input_text.replace('\n', ', ')}. Please dispatch help immediately and "
        "contact the individual using the preferred method listed in the details."
    )

def mock_llm_generate_answer(question: str, context: str) -> str:
    """MOCK function to replace proprietary LLM answering (Law Bot)."""
    if context and "No specific legal data found" not in context:
        return (
            f"Based on our local legal knowledge base, the following relevant information "
            f"was found regarding your question: '{question}'.\n\n"
            f"Context Snippet: {context[:200]}..." 
        )
    return "I am sorry, but without access to a powerful online LLM, I can only search my local knowledge base. The search for this question did not yield a relevant answer in the stored documents."


# ==============================================================================
# 5. RAG/DB UTILITY ENDPOINTS
# ==============================================================================

@app.post("/api/load_legal_data")
def load_legal_data_endpoint():
    """Converts mock legal data into vectors and loads them into MongoDB."""
    global db, embedding_model
    if db is None or embedding_model is None:
        raise HTTPException(status_code=500, detail="Database or Embedding Model not available.")

    try:
        text_chunks = LEGAL_DATA
        # NOTE: This uses the direct SentenceTransformer.encode method, which is correct here.
        embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True) 
        
        documents = []
        for text, embedding in zip(text_chunks, embeddings):
            documents.append({
                "text": text,
                "embedding": embedding.tolist()
            })

        db.legal_knowledge.delete_many({})
        result = db.legal_knowledge.insert_many(documents)

        return {"message": f"Successfully loaded {len(result.inserted_ids)} legal documents.", 
                "note": "Remember to manually configure a Vector Search Index in MongoDB Atlas if using cloud DB."}

    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise HTTPException(status_code=500, detail=f"Data loading failed: {e}")


@app.post("/upload_embeddings/")
async def upload_embeddings():
    """Upload embeddings to MongoDB (Utility from the first file)."""
    global embedding_model
    try:
        file_contents = read_files_from_directory("backend/docs")
        
        # --- FIX APPLIED: Pass the global embedding_model to the utility function ---
        upload_embeddings_to_mongo(file_contents, embedding_model) 
        
        return {"message": "Embeddings uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading embeddings: {e}")


@app.get("/find-match")
def find_top_matching_posts(info: str, collection: str):
    """Find top matches based on embedding similarity (from the first file)."""
    global db, embedding_model
    try:
        # --- FIX APPLIED: Pass the global embedding_model to the embedding function ---
        description_vector = generate_text_embedding(embedding_model, info) 
        
        # Use the local RAG match function
        top_matches = find_top_matches(db[collection], description_vector)
        return [serialize_object_id(match) for match in top_matches]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding matches: {e}")


# ==============================================================================
# 6. CORE ENDPOINTS (Refactored for Free/Local execution)
# ==============================================================================

@app.post("/text-generation")
async def get_post_and_expand_its_content(post_info: PostInfo):
    """Expand user input text for help message generation using local mock LLM."""
    try:
        concatenated_text = (
            f"Name: {post_info.name}\nPhone: {post_info.phone}\nLocation: {post_info.location}\n"
            f"Duration of Abuse: {post_info.duration_of_abuse}\nFrequency of Incidents: {post_info.frequency_of_incidents}\n"
            f"Preferred Contact Method: {post_info.preferred_contact_method}\nCurrent Situation: {post_info.current_situation}\n"
            f"Culprit Description: {post_info.culprit_description}\nCustom Text: {post_info.custom_text}"
        )
        
        # Uses mock LLM function
        expanded_text = mock_llm_expand_text(concatenated_text)
        
        return {
            "mock_local_response": expanded_text,
            "note": "Proprietary LLMs (Gemini/Gemma) have been replaced with a local mock function."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error expanding text: {e}")


@app.post("/api/law_bot", response_model=LawBotResponse)
async def law_bot_endpoint(request: LawBotRequest):
    """Answers legal questions by performing RAG using local SentenceTransformer and Mock LLM."""
    global db, embedding_model

    if db is None or embedding_model is None:
        raise HTTPException(status_code=500, detail="Law Bot prerequisites (DB/Model) are not met.")

    try:
        # --- RAG STEP 1: EMBED THE QUERY ---
        # NOTE: This is correct, as it uses the SentenceTransformer instance directly.
        query_vector = embedding_model.encode(request.question, convert_to_numpy=True).tolist()

        # --- RAG STEP 2: MOCK VECTOR SEARCH ---
        relevant_docs = db.legal_knowledge.find().limit(2) 
        context_list = [doc['text'] for doc in relevant_docs]
        context_str = "\n---\n".join(context_list)
        
        if not context_str:
            context_str = "No specific legal data found in the knowledge base."

        # --- RAG STEP 3: GENERATE ANSWER WITH CONTEXT (MOCK) ---
        answer = mock_llm_generate_answer(request.question, context_str)
        
        return LawBotResponse(
            answer=answer,
            sources=context_list
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Law Bot query failed: {e}")


@app.post("/text-decomposition")
async def decompose_text_content(data: dict):
    """Decompose and extract information from user text."""
    try:
        text = data.get("text")
        # NOTE: decompose_user_text must be implemented with local logic/regex.
        decomposed_text = decompose_user_text(text) 
        return {"extracted_data": extract_info(decomposed_text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error decomposing text: {e}")


# ==============================================================================
# 7. IMAGE AND STENOGRAPHY ENDPOINTS (Free/Local)
# ==============================================================================

@app.post("/encode")
async def encode_text_in_image_endpoint(
    text: str, img_url: str = None, file: UploadFile = File(None)
):
    """Encode text into an image (Steganography)."""
    try:
        image = load_image_from_url_or_file(img_url, file)
        encoded_image = encode_text_in_image(image, text)
        
        img_buffer = BytesIO()
        encoded_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        return StreamingResponse(
            img_buffer,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=encoded_image.png"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error encoding text in image: {e}"
        )


@app.post("/decode")
async def decode_text_from_image_endpoint(
    img_url: str = None, file: UploadFile = File(None)
):
    """Decode text from an image (Steganography)."""
    try:
        image = load_image_from_url_or_file(img_url, file)
        return {"decoded_text": decode_text_from_image(image)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error decoding text from image: {e}"
        )


# ==============================================================================
# 8. POST MANAGEMENT ENDPOINTS (Unchanged DB logic)
# ==============================================================================

@app.post("/save-extracted-data")
async def save_extracted_data(data: dict):
    """Save decomposed/extracted data to the database."""
    try:
        db["admin"].insert_one(data)
        return {"status": "Data saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving data: {e}")


@app.get("/get-admin-posts")
def get_all_posts():
    """Retrieve all posts from the database."""
    try:
        posts = [serialize_object_id(post) for post in db["admin"].find()]
        return JSONResponse(content=posts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving posts: {e}")


@app.get("/get-post/{post_id}")
def get_post_by_id(post_id: str):
    """Retrieve a specific post by its ID."""
    try:
        post = db["admin"].find_one({"_id": ObjectId(post_id)})
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        return JSONResponse(content=serialize_object_id(post))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving post by ID: {e}")

@app.post("/close-issue/{issue_id}")
async def close_issue(issue_id: str):
    """Mark an issue as closed by updating its status."""
    try:
        result = db["admin"].update_one(
            {"_id": ObjectId(issue_id)},
            {"$set": {"status": "closed"}}
        )
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Issue not found or already closed")
        return {"status": "Issue marked as closed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error closing issue: {e}")


# ==============================================================================
# 9. ROOT ENDPOINT
# ==============================================================================

@app.get("/")
def read_root():
    return {"status": "ResQ-Her Backend running successfully (Free/Open-Source)", 
            "api_version": "v1.0"}