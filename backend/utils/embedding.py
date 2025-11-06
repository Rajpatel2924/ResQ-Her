import numpy as np
from typing import List, Dict, Optional
from pymongo.collection import Collection

# --- NOTE: Removed google.generativeai and environment loading ---


def generate_text_embedding(embedding_model, text: str) -> Optional[List[float]]:
    """
    Generates an embedding vector using the SentenceTransformer model instance passed in.
    """
    if embedding_model is None:
        # Fallback if the model in main.py failed to load
        return [0.0] * 384
    
    # Use the passed SentenceTransformer instance
    # Returns a list of floats (required for MongoDB Atlas Vector Search)
    return embedding_model.encode(text, convert_to_numpy=False).tolist()


def calculate_similarity_percentage(query_vector: List[float], result_vector: List[float]) -> float:
    """Calculates Euclidean distance-based similarity. Kept as is."""
    distance = sum((q - r) ** 2 for q, r in zip(query_vector, result_vector)) ** 0.5
    max_distance = len(query_vector) ** 0.5 
    similarity_percentage = max(0, (1 - distance / max_distance) * 100)
    return round(similarity_percentage, 2)


def find_top_matches(
    collection: Collection, description_embedding: List[float], num_results: int = 1, num_candidates: int = 100
) -> List[Dict]:
    """
    Performs a vector search using MongoDB's $vectorSearch aggregation pipeline.
    """
    # NOTE: The $project stage has been simplified to fix the import error.
    try:
        results_cursor = collection.aggregate(
            [
                {
                    "$vectorSearch": {
                        "path": "culprit_embedding", 
                        "index": "culpritIndex2",   
                        "queryVector": description_embedding,
                        "numResults": num_results,
                        "numCandidates": num_candidates,
                        "similarity": "euclidean",
                        "type": "knn",
                    },
                },
                {"$limit": num_results}, 
                {"$project": {"culprit": 1, "_id": 1, "score": {"$meta": "vectorSearchScore"}}},
            ]
        )
        return list(results_cursor)
    except Exception as e:
        print(f"VECTOR SEARCH WARNING: Returning empty list due to error: {e}")
        return []