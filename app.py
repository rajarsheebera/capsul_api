from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
DB_PATH = r"C:\Users\HP\Desktop\code\ml\api\medical_vector_db"
COLLECTION_NAME = "medical_data"
# ----------------------------------------

app = FastAPI(title="capsul api", description="Medical API for disease prediction based on symptoms")

# Load model
model = SentenceTransformer("intfloat/e5-base-v2")

# Connect to Chroma
client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_collection(
    name=COLLECTION_NAME
)


class Symptoms(BaseModel):
    symptoms: str


@app.get("/")
def home():
    return {"message": "Medical API Running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "records": collection.count()
    }

@app.post("/predict")
def predict(data: Symptoms):

    query_embedding = model.encode(
        "query: " + data.symptoms,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    metadata = results["metadatas"][0][0]
    matched_symptoms = results["documents"][0][0]
    distance = results["distances"][0][0]

    confidence = round(1 - distance, 3)

    return {
        "disease": metadata.get("disease", "Unknown"),
        "medicine": metadata.get("medicines", "Not available"),
        "advice": metadata.get("advice", "No advice available"),
        "matched_symptoms": matched_symptoms,
        "confidence": confidence
    }