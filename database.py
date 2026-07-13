import chromadb
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

# Download NLTK resources
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg)

# ---------------- CONFIG ----------------
CSV_PATH = r"C:\Users\HP\Downloads\medical_question_answer_dataset_50000_modify.csv"
DB_PATH = r"C:\Users\HP\Desktop\code\ml\api\medical_vector_db"
COLLECTION_NAME = "medical_data"
# ----------------------------------------

# Load dataset
df = pd.read_csv(CSV_PATH, encoding="cp1252")

# Fill missing values
df["symptoms"] = df["symptoms"].fillna("")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    words = [lemmatizer.lemmatize(w, pos="v") for w in words]

    return " ".join(words)


# Clean symptoms
df["symptoms"] = df["symptoms"].apply(clean_text)

# Embedding model
model = SentenceTransformer("intfloat/e5-base-v2")

# E5 models require passage prefix
embeddings = model.encode(
    ["passage: " + x for x in df["symptoms"].tolist()],
    normalize_embeddings=True,
    show_progress_bar=True
)

# Create Chroma database
client = chromadb.PersistentClient(path=DB_PATH)

# Delete old collection if it exists
try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# Prepare data
ids = [str(i) for i in range(len(df))]
documents = df["symptoms"].tolist()

metadatas = [
    {
        "disease": str(row["Disease Prediction"]),
        "medicines": str(row["Recommended Medicines"]),
        "advice": str(row["Advice"])
    }
    for _, row in df.iterrows()
]

# Insert in batches
batch_size = 500

for start in range(0, len(df), batch_size):
    end = min(start + batch_size, len(df))

    collection.add(
        ids=ids[start:end],
        documents=documents[start:end],
        embeddings=embeddings[start:end].tolist(),
        metadatas=metadatas[start:end]
    )

    print(f"Inserted {end}/{len(df)}")

print("Database created successfully!")
print("Total records:", collection.count())