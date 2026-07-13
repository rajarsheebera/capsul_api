# 🩺 Medicine Recommendation System

An AI-powered Medicine Recommendation System that uses NLP, embeddings,
and vector search to recommend medicines and provide medical advice
based on user symptoms.

## 🚀 Features

-   Predict diseases from user symptoms.
-   Recommend medicines based on similar medical cases.
-   Provide symptom descriptions and medical advice.
-   Semantic search using embeddings.
-   Fast retrieval using a vector database.
-   REST API built with FastAPI.

## 🛠️ Technologies Used

-   Python
-   FastAPI
-   ChromaDB
-   Sentence Transformers
-   Pandas
-   NumPy
-   Scikit-learn
-   Uvicorn

## 📂 Project Structure

``` text
Medicine_Recommendation_System/
│
├── app.py
├── embedding.py
├── predictor.py
├── database.py
├── medical.csv
├── requirements.txt
└── medical_vector_db/
```

## ⚙️ Installation

``` bash
cd your-repository-name
pip install -r requirements.txt
python embedding.py
uvicorn app:app --reload
```

## 🌐 API Endpoint

### POST `/predict`

Example request:

``` json
{
  "symptoms": "high fever, headache, cough"
}
```

## ⚠️ Disclaimer

This project is for educational purposes only and should not replace
professional medical advice.

## 👨‍💻 Author

Rajarshee Bera\
B.Tech CSE, Brainware University
