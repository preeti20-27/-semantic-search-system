# Semantic Search System (20 Newsgroups)

This project implements a **semantic search system** using the **20 Newsgroups dataset (~20,000 documents)**.

The system embeds documents into vector space, clusters them using a **Gaussian Mixture Model**, and performs efficient semantic retrieval using a **FAISS vector index**.

A **custom semantic cache** is implemented from scratch to avoid redundant vector searches for semantically similar queries.

---

## Features

* SentenceTransformer embeddings for semantic representation
* FAISS vector database for fast similarity search
* Fuzzy clustering using Gaussian Mixture Model (GMM)
* Cluster-aware semantic cache (no Redis or external cache)
* FastAPI service exposing query and cache endpoints

---

## Semantic Cache Strategy

1. A query is embedded using **SentenceTransformers**.
2. The embedding is assigned to a cluster using **Gaussian Mixture Model**.
3. Cached queries in the same cluster are compared using **cosine similarity**.
4. If similarity > **0.9**, cached results are returned instead of performing a new FAISS search.

This avoids redundant computation and improves query response time.

---
## System Architecture Diagram
                ┌─────────────────────┐
                │     User Query      │
                └──────────┬──────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │ SentenceTransformer Embedding  │
          │      (all-MiniLM-L6-v2)        │
          └──────────┬─────────────────────┘
                     │
                     ▼
        ┌───────────────────────────────┐
        │ Gaussian Mixture Model (GMM)  │
        │     Predict Query Cluster     │
        └──────────┬────────────────────┘
                   │
                   ▼
        ┌───────────────────────────────┐
        │  Cluster-Aware Semantic Cache │
        │   (Store queries per cluster) │
        └──────────┬────────────────────┘
                   │
        ┌──────────┴───────────┐
        │                      │
        ▼                      ▼
 ┌───────────────┐      ┌──────────────────┐
 │  Cache Hit    │      │   Cache Miss     │
 │ similarity>0.9│      │                  │
 └──────┬────────┘      └────────┬─────────┘
        │                         │
        ▼                         ▼
┌───────────────┐        ┌─────────────────┐
│ Return Cached │        │  FAISS Vector   │
│    Results    │        │      Search     │
└──────┬────────┘        └────────┬────────┘
       │                          │
       ▼                          ▼
                     ┌────────────────────────┐
                     │ Store Result in Cache  │
                     └──────────┬─────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Return Results  │
                       └─────────────────┘

## Project Structure

```
semantic-search-system
│
├── main.py              # FastAPI service
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── embeddings.npy       # Document embeddings
├── faiss_index.bin      # FAISS vector index
└── gmm_model.pkl        # Trained clustering model
```

---

## API Endpoints

### POST `/query`

Accepts a JSON query and returns semantic search results.

Example request:

```json
{
  "query": "space shuttle launch"
}
```

Example response:

```json
{
  "query": "space shuttle launch",
  "cache_hit": true,
  "similarity_score": 0.99,
  "result": [4511, 8214, 14635, 2018, 1520],
  "dominant_cluster": 19
}
```

---

### GET `/cache/stats`

Returns cache usage statistics.

### DELETE `/cache`

Clears the semantic cache.

---

## Setup Virtual Environment

Create and activate a virtual environment.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Open API documentation:

```
http://127.0.0.1:8000/docs
```

---

## Docker Support

Build the container:

```bash
docker build -t semantic-search-system .
```

Run the container:

```bash
docker run -p 8000:8000 semantic-search-system
```

---

## Model and Index Files

The following files are generated during preprocessing and are not included in the repository due to GitHub file size limits:

* embeddings.npy
* faiss_index.bin
* gmm_model.pkl

These files contain the document embeddings, FAISS vector index, and the trained clustering model used by the API.


