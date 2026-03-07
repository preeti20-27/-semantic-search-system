from fastapi import FastAPI
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("faiss_index.bin")

with open("gmm_model.pkl", "rb") as f:
    gmm = pickle.load(f)

cluster_cache = {i: [] for i in range(20)}

cache_stats = {"hit": 0, "miss": 0}


def semantic_search(query_embedding, top_k=5):
    distances, indices = index.search(query_embedding, top_k)
    return indices[0].tolist()


def get_query_cluster(query_embedding):
    probs = gmm.predict_proba(query_embedding)
    return int(np.argmax(probs))


@app.post("/query")
def query_api(data: dict):

    query = data["query"]

    query_embedding = model.encode([query])

    cluster_id = get_query_cluster(query_embedding)

    cached_entries = cluster_cache[cluster_id]

    for entry in cached_entries:

        sim = cosine_similarity(query_embedding, entry["embedding"])[0][0]

        if sim > 0.9:
            cache_stats["hit"] += 1

            return {
                "query": query,
                "cache_hit": True,
                "matched_query": entry["query"],
                "similarity_score": float(sim),
                "result": entry["result"],
                "dominant_cluster": cluster_id
            }

    cache_stats["miss"] += 1

    results = semantic_search(query_embedding)

    cluster_cache[cluster_id].append({
        "query": query,
        "embedding": query_embedding,
        "result": results
    })

    return {
        "query": query,
        "cache_hit": False,
        "result": results,
        "dominant_cluster": cluster_id
    }


@app.get("/cache/stats")
def cache_stats_api():

    total = cache_stats["hit"] + cache_stats["miss"]

    hit_rate = cache_stats["hit"] / total if total > 0 else 0

    return {
        "total_entries": sum(len(v) for v in cluster_cache.values()),
        "hit_count": cache_stats["hit"],
        "miss_count": cache_stats["miss"],
        "hit_rate": hit_rate
    }


@app.delete("/cache")
def clear_cache():

    for key in cluster_cache:
        cluster_cache[key] = []

    cache_stats["hit"] = 0
    cache_stats["miss"] = 0

    return {"message": "cache cleared"}