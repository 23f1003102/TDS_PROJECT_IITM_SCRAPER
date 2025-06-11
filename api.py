import os
import uvicorn
import requests
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langchain_nomic import NomicEmbeddings  # ✅ Using Nomic Embeddings with API Key

# ✅ Redirect Nomic's cache and working directory to writable locations 


# ✅ Load Environment Variables
load_dotenv()
api_key = os.getenv("AIPIPE_API_KEY")
nomic_api_key = os.getenv("NOMIC_API_KEY")  # ✅ Load Nomic API Key

if not api_key:
    raise RuntimeError("Missing AIPIPE API key in environment variables.")

if not nomic_api_key:
    raise RuntimeError("Missing Nomic API key in environment variables.")  # ✅ Ensure API key is available

# ✅ Initialize Nomic Embeddings
embedder = NomicEmbeddings(model="nomic-embed-text-v1.5")

# ✅ Initialize FastAPI
app = FastAPI()

# --- Load Discourse Data ---
try:
    with open("data/discourse_posts.json", "r", encoding="utf-8") as f:
        posts_data = json.load(f)
except FileNotFoundError:
    raise RuntimeError("Could not find 'data/discourse_posts.json'. Ensure the file is in the correct location.")

# Group posts by topic
topics = defaultdict(lambda: {"topic_title": "", "posts": []})
for post in posts_data:
    tid = post["topic_id"]
    topics[tid]["posts"].append(post)
    if "topic_title" in post:
        topics[tid]["topic_title"] = post["topic_title"]

# Sort posts within topics by post_number
for topic in topics.values():
    topic["posts"].sort(key=lambda x: x.get("post_number", 0))

# --- Embedding Setup ---
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

embedding_data = []
embeddings = []

# Process topics for FAISS
for tid, data in topics.items():
    posts = data["posts"]
    title = data["topic_title"]
    reply_map = defaultdict(list)
    by_number = {}

    for p in posts:
        pn = p.get("post_number")
        if pn is not None:
            by_number[pn] = p
        parent = p.get("reply_to_post_number")
        reply_map[parent].append(p)

    def extract(pn):
        collected = []
        def dfs(n):
            if n not in by_number:
                return
            p = by_number[n]
            collected.append(p)
            for child in reply_map.get(n, []):
                dfs(child.get("post_number"))
        dfs(pn)
        return collected

    roots = [p for p in posts if not p.get("reply_to_post_number")]
    for root in roots:
        root_num = root.get("post_number", 1)
        thread = extract(root_num)
        text = f"Topic: {title}\n\n" + "\n\n---\n\n".join(
            p.get("content", "").strip() for p in thread if p.get("content")
        )
        emb = normalize(embedder.embed_query(text))  # ✅ Updated Embedding Call
        embedding_data.append({
            "topic_id": tid,
            "topic_title": title,
            "root_post_number": root_num,
            "post_numbers": [p.get("post_number") for p in thread],
            "combined_text": text
        })
        embeddings.append(emb)

# Create FAISS index
index = faiss.IndexFlatIP(len(embeddings[0]))
index.add(np.vstack(embeddings).astype("float32"))

# --- API Input Model ---
class QuestionInput(BaseModel):
    question: str
    image: str = None  # Optional image input, unused here

# --- AIPIPE API Configuration ---
AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
AIPIPE_KEY = api_key

def query_aipipe(prompt):
    headers = {"Authorization": f"Bearer {AIPIPE_KEY}", "Content-Type": "application/json"}
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}

    response = requests.post(AIPIPE_URL, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=500, detail=f"AIPIPE API error: {response.text}")

# --- API Endpoint for Answer Generation ---
@app.post("/api/")
async def answer_question(payload: QuestionInput):
    q = payload.question

    # Ensure query is valid
    if not q:
        raise HTTPException(status_code=400, detail="Question field cannot be empty.")

    # Search FAISS Index
    q_emb = normalize(embedder.embed_query(q)).astype("float32")  # ✅ Updated Query Embedding Call
    D, I = index.search(np.array([q_emb]), 3)

    top_results = []
    for score, idx in zip(D[0], I[0]):
        data = embedding_data[idx]
        top_results.append({
            "score": float(score),
            "text": data["combined_text"],
            "topic_id": data["topic_id"],
            "url": f"https://discourse.onlinedegree.iitm.ac.in/t/{data['topic_id']}"
        })

    # Generate answer using AIPIPE
    try:
        answer_response = query_aipipe(q)
        answer = answer_response.get("choices", [{}])[0].get("message", {}).get("content", "No response.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching response from AIPIPE: {str(e)}")

    links = [{"url": r["url"], "text": r["text"][:120]} for r in top_results]
    return {"answer": answer, "links": links}

# --- Run the Server ---
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)