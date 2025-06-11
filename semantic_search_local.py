# semantic_search_pipeline.py

import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss

# --- Utility functions ---
def clean_text(text):
    return " ".join(text.strip().split()) if text else ""

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

# --- Load posts ---
with open("data/discourse_posts.json", "r", encoding="utf-8") as f:
    posts_data = json.load(f)

print(f"✅ Loaded {len(posts_data)} posts")

# 🔧 Fix missing 'post_number'
grouped = defaultdict(list)
for post in posts_data:
    grouped[post["topic_id"]].append(post)

for topic_id, posts in grouped.items():
    for i, post in enumerate(posts, start=1):
        post.setdefault("post_number", i)

# --- Group by topic_id ---
topics = defaultdict(lambda: {"topic_title": "", "posts": []})
for post in posts_data:
    tid = post["topic_id"]
    topics[tid]["posts"].append(post)
    if "topic_title" in post:
        topics[tid]["topic_title"] = post["topic_title"]

for topic in topics.values():
    topic["posts"].sort(key=lambda x: x.get("post_number", 0))

print(f"✅ Grouped into {len(topics)} topics")

# --- Embedding Model ---
model_name = "all-MiniLM-L6-v2"  # Or "GritLM/GritLM-8x7B"
embedder = SentenceTransformer(model_name)

# --- Build reply tree ---
def build_reply_map(posts):
    reply_map = defaultdict(list)
    posts_by_number = {}
    for post in posts:
        num = post.get("post_number")
        if num is None:
            continue
        posts_by_number[num] = post
        parent = post.get("reply_to_post_number")
        reply_map[parent].append(post)
    return reply_map, posts_by_number

def extract_subthread(root_num, reply_map, posts_by_number):
    collected = []
    def dfs(pn):
        if pn not in posts_by_number:
            return
        p = posts_by_number[pn]
        collected.append(p)
        for child in reply_map.get(pn, []):
            dfs(child["post_number"])
    dfs(root_num)
    return collected

# --- Embed subthreads ---
embedding_data = []
embeddings = []

print("🔄 Building subthread embeddings...")

for tid, data in tqdm(topics.items()):
    posts = data["posts"]
    title = data["topic_title"]
    reply_map, by_number = build_reply_map(posts)

    root_posts = [p for p in posts if not p.get("reply_to_post_number")]

    if not root_posts:
        print(f"⚠️ No root posts found for topic ID {tid}. Skipping.")
        continue

    for root in root_posts:
        if "post_number" not in root:
            print(f"⚠️ Skipping root post due to missing 'post_number': {root}")
            continue
        root_num = root["post_number"]

        subthread = extract_subthread(root_num, reply_map, by_number)
        combined = f"Topic: {title}\n\n" + "\n\n---\n\n".join(
            clean_text(p["content"]) for p in subthread if "content" in p
        )

        emb = embedder.encode(combined, convert_to_numpy=True)
        emb = normalize(emb)

        embedding_data.append({
            "topic_id": tid,
            "topic_title": title,
            "root_post_number": root_num,
            "post_numbers": [p["post_number"] for p in subthread if "post_number" in p],
            "combined_text": combined
        })
        embeddings.append(emb)

if not embeddings:
    print("❌ No embeddings were generated. Exiting.")
    exit()

embeddings = np.vstack(embeddings).astype("float32")

# --- Build FAISS index ---
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

print(f"✅ Indexed {len(embedding_data)} subthreads")

# --- Semantic retrieval ---
def retrieve(query, top_k=5):
    q_emb = embedder.encode(query, convert_to_numpy=True)
    q_emb = normalize(q_emb).astype("float32")
    D, I = index.search(np.array([q_emb]), top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        data = embedding_data[idx]
        results.append({
            "score": float(score),
            "topic_id": data["topic_id"],
            "topic_title": data["topic_title"],
            "root_post_number": data["root_post_number"],
            "post_numbers": data["post_numbers"],
            "combined_text": data["combined_text"],
        })
    return results

# --- QA generation using T5 ---
gen_model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
qa_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

def generate_answer(query, contexts, max_len=256):
    context = "\n\n".join(contexts)
    prompt = f"Answer the question based on the following forum discussion:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True)
    outputs = qa_model.generate(**inputs, max_length=max_len, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Run Example ---
if __name__ == "__main__":
    query = "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?"

    results = retrieve(query, top_k=3)

    print("\n🔍 Top Retrieved Threads:")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']:.4f}")
        print(f"Topic Title: {r['topic_title']}")
        print(f"Root Post #: {r['root_post_number']} | Post IDs: {r['post_numbers']}")
        print(f"Snippet:\n{r['combined_text'][:300]}...\n")

    contexts = [r["combined_text"] for r in results]
    answer = generate_answer(query, contexts)

    print("\n💡 Generated Answer:\n", answer)
