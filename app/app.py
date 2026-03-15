import pandas as pd
import numpy as np
import joblib

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import gradio as gr

# Load data

books = pd.read_csv("../data/books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "../assets/cover-not-found.jpg",
    books["large_thumbnail"],
)


# models loading

print("Loading embedding model")
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading reranker.")
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Loading emotion model")
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device="cpu",
)

print("Loading category classifier")
category_classifier = joblib.load(
    "../models/book_category_classifier.pkl"
)

label_map = {0: "Fiction", 1: "Nonfiction"}

# vector database

vector_embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db_books = Chroma(
    persist_directory="../vector_db",
    embedding_function=vector_embedding
)


# TF-IDF Keyword search (cached)

print("Building TF-IDF index")

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(
    books["tagged_description"]
)


def keyword_search(query, top_k=30):

    query_vec = tfidf.transform([query])

    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    indices = scores.argsort()[-top_k:][::-1]

    return books.iloc[indices]

# category prediction function

def predict_query_category(query):

    embedding = embedding_model.encode([query])

    pred = category_classifier.predict(embedding)[0]

    return label_map[pred]

# emotion detection function

def detect_query_emotion(query):

    prediction = emotion_model(query)[0]

    return max(prediction, key=lambda x: x["score"])["label"]

# hybrid retrieval function

def retrieve_books(query, category_filter, tone_filter, top_k=16):

    semantic_results = db_books.similarity_search(query, k=30)

    semantic_isbns = [doc.metadata["isbn13"] for doc in semantic_results]

    keyword_results = keyword_search(query)

    keyword_isbns = keyword_results["isbn13"].tolist()

    candidate_isbns = list(set(semantic_isbns + keyword_isbns))

    candidates = books[
        books["isbn13"].isin(candidate_isbns)
    ]

    if category_filter != "All":

        candidates = candidates[
            candidates["simple_categories"] == category_filter
        ]

    if len(candidates) == 0:
        return pd.DataFrame()

    pairs = [(query, text) for text in candidates["tagged_description"]]

    candidates = candidates.copy()

    candidates["rerank_score"] = reranker.predict(pairs)

    candidates["rating_score"] = candidates["average_rating"].fillna(0)

    candidates["popularity_score"] = np.log1p(
        candidates["ratings_count"].fillna(0)
    )

    candidates["final_score"] = (
        0.6 * candidates["rerank_score"]
        + 0.3 * candidates["rating_score"]
        + 0.1 * candidates["popularity_score"]
    )

    # Tone ranking
    if tone_filter != "All":

        emotion_map = {
            "Happy": "joy",
            "Sad": "sadness",
            "Angry": "anger",
            "Surprising": "surprise",
            "Suspenseful": "fear",
        }

        emotion_col = emotion_map.get(tone_filter)

        if emotion_col:
            candidates["final_score"] += candidates[emotion_col]

    candidates = candidates.sort_values(
        "final_score",
        ascending=False
    )

    return candidates.head(top_k)

# recommendation pipeline

def recommend_books(query, category, tone):

    if not query:
        return []

    results_df = retrieve_books(query, category, tone)

    results = []

    for _, row in results_df.iterrows():

        description = " ".join(
            row["description"].split()[:20]
        ) + "..."

        caption = f"""
**{row['title']}**

{row['authors']}

{description}

⭐ {row['average_rating']} |  {row['categories']}
"""

        results.append((row["large_thumbnail"], caption))

    return results

# dashboard

categories = ["All"] + sorted(
    books["simple_categories"].dropna().unique()
)

tones = [
    "All",
    "Happy",
    "Sad",
    "Angry",
    "Surprising",
    "Suspenseful"
]


with gr.Blocks(theme=gr.themes.Glass()) as dashboard:

    gr.Markdown("# Book Recommendation System")

    gr.Markdown(
        "Describe a book you want and get recommendations instantly."
    )

    with gr.Row():

        query = gr.Textbox(
            label="Describe the book you want",
            placeholder="Example: A sad emotional story about love"
        )

    with gr.Row():

        category_dropdown = gr.Dropdown(
            choices=categories,
            value="All",
            label="Category"
        )

        tone_dropdown = gr.Dropdown(
            choices=tones,
            value="All",
            label="Tone"
        )

    search_btn = gr.Button("Find Books")

    gr.Markdown("### Recommendations")

    output = gr.Gallery(
        columns=4,
        height="auto"
    )

    search_btn.click(
        fn=recommend_books,
        inputs=[query, category_dropdown, tone_dropdown],
        outputs=output
    )

# app launch

if __name__ == "__main__":
    dashboard.launch(
        allowed_paths=["../assets"]
    )