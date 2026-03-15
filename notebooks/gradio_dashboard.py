import pandas as pd
import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import gradio as gr

import sys
sys.path.append("..")
# Load Dataset

books = pd.read_csv("../data/books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "../assets/cover-not-found.jpg",
    books["large_thumbnail"],
)

# Embedding Model

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


# Load Vector Database

db_books = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)

# Recommendation Retrieval

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    # Vector search
    recs = db_books.similarity_search(query, k=initial_top_k)

    books_list = [rec.metadata["isbn13"] for rec in recs]

    book_recs = books[books["isbn13"].isin(books_list)].copy()


    # Emotion ranking
    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)

    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)

    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)

    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)

    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)


    # Category filtering
    if category != "All":
        book_recs = book_recs[
            book_recs["simple_categories"] == category
        ]


    # Return final recommendations
    return book_recs.head(final_top_k)

# Format Results for UI

def recommend_books(query: str, category: str, tone: str):

    recommendations = retrieve_semantic_recommendations(query, category, tone)

    results = []

    for _, row in recommendations.iterrows():

        description = row["description"]

        truncated_desc_split = description.split()

        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")

        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"

        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"

        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"

        results.append((row["large_thumbnail"], caption))

    return results



# UI Options

categories = ["All"] + sorted(books["simple_categories"].dropna().unique())

tones = [
    "All",
    "Happy",
    "Surprising",
    "Angry",
    "Suspenseful",
    "Sad"
]

# Gradio Dashboard

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:

    gr.Markdown("# Book Recommendation System")

    with gr.Row():

        user_query = gr.Textbox(
            label="Describe the book you want",
            placeholder="Example: A magical adventure story about friendship"
        )

        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Select Category",
            value="All"
        )

        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Select Emotional Tone",
            value="All"
        )

        submit_button = gr.Button("Find Recommendations")


    gr.Markdown("## Recommended Books")

    output = gr.Gallery(
        label="Recommended Books",
        columns=8,
        rows=2
    )


    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

# Launch Dashboard

if __name__ == "__main__":
    dashboard.launch()