### Book Recommendation System














This project implements an AI-powered book recommendation system that combines multiple modern NLP techniques to generate meaningful book recommendations.

Instead of relying only on keywords or metadata, the system uses a combination of:

- semantic search

- keyword search

- emotion analysis

- machine learning classification

- ranking models

To recommend books that better match what the user is looking for.

The system also includes a Gradio dashboard where users can describe the type of book they want and receive recommendations instantly.

### Live Demo

Try the live Book Recommendation System here:

👉 https://huggingface.co/spaces/absg2205/book-recommedation-system

or click below:

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/absg2205/book-recommedation-system)

#### Project Motivation

Traditional recommendation systems usually rely on collaborative filtering or metadata such as genre or author. While those methods work well in some cases, they often fail when a user wants something more nuanced, such as:

“a sad story about love and loss”

“a suspenseful mystery set in a small town”

This project explores how combining semantic understanding, emotional tone detection, and ranking signals can improve recommendation quality.

#### System Architecture

The system is built as a multi-stage recommendation pipeline, where different components contribute different signals before producing the final recommendations.


- Architecture Diagram

<img width="959" height="1416" alt="system_architecture" src="https://github.com/user-attachments/assets/2bacb8d7-873a-4e57-a57d-a82e971fbbec" />

- Example Dashboard
Query Input

Users can describe the type of book they want using natural language.

Example query:

A sad emotional story about love

<img width="1303" height="367" alt="dashboard-query" src="https://github.com/user-attachments/assets/741caa86-048e-4a18-a333-8cfcddea1f64" />


- Recommendation Output

The system returns books along with short explanations describing why they were recommended.

Example explanation:

The Fault in Our Stars by John Green

<img width="1288" height="509" alt="img_4" src="https://github.com/user-attachments/assets/9a053c58-9a7a-4e12-988c-b80ce86fd685" />



#### Key Features
- Hybrid Retrieval (Semantic + Keyword Search)

  The system combines two retrieval methods:

  - Semantic Search using sentence embeddings and a vector database

  - Keyword Search using TF-IDF

  This hybrid approach improves recall and helps the system retrieve relevant books even when wording differs.


- Neural Reranking

  A cross-encoder reranking model evaluates candidate books and produces a relevance score for each (query, book description) pair.

  This improves recommendation quality compared to simple similarity search.

- Multi-Signal Ranking

  Books are ranked using a combination of signals:

  Final Score = 0.6 * rerank_score+ 0.3 * rating_score+ 0.1 * popularity_score

  Where:

  rerank_score measures semantic relevance

  rating_score reflects book quality

  popularity_score reflects how widely the book is read

  Emotion-Aware Recommendations

  Book descriptions are analyzed using a transformer model that detects emotional tone such as:

  joy

  sadness

  anger

  fear

  surprise

  disgust

  This allows queries like:

  “a sad story about grief”

  to return emotionally relevant books.

- Category Classification

  A machine learning classifier predicts whether the query is related to:

  Fiction

  Nonfiction

  This helps filter results and improve recommendation relevance.

- Personalization

  The system allows user preferences to influence recommendations.

  Example:

  user_preferences = ["fantasy", "science fiction"]

  Books matching these preferences receive a ranking boost.


#### Tech Stack
- Programming Language

  - Python

- Machine Learning

  - LightGBM

  - Scikit-learn

- NLP Models

  - Sentence Transformers (embeddings)

  - DistilRoBERTa emotion classifier

  - Cross-encoder reranker

- Vector Search

  - Chroma vector database

- Libraries

  - Transformers

  - LangChain

  - Pandas

  - NumPy

  - Gradio

### Project Directory Structure

```text
book-recommendation-project/
├── data/
│   ├── books_cleaned.csv
│   ├── books_with_categories.csv
│   └── books_with_emotions.csv
├── models/
│   └── book_category_classifier.pkl
├── embeddings/
│   ├── train_embeddings.npy
│   └── test_embeddings.npy
├── vector_db/
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── text_classification.ipynb
│   ├── sentiment_analysis.ipynb
│   └── vector_search.ipynb
|   |__ gradio_dashboard.py
├── app/
│   └── app.py
├── assets/
│   └── cover-not-found.jpg
├── images/
│   ├── system_architecture.png
│   ├── dashboard-query.png
│   └── dashboard-results.png
├── requirements.txt
└── README.md
```

#### Running the Project
1. Clone the repository

   git clone https://github.com/absg2205/book-recommendation-system.git

   cd book-recommendation-project

2. Install dependencies

   pip install -r requirements.txt

3. Run the dashboard

   python app/app.py


The dashboard will start locally in your browser.

Example Query:

A suspenseful mystery set in a small town

The system will:

classify the query category

detect emotional tone

retrieve candidate books using hybrid search

rerank them using a neural model

rank them using multiple signals 

#### Final Notes:

This project was built mainly as a hands-on exploration of how modern NLP techniques can be combined into a practical recommendation system.

The goal was not only to build a working application, but also to understand how different components — embeddings, classifiers, reranking models, and ranking signals — can work together in a real system.
