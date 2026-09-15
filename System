import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)


# ============================================================
# TASK 1: Bag of Words Matrix Construction (Customer Reviews)
# ============================================================
def task1_bow_matrix():
    corpus = [
        "The product performance is amazing and fast",
        "The service was fast and performance was great",
        "Terrible customer service and bad performance"
    ]

    # Step 1: Instantiate CountVectorizer with English stop words removed
    vectorizer = CountVectorizer(stop_words='english')

    # Step 2: Fit and transform the corpus into a term-frequency matrix
    bow_matrix = vectorizer.fit_transform(corpus)

    # Step 3: Extract vocabulary
    vocabulary = vectorizer.get_feature_names_out()

    # Step 4: Convert sparse matrix -> Pandas DataFrame
    df_bow = pd.DataFrame(
        bow_matrix.toarray(),
        columns=vocabulary,
        index=[f"Doc{i+1}" for i in range(len(corpus))]
    )

    print("=" * 60)
    print("TASK 1: Bag of Words Matrix")
    print("=" * 60)
    print(f"\nVocabulary ({len(vocabulary)} terms): {list(vocabulary)}\n")
    print(df_bow)
    print()
    return df_bow


# ============================================================
# TASK 2: Document Search Engine & Relevance Ranking
# ============================================================
def task2_search_engine():
    documents = [
        "Machine learning algorithms analyze structured data effectively",
        "Deep learning and neural networks excel at processing unstructured data",
        "Natural language processing helps computers understand human language",
        "Python is widely used for machine learning and data science"
    ]
    query = ["machine learning algorithms for data"]

    # Step 1: Fit CountVectorizer on the documents
    vectorizer = CountVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)

    # Step 2: Transform the query using the SAME fitted vectorizer
    query_vector = vectorizer.transform(query)

    # Step 3: Compute cosine similarity between query and each document
    scores = cosine_similarity(query_vector, doc_vectors).flatten()

    # Step 4: Rank documents from highest to lowest score
    ranked = sorted(
        zip(range(1, len(documents) + 1), documents, scores),
        key=lambda x: x[2],
        reverse=True
    )

    print("=" * 60)
    print("TASK 2: Document Search Engine - Relevance Ranking")
    print("=" * 60)
    print(f"\nQuery: {query[0]}")
    print(f"Vocabulary ({len(vectorizer.get_feature_names_out())} terms): "
          f"{list(vectorizer.get_feature_names_out())}\n")
    print(f"{'Rank':<6}{'Score':<10}{'Document':<10}Text")
    print("-" * 60)
    for rank, (doc_id, text, score) in enumerate(ranked, start=1):
        print(f"{rank:<6}{score:.4f}    Doc{doc_id:<7}{text}")
    print()
    return ranked


if __name__ == "__main__":
    task1_bow_matrix()
    task2_search_engine()
