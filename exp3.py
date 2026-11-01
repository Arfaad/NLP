from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Given corpus
documents = [
    "Machine learning is a branch of artificial intelligence",
    "Natural language processing deals with human language",
    "Deep learning uses neural networks for learning",
    "Natural language processing is used in chatbots",
    "Machine learning algorithms are used for prediction"
]

# Get query from user
query = input("Enter your search query: ")

# TF-IDF representation
vectorizer = TfidfVectorizer()

document_vectors = vectorizer.fit_transform(documents)

query_vector = vectorizer.transform([query])

# Calculate similarity
similarity_scores = cosine_similarity(
    query_vector,
    document_vectors
)

scores = similarity_scores[0]
print("scores",scores)

# Sort documents according to similarity
ranked_documents = scores.argsort()[::-1]
print("ranked_documents",ranked_documents)

# Display results
print("\n========== SEARCH RESULTS ==========")

for index in ranked_documents:

    if scores[index] > 0:
        print("\nDocument", index + 1)
        print(documents[index])
        print("Similarity Score:", round(scores[index], 4))