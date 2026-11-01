from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


document1 = "Natural Language Processing is a branch of Artificial Intelligence"

document2 = "Natural Language Processing is an important field of Artificial Intelligence"

documents = [document1, document2]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

similarity = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:2]
)
print("Document 1:", documents[0])
print("Document 2:", documents[1])
cosine = similarity[0][0]
print("Cosine Similarity:", round(cosine, 4))
print("Cosine Similarity:", round(cosine * 100, 2), "%")