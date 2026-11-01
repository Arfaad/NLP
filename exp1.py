from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


documents = [
    "I love machine learning",
    "I love deep learning",
    "Machine learning is interesting"
]


# --------------------------------------------------
# 1. BAG OF WORDS
# --------------------------------------------------

print("=" * 50)
print("BAG OF WORDS")
print("=" * 50)

bow_vectorizer = CountVectorizer()

bow_matrix = bow_vectorizer.fit_transform(documents)

print("Vocabulary:")
print(bow_vectorizer.get_feature_names_out())

print("\nBoW Matrix:")
print(bow_matrix.toarray())


# --------------------------------------------------
# 2. TF-IDF
# --------------------------------------------------

print("\n" + "=" * 50)
print("TF-IDF")
print("=" * 50)

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("Vocabulary:")
print(tfidf_vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix:")

for row in tfidf_matrix.toarray():
    print(row)


# --------------------------------------------------
# 3. BIGRAM
# --------------------------------------------------

print("\n" + "=" * 50)
print("BIGRAM")
print("=" * 50)

bigram_vectorizer = CountVectorizer(
    ngram_range=(2, 2)
)

bigram_matrix = bigram_vectorizer.fit_transform(documents)

print("Bigrams:")
print(bigram_vectorizer.get_feature_names_out())

print("\nBigram Matrix:")
print(bigram_matrix.toarray())


# --------------------------------------------------
# 4. TRIGRAM
# --------------------------------------------------

print("\n" + "=" * 50)
print("TRIGRAM")
print("=" * 50)

trigram_vectorizer = CountVectorizer(
    ngram_range=(3, 3)
)

trigram_matrix = trigram_vectorizer.fit_transform(documents)

print("Trigrams:")
print(trigram_vectorizer.get_feature_names_out())

print("\nTrigram Matrix:")
print(trigram_matrix.toarray())