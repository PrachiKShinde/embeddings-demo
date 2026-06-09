from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "SwiftUI is Apple's framework for building user interfaces on iOS.",
    "Kotlin is commonly used for Android development.",
    "Dogs are friendly animals.",
    "Vector databases store embeddings for similarity search."
]

query = "I dont like pet animals at all"

doc_embeddings = model.encode(documents)
query_embedding = model.encode(query)

scores = model.similarity(query_embedding, doc_embeddings)

print(scores)

best_match_index = scores.argmax()
print("Best matching document:")
print(documents[best_match_index])
