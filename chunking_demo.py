from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

chunks = [
    "SwiftUI is Apple's modern UI framework.",
    "State management can be done using @State.",
    "NavigationStack is used for navigation.",
    "SwiftUI integrates with UIKit."
]

query = "How do I navigate between screens?"

chunk_embeddings = model.encode(chunks)
query_embedding = model.encode(query)

scores = model.similarity(
    query_embedding,
    chunk_embeddings
)

for chunk, score in zip(chunks, scores[0]):
    print(f"{score:.4f} -> {chunk}")
