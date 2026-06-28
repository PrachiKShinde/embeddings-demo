from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

with open("doc_intro.txt", "r") as file:
    chunk = file.read()

chunks = chunk.split("\n\n")

query = "Can DocC generate websites?"

chunk_embeddings = model.encode(chunks)
query_embedding = model.encode(query)

scores = model.similarity(
    query_embedding,
    chunk_embeddings
)

for chunk, score in zip(chunks, scores[0]):
    print(f"{score:.4f} -> {chunk}")
