from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

with open("doc_intro.txt", "r") as file:
    document = file.read()

documents = document.split("\n\n")

query = "Where can I view DocC documentation?"

doc_embeddings = model.encode(documents)

query_embedding = model.encode(query)

scores = model.similarity(query_embedding, doc_embeddings)

best_match_index = scores.argmax()

print("Best matching paragraph:\n")
print(documents[best_match_index])
