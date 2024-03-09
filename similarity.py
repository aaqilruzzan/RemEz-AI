import numpy as np
from numpy.linalg import norm
import faiss  # Import Faiss for vector database
from transformers import AutoTokenizer, AutoModel
import torch

class HuggingFaceEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.vector_db = faiss.IndexFlatL2(768)  # Initialize vector database

    def embed_query(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the mean of the last hidden state as the embedding vector
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        # Add the embeddings to the vector database
        self.vector_db.add(embeddings)
        return embeddings

def calculate_embedded_text_similarity_percentage(embeddings, query_embedding):
    # Search the vector database for the most similar embeddings to the query embedding
    _, indices = embeddings.vector_db.search(query_embedding.reshape(1, -1), k=10)
    return indices

# Initialize embeddings with the specified deployment
embeddings = HuggingFaceEmbeddings()

# Example using
text1 = "I am going to play Football."
text2 = "They are going to ruin Baseball."

# Embed the texts and get query embedding
query_embedding = embeddings.embed_query(text1)

# Calculate similarity using the query embedding
similar_indices = calculate_embedded_text_similarity_percentage(embeddings, query_embedding)

print("Most similar indices:", similar_indices)
