import numpy as np
from numpy.linalg import norm
from langchain.embeddings import OpenAIEmbeddings
from transformers import AutoTokenizer, AutoModel
import torch

def convert_empty_string_to_null(text):
    return None if text == "" else text

def calculate_embedded_text_similarity_percentage(text1, text2, embeddings):
    text1 = convert_empty_string_to_null(text1)
    text2 = convert_empty_string_to_null(text2)
    if text1 is None or text2 is None:
        return 100.0 if text1 == text2 else 0.0
    else:
        a = embeddings.embed_query(text1)
        b = embeddings.embed_query(text2)
        cosine_similarity = np.dot(a, b) / (norm(a) * norm(b))
        # Normalize and convert to percentage
        similarity_percentage = (cosine_similarity + 1) / 2
        # Apply a non-linear transformation to adjust the similarity scores
        adjusted_similarity_percentage = adjust_similarity_score(similarity_percentage)
        return adjusted_similarity_percentage
    
def adjust_similarity_score(score):
    # Applying a power transformation to spread the scores
    adjusted_score = pow(score, 2)  
    return adjusted_score * 100  
    
class HuggingFaceEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_query(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the mean of the last hidden state as the embedding vector
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings

# Initialize embeddings with the specified deployment
#embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002")
embeddings = HuggingFaceEmbeddings()

# Example using
text1 = "I am going to play Football."
text2 = "They are going to ruin Baseball."

similarity_percentage = calculate_embedded_text_similarity_percentage(text1, text2, embeddings)
print("Similarity between the texts:", similarity_percentage, "%")