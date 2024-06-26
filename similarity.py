import numpy as np
from numpy.linalg import norm
from langchain.embeddings import OpenAIEmbeddings
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def convert_empty_string_to_null(text):
    return None if text == "" else text

def calculate_embedded_text_similarity_percentage(text1, text2, embeddings):
    text1 = convert_empty_string_to_null(text1)
    text2 = convert_empty_string_to_null(text2)
    if text1 is None or text2 is None:
        return 100.0 if text1 == text2 else 0.0
    else:
        a = embeddings_store.get_embedding(text1)
        if a is None:
            a = embeddings_store.compute_and_store_embedding(text1)
        b = embeddings_store.get_embedding(text2)
        if b is None:
            b = embeddings_store.compute_and_store_embedding(text2)
        # Calculating cosine similarity between the two embeddings
        cosine_similarity = np.dot(a, b) / (norm(a) * norm(b))
        # Normalizing the similarity score to a 0-1 range and then to percentage
        similarity_percentage = (cosine_similarity + 1) / 2
        # Adjusting the similarity score for more intuitive understanding
        adjusted_similarity_percentage = adjust_similarity_score(similarity_percentage)
        return adjusted_similarity_percentage
    
def adjust_similarity_score(score):
    # Applying a power transformation to spread the scores
    adjusted_score = pow(score, 2)  
    return adjusted_score * 100  
    
# class HuggingFaceEmbeddings:
#     def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)

#     def embed_query(self, text):
#         inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         # Use the mean of the last hidden state as the embedding vector
#         embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
#         return embeddings
    
class EmbeddingsStore:
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        self.embedding_dict = {}

    # Method to get the embedding for a given text
    def get_embedding(self, text):
        return self.embedding_dict.get(text, None)

    # Method to compute and store the embedding for a given text
    def compute_and_store_embedding(self, text):
        embedding = self.embeddings_model.embed_query(text)
        self.embedding_dict[text] = embedding
        return embedding


embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002")
# embeddings = HuggingFaceEmbeddings()
embeddings_store = EmbeddingsStore(embeddings)

system_answer = "Sri Lanka is making strides in conservation efforts by protecting its unique biodiversity and promoting sustainable tourism practices through government and NGO initiatives."
user_answer = "Governments and Non government organization efforts"

def calculate_similarity(text1,text2):
    vectorizer = TfidfVectorizer()
    combined_texts = [text1, text2]
    tfidf_matrix = vectorizer.fit_transform(combined_texts)
    cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    similarity_percentage = cos_sim * 100
    rounded_similarity_percentage = round(similarity_percentage, 2)
    print(f"Similarity: {rounded_similarity_percentage:.2f}%")
    return rounded_similarity_percentage

calculate_similarity(system_answer,user_answer)
