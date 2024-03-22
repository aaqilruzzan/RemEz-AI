import numpy as np
from numpy.linalg import norm
from langchain.embeddings import OpenAIEmbeddings
from transformers import AutoTokenizer, AutoModel
import torch
from dotenv import load_dotenv
import os
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from collections import defaultdict


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
        # Calculate cosine similarity between the two embeddings
        cosine_similarity = np.dot(a, b) / (norm(a) * norm(b))
        # Normalize the similarity score to a 0-1 range and then to percentage
        similarity_percentage = (cosine_similarity + 1) / 2
        # Adjust the similarity score for more intuitive understanding
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

# Initialize embeddings with the specified deployment
#embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002")
embeddings = HuggingFaceEmbeddings()
embeddings_store = EmbeddingsStore(embeddings)

def get_wordnet_pos(treebank_tag):
    """Converts treebank tags to WordNet tags."""
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN

def synonym_sets(word, pos):
    """Finds synonyms for a word with a given part-of-speech tag."""
    lemmatizer = WordNetLemmatizer()
    lemmatized_word = lemmatizer.lemmatize(word, pos=pos)
    synsets = wn.synsets(lemmatized_word, pos=pos)
    synonyms = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def calculate_similarity(text1, text2):
    """Compares two texts for similarity, considering synonyms."""
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    
    pos_tags1 = pos_tag(tokens1)
    pos_tags2 = pos_tag(tokens2)
    
    words1 = {word: get_wordnet_pos(pos) for word, pos in pos_tags1}
    words2 = {word: get_wordnet_pos(pos) for word, pos in pos_tags2}
    
    match_count = 0
    total_count = max(len(words1), len(words2))
    
    for word1, pos1 in words1.items():
        synonyms1 = synonym_sets(word1, pos1)
        for word2, pos2 in words2.items():
            if word1 == word2 or word2 in synonyms1:
                match_count += 1
                break
    
    similarity = (match_count / total_count) * 100
    return similarity

system_answer = "Sri Lanka has a documented history dating back 3,000 years, with evidence of prehistoric human settlements and a 26-year civil war that ended in 2009."
user_answer = "The Sinhala and Tamil New Year stands as a significant cultural event in Sri Lanka, signifying the conclusion of the harvesting period."

similarity_score = calculate_similarity(system_answer, user_answer)

print(f"Similarity: {similarity_score:.2f}%")