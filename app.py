import streamlit as st  
from dotenv import load_dotenv  
from PyPDF2 import PdfReader    
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import faiss
import torch

def get_pdf_text(pdf_docs):
    text = ""   
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf) 
        for page in pdf_reader.pages:
            if page.extract_text() is not None:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(  
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_embeddings(text_chunks):
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def get_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    embeddings = [get_embedding(chunk) for chunk in text_chunks]
    return embeddings

def create_faiss_index(embeddings):
    dimension = embeddings[0].shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    for embedding in embeddings:
        faiss_index.add(embedding)
    return faiss_index


def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Q&A")
    
    st.header("PDF Question Generator")
    st.text_input("Ask a question about your documents:") 

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):  
    
                raw_text = get_pdf_text(pdf_docs)
                
                text_chunks = get_text_chunks(raw_text)

                embeddings = get_embeddings(text_chunks)
                faiss_index = create_faiss_index(embeddings)

                st.success("Documents processed and embeddings stored!")

if __name__ == '__main__':
    main()