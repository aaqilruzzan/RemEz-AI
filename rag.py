from dotenv import load_dotenv  
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
import faiss
# import torch
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(  
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# def get_embeddings(text_chunks):
#     model_name = "distilbert-base-uncased"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)

#     def get_embedding(text):
#         inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
#     embeddings = [get_embedding(chunk) for chunk in text_chunks]
#     return embeddings

def create_faiss_index(embeddings):
    dimension = embeddings[0].shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    for embedding in embeddings:
        faiss_index.add(embedding)
    return faiss_index

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def generate_questions_answers(raw_text,questionNo):
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation = get_conversation_chain(vectorstore)

    generated_questions = {}
    generated_answers = {}
    
    user_question = f"Generate {questionNo} questions of 2 lines each and generate the corresponding answers to those question dont make them short answers. End each question with a question mark. Add two line breaks after each question-answer pair. Don't add two line breaks anywhere else. Don't put numbers for questions"
    response = conversation({'question': user_question})
    print(response)
    
    ai_text = response['answer']
    
    lines = ai_text.split('\n\n')  # Splits the text into lines
    
    generated_questions = {}
    generated_answers = {}
    question_number = 1  # To keep track of question numbers
    
    for line in lines:
        
        parts = line.split("?", 1)
        question = parts[0].strip()
        answer = parts[1].strip() if len(parts) > 1 else ""
        
        generated_questions[question_number] = question
        generated_answers[question_number] = answer
        question_number += 1
    
    return generated_questions, generated_answers

def main(raw_text,questionNo):
    load_dotenv()
    
    generated_questions, generated_answers = generate_questions_answers(raw_text,questionNo)
    
    return generated_questions, generated_answers

if __name__== '__main__':
    # Pass the raw text input directly to the main function
    raw_text = "Replace this with your raw text input"
    generated_questions, generated_answers = main(raw_text)
    
    # Display the generated questions and answers
    for i in range(1, 4):
        print(f"Question {i}:")
        print(generated_questions[i])
        print(f"Answer {i}:")
        print(generated_answers[i])