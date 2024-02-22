import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardentoolVA")

def get_categories():
    return sorted(next(os.walk(BASE_DIR))[1])

def get_documents(category):
    category_path = os.path.join(BASE_DIR, category)
    return sorted([doc for doc in os.listdir(category_path) if doc.endswith('.pdf')])

def extract_text_from_pdf(file_path):
    document_text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                document_text += text + "\n"
    return document_text

def split_and_embed_text(document_text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(document_text)
    
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    vector_store = FAISS(dimension=embeddings.embedding_dimension)
    vector_store.add_texts(chunks, embeddings)
    return vector_store

def main():
    st.title("Polisvoorwaardentool - verbeterde versie met FAISS")

    categories = get_categories()
    selected_category = st.selectbox("Kies een categorie:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Selecteer een polisvoorwaardendocument:", documents)
    document_path = os.path.join(BASE_DIR, selected_category, selected_document)

    with open(document_path, "rb") as file:
        st.download_button(
            label="Download PDF",
            data=file,
            file_name=selected_document,
            mime="application/pdf"
        )

    question = st.text_input("Vraag maar raak:")
    
    if st.button("Antwoord") and question:
        document_text = extract_text_from_pdf(document_path)
        vector_store = split_and_embed_text(document_text)
        
        # Find most relevant chunks
        docs = vector_store.similarity_search(question, top_k=5)  # Adjust top_k as needed
        
        llm = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        chain = load_qa_chain(llm, chain_type="map_reduce")
        
        # Assuming docs are the indices of the chunks
        relevant_chunks = [document_text[doc["id"]] for doc in docs]  # Adjust indexing as needed
        response = chain.run(input_documents=relevant_chunks, question=question)
        
        st.write(response)

if __name__ == "__main__":
    main()
