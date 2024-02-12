import streamlit as st
import os
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import pickle

# Initialize tokenizer and model for embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Set the base directory for preloaded PDFs and cache directory
BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardenVA")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_categories():
    return sorted(os.listdir(BASE_DIR))

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

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

def check_cache(document_name):
    embeddings_path = os.path.join(CACHE_DIR, f"{document_name}_embeddings.npy")
    index_path = os.path.join(CACHE_DIR, f"{document_name}_index.faiss")
    return os.path.exists(embeddings_path) and os.path.exists(index_path)

def load_cached_data(document_name):
    embeddings_path = os.path.join(CACHE_DIR, f"{document_name}_embeddings.npy")
    index_path = os.path.join(CACHE_DIR, f"{document_name}_index.faiss")
    embeddings = np.load(embeddings_path)
    index = faiss.read_index(index_path)
    return embeddings, index

def save_to_cache(document_name, embeddings, index):
    embeddings_path = os.path.join(CACHE_DIR, f"{document_name}_embeddings.npy")
    index_path = os.path.join(CACHE_DIR, f"{document_name}_index.faiss")
    np.save(embeddings_path, embeddings)
    faiss.write_index(index, index_path)

def process_pdf(document_path):
    document_name = os.path.splitext(os.path.basename(document_path))[0]
    if check_cache(document_name):
        return load_cached_data(document_name)
    else:
        document_text = extract_text_from_pdf(document_path)
        embeddings = embed_text(document_text)  # This should produce a (n, d) array
        # Ensure embeddings are in the correct shape (n, d)
        if len(embeddings.shape) == 1:
            # If embeddings is a 1D array, reshape it to (1, d)
            embeddings = embeddings.reshape(1, -1)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)  # embeddings is now guaranteed to be (n, d)
        save_to_cache(document_name, embeddings, index)
        return embeddings, index

def main():
    st.title("Polisvoorwaardentool - Testversie 1.1. - FAISS")
    
    debug_mode = st.checkbox('Debugmodus', value=False)
    model_choice = st.selectbox("Kies model versie:", ["ChatGPT 3.5 Turbo", "gpt-4-turbo-preview"])
    model_version = "gpt-3.5-turbo" if model_choice == "ChatGPT 3.5 Turbo" else "gpt-4-turbo-preview"
    
    categories = get_categories()
    selected_category = st.selectbox("Selecteer een categorie:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Selecteer een document:", documents)
    
    question = st.text_input("Stel een vraag over het document:")
    
    if st.button("Krijg Antwoord") and question:
        document_path = os.path.join(BASE_DIR, selected_category, selected_document)
        embeddings, index = process_pdf(document_path)
        # Note: Implement logic to use embeddings and index for answering the question
        # This part of the code is left as an exercise for integration with langchain or your custom logic

if __name__ == "__main__":
    main()
