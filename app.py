import streamlit as st
import os
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np

# Set the base directory for preloaded PDFs within the same GitHub repo as app.py
BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardenVA")

# Initialize tokenizer and model for embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Function to encode text to embeddings
def encode_text_to_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.pooler_output.detach().numpy()

# Function to get categories and documents
def get_categories():
    return sorted(os.listdir(BASE_DIR))

def get_documents(category):
    category_path = os.path.join(BASE_DIR, category)
    return sorted([doc for doc in os.listdir(category_path) if doc.endswith('.pdf')])

# Function to extract text from a PDF document
def extract_text_from_pdf(file_path):
    document_text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                document_text += text + "\n"
    return document_text

# Main Streamlit application
def main():
    st.title("Polisvoorwaardentool - Testversie 1.0")
    
    debug_mode = st.checkbox('Debugmodus', value=False)
    
    # Document selection UI
    categories = get_categories()
    selected_category = st.selectbox("Selecteer een categorie:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Selecteer een document:", documents)
    
    question = st.text_input("Stel een vraag over het document:")
    
    if st.button("Krijg Antwoord") and question:
        document_path = os.path.join(BASE_DIR, selected_category, selected_document)
        document_text = extract_text_from_pdf(document_path)
        
        # Process the document text into embeddings and create a FAISS index
        document_embeddings = encode_text_to_embeddings(document_text)
        dim = document_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(document_embeddings)
        
        # Encode the question for similarity search
        question_embedding = encode_text_to_embeddings(question)
        _, I = index.search(question_embedding, 1)  # Searching for the top 1 similar chunk
        
        if debug_mode:
            st.subheader("Debug Informatie")
            st.write(f"Vraag: {question}")
            if st.checkbox('Toon documenttekst'):
                st.write(document_text)
        
        # Display a placeholder response or further process I for retrieving the answer
        st.write("Response placeholder. Further implementation required to display the specific answer.")

if __name__ == "__main__":
    main()
