import streamlit as st
import os
import time
from PyPDF2 import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Initialize the sentence transformer model and FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
dim = model.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(dim)

BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardenVA")

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

def chunk_text(text, chunk_size=500):
    # Simple logic to chunk text
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def embed_and_index_chunks(chunks):
    # Embed chunks and add to FAISS index
    embeddings = model.encode(chunks)
    faiss_index.add(np.array(embeddings))

def find_relevant_chunks(question, top_k=5):
    question_embedding = model.encode([question])[0].reshape(1, -1)
    _, indices = faiss_index.search(question_embedding, top_k)
    return indices[0]

def main():
    st.title("Polisvoorwaardentool")

    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

    if st.button('Toggle Debug Mode'):
        st.session_state.debug_mode = not st.session_state.debug_mode

    categories = get_categories()
    selected_category = st.selectbox("Select a category:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Select a document:", documents)
    document_path = os.path.join(BASE_DIR, selected_category, selected_document)
    document_text = extract_text_from_pdf(document_path)

    # Process document text into chunks and index
    chunks = chunk_text(document_text)
    embed_and_index_chunks(chunks)

    question = st.text_input("Ask a question about the document:")
    if st.button("Get Answer") and document_text and question:
        start_time = time.time()

        # Find indices of relevant chunks
        relevant_chunk_indices = find_relevant_chunks(question)
        relevant_text = " ".join([chunks[i] for i in relevant_chunk_indices])

        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-preview")
        response = llm.generate(prompt=question + "\n\n" + relevant_text, max_tokens=512)

        processing_time = time.time() - start_time

        st.write(response)  # Display the answer

        if st.session_state.debug_mode:
            st.subheader("Debug Information")
            st.write(f"Question: {question}")
            st.write(f"Relevant Text: {relevant_text}")
            st.write(f"Processing Time: {processing_time:.2f} seconds")
            if st.checkbox('Show Document Text'):
                st.write(document_text)

if __name__ == "__main__":
    main()
