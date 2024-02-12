import streamlit as st
import os
import numpy as np
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Function to encode text to embeddings
def encode_text_to_embeddings(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Set the base directory for preloaded PDFs
BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardentoolVA")

# Functions to get categories, documents, and extract text from a PDF document
# Your existing functions remain unchanged

# Main Streamlit app function
def main():
    st.title("Polisvoorwaardentool - Testversie 1.2 - chunks")
    
    # Your existing Streamlit UI code remains unchanged
    
    if st.button("Krijg Antwoord") and question:
        document_path = os.path.join(BASE_DIR, selected_category, selected_document)
        document_text = extract_text_from_pdf(document_path)
        
        # Encode document text to embeddings
        document_embeddings = encode_text_to_embeddings([document_text])[0]
        
        # Create a FAISS index
        embedding_dim = document_embeddings.shape[0]
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(document_embeddings.reshape(1, -1))
        
        # Encode the question to embeddings
        question_embedding = encode_text_to_embeddings([question])[0]
        
        # Perform the search to find the most relevant chunk
        _, I = index.search(question_embedding.reshape(1, -1), 1)
        
        # Assume the whole document is relevant if FAISS can't find a closer chunk
        relevant_chunk_index = I[0][0] if I.size else 0
        relevant_text = document_text  # Simplification for demonstration
        
        # Initialize ChatOpenAI with the selected model
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model=model_version)
        
        # Prepare the messages for the chat with relevant_text
        messages = [
            SystemMessage(content="Jij bent een expert in het analyseren van polisvoorwaarden. De gebruiker is een schadebehandelaar en wil graag jouw hulp bij het vinden van specifieke en relevante informatie voor de schadebehandeling van een polis. Nauwkeurigheid is prioriteit nummer 1"),
            SystemMessage(content=relevant_text),
            HumanMessage(content=question)
        ]
        
        # Get the response
        try:
            response = llm.invoke(messages)
            if response:
                st.write(response.content)
                if debug_mode:
                    st.subheader("Debug Informatie")
                    st.write(f"Vraag: {question}")
                    if st.checkbox('Toon documenttekst'):
                        st.write(relevant_text)
            else:
                st.error("Geen antwoord gegenereerd.")
        except Exception as e:
            st.error(f"Er is een fout opgetreden: {e}")

if __name__ == "__main__":
    main()
