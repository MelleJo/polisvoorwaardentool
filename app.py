import streamlit as st
import os
from PyPDF2 import PdfReader
from transformers import pipeline

# Set the base directory for preloaded PDFs
BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardentoolVA")

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Function to get categories from the base directory
def get_categories():
    return sorted(os.listdir(BASE_DIR))

# Function to get document names within a selected category
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

# Main Streamlit app function
def main():
    st.title("Polisvoorwaardentool - Testversie 1.2. - chunks")
    
    # Debug mode toggle
    debug_mode = st.checkbox('Debugmodus', value=False)

    # Model version selection
    model_choice = st.selectbox("Kies model versie:", ["Default summarization model"])
    
    # Document selection UI
    categories = get_categories()
    selected_category = st.selectbox("Selecteer een categorie:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Selecteer een document:", documents)
        
    # Question input
    question = st.text_input("Stel een vraag over het document:")
    
    if st.button("Krijg Antwoord") and question:
        document_path = os.path.join(BASE_DIR, selected_category, selected_document)
        document_text = extract_text_from_pdf(document_path)
        
        try:
            # Summarize the document
            summary = summarizer(document_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            
            # Display the summary
            st.write("Samenvatting van het document:")
            st.write(summary)
            
            if debug_mode:
                st.subheader("Debug Informatie")
                st.write(f"Vraag: {question}")
                if st.checkbox('Toon documenttekst'):
                    st.write(document_text)
        except Exception as e:
            st.error(f"Er is een fout opgetreden: {e}")

if __name__ == "__main__":
    main()
