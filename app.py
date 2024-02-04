import streamlit as st
import os
from PyPDF2 import PdfReader

# Base directory where your preloaded PDFs are stored within your project structure
BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardenVA")

def get_categories():
    """Get a list of categories based on folder names."""
    return sorted(next(os.walk(BASE_DIR))[1])

def get_documents(category):
    """Get a list of document names for a given category."""
    category_path = os.path.join(BASE_DIR, category)
    return sorted([doc for doc in next(os.walk(category_path))[2] if doc.endswith('.pdf')])

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file given its path."""
    document_text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                document_text += text + "\n"
    return document_text

def main():
    st.title("Polisvoorwaardentool")

    # Allow user to select a category
    categories = get_categories()
    selected_category = st.selectbox("Select a category:", categories)

    # Display documents based on selected category
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Select a document:", documents)

    if st.button("Extract Text"):
        document_path = os.path.join(BASE_DIR, selected_category, selected_document)
        document_text = extract_text_from_pdf(document_path)
        st.text_area("Document Text", document_text, height=300)

if __name__ == "__main__":
    main()
