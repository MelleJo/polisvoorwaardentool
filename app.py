import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain

# Directly set BASE_DIR to the path where your PDF documents are stored
# Adjust this path as necessary to match your project's structure
BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardenVA")

def get_categories():
    """Get sorted list of categories (folders) within the base directory."""
    return sorted(next(os.walk(BASE_DIR))[1])

def get_documents(category):
    """Get sorted list of PDF documents within a specified category."""
    category_path = os.path.join(BASE_DIR, category)
    return sorted([doc for doc in os.listdir(category_path) if doc.endswith('.pdf')])

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
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

    categories = get_categories()
    selected_category = st.selectbox("Select a category:", categories)

    documents = get_documents(selected_category)
    selected_document = st.selectbox("Select a document:", documents)

    document_path = os.path.join(BASE_DIR, selected_category, selected_document)
    document_text = extract_text_from_pdf(document_path)

    question = st.text_input("Ask a question about the document:")
    if st.button("Get Answer") and document_text and question:
        # Initialize the ChatOpenAI model with the API key and specify the model
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-preview")

        # Load the question answering chain with the ChatOpenAI model
        qa_chain = load_qa_chain(llm, chain_type="map_reduce")

        # Initialize AnalyzeDocumentChain with the QA chain
        qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

        # Attempt to get an answer for the question based on the document
        try:
            response = qa_document_chain.invoke(input={"document_text": document_text, "question": question})
            st.write(response)  # Display the response
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Optional: Implement download buttons for PDF and text version

if __name__ == "__main__":
    main()
