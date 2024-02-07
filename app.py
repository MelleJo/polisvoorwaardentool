import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI  # Ensure this import is correct
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain

# Initialize the ChatOpenAI model with the API key and specify gpt-4-turbo-preview
openai_api_key = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4-turbo-preview")

# Load the question answering chain with the ChatOpenAI model
qa_chain = load_qa_chain(llm, chain_type="map_reduce")

# Initialize AnalyzeDocumentChain with the QA chain
qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

# Setup your base directory for preloaded PDFs
BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardenVA")

def get_categories():
    # Function implementation remains the same

def get_documents(category):
    # Function implementation remains the same

def extract_text_from_pdf(file_path):
    # Function implementation remains the same

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
        # Use 'invoke' method correctly with the defined 'qa_document_chain'
        try:
            response = qa_document_chain.invoke(input={"document_text": document_text, "question": question})
            st.write(response)  # Adjust according to the actual response structure
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Download buttons logic

if __name__ == "__main__":
    main()
