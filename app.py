import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI

# Initialize the ChatOpenAI model with GPT-4 Turbo
openai_api_key = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4-turbo-preview", temperature=0)

# Setup your base directory for preloaded PDFs
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

ef main():
    st.title("Polisvoorwaardentool")

    categories = get_categories()
    selected_category = st.selectbox("Select a category:", categories)

    documents = get_documents(selected_category)
    selected_document = st.selectbox("Select a document:", documents)

    document_path = os.path.join(BASE_DIR, selected_category, selected_document)
    document_text = extract_text_from_pdf(document_path)

    question = st.text_input("Ask a question about the document:")
    if st.button("Get Answer") and document_text and question:
        # Assuming invoke now expects an 'input' argument
        try:
            # Adjusting to the expected 'input' argument format for invoke
            response = qa_document_chain.invoke(input={
                "document_text": document_text,
                "question": question
            })
            # Assuming 'response' contains the answer directly or needs to be accessed appropriately
            st.write(response)
        except Exception as e:
            st.write(f"An error occurred: {e}")

    # Download buttons logic remains the same as before

if __name__ == "__main__":
    main()
