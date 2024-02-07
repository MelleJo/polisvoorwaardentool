import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_openai import OpenAI  # Import the OpenAI class

# Initialize the OpenAI model
openai_api_key = st.secrets["OPENAI_API_KEY"]
llm = OpenAI(openai_api_key=openai_api_key)  # Use your API key here

# Setup your base directory
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

def main():
    st.title("Polisvoorwaardentool")

    categories = get_categories()
    selected_category = st.selectbox("Select a category:", categories)

    documents = get_documents(selected_category)
    selected_document = st.selectbox("Select a document:", documents)

    document_path = os.path.join(BASE_DIR, selected_category, selected_document)
    document_text = extract_text_from_pdf(document_path)

    question = st.text_input("Ask a question about the document:")
    if st.button("Get Answer"):
        if document_text and question:
            # Formulate the prompt for the LLM
            prompt_text = f"{document_text}\n\nQuestion: {question}\nAnswer:"
            # Use the invoke method to get the answer
            response = llm.invoke(prompt_text)
            st.write(response)
        else:
            st.write("Please make sure both the document is selected and a question is entered.")

    # Download buttons logic remains the same as before

if __name__ == "__main__":
    main()
