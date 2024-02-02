import streamlit as st
import os
from PyPDF2 import PdfReader
# Updated imports to use langchain-openai
from langchain_openai import OpenAI, ChatOpenAI

BASE_DIR = "preloaded_pdfs/PolisvoorwaardenVA"

openai_api_key = st.secrets["OPENAI_API_KEY"]
# Initialize OpenAI LLM with the API key
llm = OpenAI(api_key=openai_api_key)

def get_categories():
    return sorted(next(os.walk(BASE_DIR))[1])

def get_documents(category):
    category_path = os.path.join(BASE_DIR, category)
    return sorted([doc for doc in next(os.walk(category_path))[2] if doc.endswith('.pdf')])

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
    # The rest of your Streamlit app logic here

if __name__ == "__main__":
    main()
