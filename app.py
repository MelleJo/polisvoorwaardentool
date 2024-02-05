import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Initialize the OpenAI model with your API key
openai_api_key = st.secrets["OPENAI_API_KEY"]
model = ChatOpenAI(api_key=openai_api_key, model="gpt-4-turbo-preview")

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
            prompt = ChatPromptTemplate.from_template("Answer the following question about the document: {question}")
            response = model(prompt.format(question=question, document_text=document_text))
            st.write(response)
        else:
            st.write("Please make sure both the document is selected and a question is entered.")
    
    # Download buttons
    if st.button("Download PDF"):
        with open(document_path, "rb") as file:
            btn = st.download_button(label="Download PDF", data=file, file_name=selected_document, mime='application/pdf')
    if st.button("Download Text Version"):
        btn = st.download_button(label="Download Text", data=document_text.encode('utf-8'), file_name=selected_document.replace('.pdf', '.txt'), mime='text/plain')

if __name__ == "__main__":
    main()
