import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

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

    if st.button("Get Answer") and document_text and question:
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-preview")
        
        messages = [
            SystemMessage(content="You're a helpful assistant."),
            HumanMessage(content=question),
        ]
        
        try:
            response = llm.invoke(input={'messages': messages})
            if response:
                st.write(response.choices[0].text)
            else:
                st.error("No response generated.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
