import streamlit as st
from PyPDF2 import PdfReader
import os
# Adjust imports according to the correct LangChain usage
from langchain.llms import OpenAI  # Assuming this is the correct import based on your setup

BASE_DIR = "/path/to/your/documents"

openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI model with the API key
llm = OpenAI(api_key=openai_api_key, model="gpt-4-turbo")  # Adjust model name as necessary

def list_categories(base_dir=BASE_DIR):
    return sorted(os.listdir(base_dir))

def list_documents(category):
    category_path = os.path.join(BASE_DIR, category)
    return [doc for doc in os.listdir(category_path) if doc.endswith('.pdf')]

def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def generate_response(document_text, question):
    # Constructing the prompt
    prompt = f"Given the document text: {document_text}\n\nQuestion: {question}\nAnswer:"
    response = llm.generate(prompt=prompt, max_tokens=500)  # Adjust max_tokens if necessary
    return response

def main():
    st.title("Polisvoorwaardentool Q&A")
    
    categories = list_categories(BASE_DIR)
    selected_category = st.selectbox("Choose a category:", categories)
    
    documents = list_documents(selected_category)
    selected_document = st.selectbox("Choose a document:", documents)
    
    user_question = st.text_input("Enter your question:")
    
    if st.button("Get Answer") and user_question:
        document_path = os.path.join(BASE_DIR, selected_category, selected_document)
        document_text = extract_text_from_pdf(document_path)
        answer = generate_response(document_text, user_question)
        st.write(answer)

if __name__ == "__main__":
    main()
