import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.llms import OpenAI  # Using OpenAI LLM for direct text generation

# Correct retrieval of the API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI model with the API key
llm = OpenAI(api_key=openai_api_key)

BASE_DIR = "preloaded_pdfs/PolisvoorwaardenVA"

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

def get_gpt_response(document_text, user_question):
    prompt = f"Analyse the following document from the perspective of a claims handler and answer the user's question:\n\n{document_text}\n\nQuestion: {user_question}\nAnswer:"
    response = llm.generate(prompt=prompt, max_tokens=500)  # Adjust max_tokens as needed
    return response.text.strip()  # Ensure you're accessing the text attribute of the response

def main():
    st.title("Polisvoorwaardentool")

    categories = get_categories()
    selected_category = st.selectbox("Select a category:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Select a document:", documents)
    user_question = st.text_input("Vraag:")

    if user_question:
        document_path = os.path.join(BASE_DIR, selected_category, selected_document)
        document_text = extract_text_from_pdf(document_path)
        answer = get_gpt_response(document_text, user_question)
        st.text_area("Answer", answer, height=300)

if __name__ == "__main__":
    main()
