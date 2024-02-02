import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate

BASE_DIR = "preloaded_pdfs/PolisvoorwaardenVA"

# Corrected secrets access
openai_api_key = st.secrets["OPENAI_API_KEY"]
# Initialize the model with the API key
model = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4-turbo-preview", temperature=0.20)

human_message_prompt = HumanMessagePromptTemplate.from_template(prompt_template)

chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])

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

    categories = get_categories()
    selected_category = st.selectbox("Select a category:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Select a document:", documents)
    
    user_question = st.text_input("Vraag:")
    
    if user_question:
        document_path = os.path.join(BASE_DIR, selected_category, selected_document)
        document_text = extract_text_from_pdf(document_path)
        # Format the prompt with the extracted text and the user's question
        prompt = f"{document_text}\n\nJij bent een expert in het analyseren van polisvoorwaarden vanuit het perspectief van een schadebehandelaar, geef antwoord op de vraag van gebruiker: {user_question}?"
        
        # Invoke the model
        response = model.generate(prompt, max_tokens=500)  # Adjust max_tokens if needed
        
        # Display the response
        st.text_area("Antwoord", response, height=150)

if __name__ == "__main__":
    main()
