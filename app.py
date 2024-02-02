import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.llms import OpenAI
from langchain.chats import ChatCompletion

# Setup API key and initialize LangChain OpenAI model
openai_api_key = st.secrets["OPENAI_API_KEY"]
llm = OpenAI(api_key=openai_api_key, model="gpt-4-turbo")

# Define paths to your document categories
BASE_DIR = "/path/to/your/documents"

def list_categories(base_dir=BASE_DIR):
    """List all categories (directories) within the base directory."""
    return next(os.walk(base_dir))[1]

def list_documents(category):
    """List all documents within a given category."""
    category_path = os.path.join(BASE_DIR, category)
    return [doc for doc in os.listdir(category_path) if doc.endswith('.pdf')]

def extract_text_from_pdf(filepath):
    """Extract text from a PDF file."""
    text = ""
    with open(filepath, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def main():
    st.title("Polisvoorwaardentool Q&A")
    
    # Document selection UI
    category = st.selectbox("Choose a category:", list_categories())
    document_name = st.selectbox("Choose a document:", list_documents(category))
    question = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        document_path = os.path.join(BASE_DIR, category, document_name)
        document_text = extract_text_from_pdf(document_path)
        
        # Format input for LangChain ChatCompletion
        chat_input = {"messages": [{"role": "system", "content": document_text}, {"role": "user", "content": question}]}
        
        # Get answer from LangChain model
        chat_model = ChatCompletion(llm=llm)
        answer = chat_model.complete(chat_input)
        
        # Display the answer
        st.write(answer["choices"][0]["message"]["content"])

if __name__ == "__main__":
    main()
