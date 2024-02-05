import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
#from langchain_community import ConversationalRetrievalChain

# Initialize the OpenAI model with your API key
openai_api_key = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(api_key=openai_api_key)
prompt = ChatPromptTemplate.from_template(
    "Beantwoord de volgende vraag {question} over de volgende voorwaarden {document_text}"
)


# Setup your base directory
BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardenVA")

def get_categories():
    """Get a list of categories based on folder names."""
    return sorted(next(os.walk(BASE_DIR))[1])

def get_documents(category):
    """Get a list of document names for a given category."""
    category_path = os.path.join(BASE_DIR, category)
    return sorted([doc for doc in next(os.walk(category_path))[2] if doc.endswith('.pdf')])

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file given its path."""
    document_text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                document_text += text + "\n"
    return document_text

def answer_question(document_text, question): 
    chain = prompt | llm
    response = chain.invoke({"question": question, "document_text": document_text})
    return response


def main():
    st.title("Polisvoorwaardentool")

    categories = get_categories()
    selected_category = st.selectbox("Select a category:", categories)

    documents = get_documents(selected_category)
    selected_document = st.selectbox("Select a document:", documents)

    document_path = os.path.join(BASE_DIR, selected_category, selected_document)
    document_text = extract_text_from_pdf(document_path)
    
    # UI to ask a question
    question = st.text_input("Ask a question about the document:")
    if st.button("Get Answer"):
        if document_text and question:
            answer = answer_question(document_text, question)
            st.write(answer)
        else:
            st.write("Please make sure both the document is selected and a question is entered.")

if __name__ == "__main__":
    main()
