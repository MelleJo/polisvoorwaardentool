import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
# Adjust import paths for SystemMessage and HumanMessage as necessary, 
# the following line is based on the provided code structure and may need updating
from langchain.llms.messages import SystemMessage, HumanMessage  

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

    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

    if st.button('Toggle Debug Mode'):
        st.session_state.debug_mode = not st.session_state.debug_mode

    categories = get_categories()
    selected_category = st.selectbox("Select a category:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Select a document:", documents)
    document_path = os.path.join(BASE_DIR, selected_category, selected_document)
    document_text = extract_text_from_pdf(document_path)
    question = st.text_input("Ask a question about the document:")

    if st.button("Get Answer") and document_text and question:
        start_time = time.time()
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-preview")

        # Format the messages for batch processing
        batch_messages = [
            [
                SystemMessage(content=document_text),
                HumanMessage(content=question),
            ],
        ]
        try:
            result = llm.generate(batch_messages)
            
            # Extracting the first response from the result
            if result.generations:
                response = result.generations[0][0].text  # Assuming the first generation of the first batch is what we want
                processing_time = time.time() - start_time

                st.write(response)  # Display the answer

                if st.session_state.debug_mode:
                    st.subheader("Debug Information")
                    st.write(f"Question: {question}")
                    st.write(f"Processing Time: {processing_time:.2f} seconds")
                    st.write(f"Token Usage: {result.llm_output['token_usage']}")
                    if st.checkbox('Show Document Text'):
                        st.write(document_text)
            else:
                st.error("No response generated.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
