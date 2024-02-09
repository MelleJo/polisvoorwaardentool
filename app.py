import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI

# Set the base directory for your PDF documents
BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardenVA")

def get_categories():
    """Return a sorted list of categories based on subdirectories in BASE_DIR."""
    return sorted(next(os.walk(BASE_DIR))[1])

def get_documents(category):
    """Return a sorted list of PDF documents within a specified category."""
    category_path = os.path.join(BASE_DIR, category)
    return sorted([doc for doc in os.listdir(category_path) if doc.endswith('.pdf')])

def extract_text_from_pdf(file_path):
    """Extract and return text from a PDF file."""
    document_text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                document_text += text + "\n"
    return document_text

def main():
    """Main function for the Streamlit app."""
    st.title("Polisvoorwaardentool - testversie 1.0")

    # Debug mode toggle using a checkbox
    st.session_state.debug_mode = st.checkbox('Debugmodus', value=False)

    # User inputs for category and document selection
    categories = get_categories()
    selected_category = st.selectbox("Selecteer een categorie:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Selecteer een document:", documents)
    document_path = os.path.join(BASE_DIR, selected_category, selected_document)

    # Display a button to download the selected PDF document
    with open(document_path, "rb") as file:
        st.download_button(label="Download PDF", data=file, file_name=selected_document, mime='application/pdf')

    # Text input for user question
    question = st.text_input("Stel een vraag over het document:")
    if question:
        document_text = extract_text_from_pdf(document_path)
        start_time = time.time()
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-preview")

        # Generating answer based on the document text and user question
        prompt = f"{document_text}\n\nQuestion: {question}"
        try:
            response = llm.complete(prompt=prompt, max_tokens=512)  # Adjust parameters as needed
            if response:
                processing_time = time.time() - start_time
                st.write(response.choices[0].text)  # Display the answer

                if st.session_state.debug_mode:
                    debug_information(processing_time, question, document_text, response.choices[0].text)
            else:
                st.error("Geen antwoord gegenereerd.")
        except Exception as e:
            st.error(f"Er is een fout opgetreden: {e}")

def debug_information(processing_time, question, document_text, response_text):
    """Display debug information if debug mode is enabled."""
    st.subheader("Debug Informatie")
    st.write(f"Vraag: {question}")
    st.write(f"Verwerkingstijd: {processing_time:.2f} seconden")
    if st.checkbox('Toon documenttekst', value=False):
        st.write(document_text)
    st.write(f"Antwoord: {response_text}")

if __name__ == "__main__":
    main()
