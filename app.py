import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI

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
    st.title("Polisvoorwaardentool - testversie 1.0")

    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

    if st.button('Schakel debugmodus in/uit'):
        st.session_state.debug_mode = not st.session_state.debug_mode

    categories = get_categories()
    selected_category = st.selectbox("Selecteer een categorie:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Selecteer een document:", documents)
    document_path = os.path.join(BASE_DIR, selected_category, selected_document)
    
    # Document download button
    if st.button('Download Document'):
        with open(document_path, "rb") as file:
            st.download_button(label="Download PDF", data=file, file_name=selected_document, mime='application/pdf')
    
    question = st.text_input("Stel een vraag over het document:")
    if question:
        document_text = extract_text_from_pdf(document_path)

        start_time = time.time()
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-preview")

        # Since batch processing and specific message types might not be directly supported,
        # consider using a simple prompt for the LLM call.
        prompt = f"{document_text}\n\nQuestion: {question}"
        try:
            response = llm.complete(prompt=prompt, max_tokens=512)  # Adjust parameters as needed
            
            if response:
                processing_time = time.time() - start_time
                st.write(response.choices[0].text)  # Display the first choice's text as the answer

                if st.session_state.debug_mode:
                    st.subheader("Debug Informatie")
                    st.write(f"Vraag: {question}")
                    st.write(f"Verwerkingstijd: {processing_time:.2f} seconden")
                    if st.checkbox('Toon documenttekst'):
                        st.write(document_text)
            else:
                st.error("Geen antwoord gegenereerd.")
        except Exception as e:
            st.error(f"Er is een fout opgetreden: {e}")

if __name__ == "__main__":
    main()
