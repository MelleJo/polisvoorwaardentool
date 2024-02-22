import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import AnalyzeDocumentChain
from langchain_community.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardenVA")

def get_categories():
    try:
        return sorted(next(os.walk(BASE_DIR))[1])
    except StopIteration:
        st.error(f"Failed to access categories in {BASE_DIR}. Check if the directory exists and is not empty.")
        return []

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
    st.title("Polisvoorwaardentool - verbeterde versie met FAISS")

    categories = get_categories()
    if not categories:
        return  # Stop further execution if no categories found
    
    selected_category = st.selectbox("Kies een categorie:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Selecteer een polisvoorwaardendocument:", documents)
    document_path = os.path.join(BASE_DIR, selected_category, selected_document)

    pdf = selected_document

    with open(document_path, "rb") as file:
        st.download_button(label="Download PDF", data=file, file_name=selected_document, mime="application/pdf")

    document_text = extract_text_from_pdf(document_path)  # Corrected to use document_path

    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
    chunks = text_splitter.split_text(document_text)

    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    # show user input
    user_question = st.text_input("Ask a question about your PDF:")
    
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        # Temporarily print the attributes of the first document to find out its structure
        # Corrected to use 'page_content' attribute for accessing the document text
        document_text = " ".join([doc.page_content for doc in docs])
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-preview")


        batch_messages = [
            [
            SystemMessage(content=document_text),
            HumanMessage(content=user_question),
            ],
        ]

        try:
            result = llm.generate(batch_messages)
            
            # Extracting the first response from the result
            if result.generations:
                response = result.generations[0][0].text  # Assuming the first generation of the first batch is what we want
                
                st.write(response)  # Display the answer
        
            else:
                st.error("No response generated.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        
            
if __name__ == "__main__":
    main()
