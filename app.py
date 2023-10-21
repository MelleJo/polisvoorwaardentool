from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    st.set_page_config(page_title="VA-Polisvoorwaardentool")
    st.header("VA-Polisvoorwaardentool")

    # Get list of preloaded PDFs
    pdf_dir = "preloaded_pdfs/"
    available_pdfs = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    selected_pdf = st.selectbox("Welke polisvoorwaarden wil je raadplegen?", available_pdfs)
    
    # Read the selected PDF
    pdf_path = os.path.join(pdf_dir, selected_pdf)
    with open(pdf_path, "rb") as f:
        pdf_reader = PdfReader(f)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split into chunks
        text_spliter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_spliter.split_text(text)

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Show user input
        user_question = st.text_input("Stel een vraag over de polisvoorwaarden")
        if user_question: 
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()
