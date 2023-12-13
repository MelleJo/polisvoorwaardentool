import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
from hashlib import sha256

# Set page config
st.set_page_config(page_title="VA-Polisvoorwaardentool")

# Check password
hashed_password = st.secrets["hashed_password"]
password_input = st.text_input("Wachtwoord:", type="password")

if sha256(password_input.encode()).hexdigest() != hashed_password:
    st.error("Voer het juiste wachtwoord in.")
    st.stop()

# Global variable to cache embeddings to reduce repeated API calls
knowledge_bases = {}

def categorize_pdfs(pdf_list):
    category_map = {}
    for pdf in pdf_list:
        prefix = os.path.basename(pdf).split('_')[0]
        category = "Overige"
        if prefix == "Auto":
            category = "Autoverzekering"
        # ... (Add additional categorization here)
        if category not in category_map:
            category_map[category] = []
        category_map[category].append(pdf)

    return category_map

def main():
    st.header("VA-Polisvoorwaardentool")

    api_key = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = api_key

    pdf_dir = "preloaded_pdfs/"
    all_pdfs = [os.path.join(dp, f) for dp, dn, filenames in os.walk(pdf_dir) for f in filenames if f.endswith('.pdf')]

    category_map = categorize_pdfs(all_pdfs)

    categories = list(category_map.keys())
    if not categories:
        st.warning("Geen polisvoorwaarden gevonden.")
        return

    selected_category = st.selectbox("Kies een categorie:", categories)
    available_pdfs = category_map[selected_category]
    pdf_names = [os.path.basename(pdf) for pdf in available_pdfs]
    selected_pdf_name = st.selectbox("Welke polisvoorwaarden wil je raadplegen?", pdf_names)
    selected_pdf_path = available_pdfs[pdf_names.index(selected_pdf_name)]

    if selected_pdf_path:
        with open(selected_pdf_path, "rb") as file:
            st.download_button(
                label="Download polisvoorwaarden",
                data=file,
                file_name=selected_pdf_name,
                mime="application/pdf"
            )

    if selected_pdf_path not in knowledge_bases:
        with open(selected_pdf_path, "rb") as f:
            pdf_reader = PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_text(text)
            embeddings = OpenAIEmbeddings()
            knowledge_bases[selected_pdf_path] = FAISS.from_texts(chunks, embeddings)

    knowledge_base = knowledge_bases[selected_pdf_path]
    custom_system_prompt = "Jij bent een expert in schadebehandelingen en het begrijpen en analyseren van polisvoorwaarden. Geef een duidelijke bronvermelding van pagina's, hoofdstukken of paragrafen. Start elke zin met HALLO"
    system_message_template = SystemMessagePromptTemplate.from_template(custom_system_prompt)

    user_question = st.text_input("Stel een vraag over de polisvoorwaarden")
    if user_question:
        messages = [system_message_template.format(), HumanMessage(content=user_question)]
        docs = knowledge_base.similarity_search(user_question)  # Zoek het meest relevante deel van het document

        chat = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)
        chat(messages)
        chain = load_qa_chain(chat, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=(user_question))
        st.write(response)

if __name__ == '__main__':
    main()
