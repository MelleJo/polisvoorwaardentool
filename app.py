import os
import openai
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
#from openai import OpenAI
from langchain.callbacks import get_openai_callback
#from langchain.memory import ConversationBufferMemory
from hashlib import sha256

assistant_id = st.secrets["openai_assistant_id"]

st.set_page_config(page_title="VA-Polisvoorwaardentool")

hashed_password = st.secrets["hashed_password"]
password_input = st.text_input("Wachtwoord:", type="password")

if sha256(password_input.encode()).hexdigest() != hashed_password:
    st.error("Voer het juiste wachtwoord in.")
    st.stop()
    
knowledge_bases = {}

def submit_message(assistant_id, thread_id, user_message):
    response = openai.Message.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "system", "content": "Jij bent een expert in schadebehandelingen en het begrijpen en analyseren van polisvoorwaarden."},
                  {"role": "user", "content": user_message}],
        assistant_id=assistant_id,
        thread_id=thread_id
    )
    return response

def get_response(thread_id):
    response = openai.Thread.retrieve(thread_id=thread_id)
    return response


   
def categorize_pdfs(pdf_list):
    category_map = {}
    for pdf in pdf_list:
        prefix = os.path.basename(pdf).split('_')[0]
            if prefix == "Auto":
                category = "Autoverzekering"
            elif prefix == "AVB":
                category = "Bedrijfsaansprakelijkheidsverzekering"
            elif prefix == "BestAVB":
                category = "AVB Bestuurders"
            elif prefix == "BS":
                category = "Bedrijfsschadeverzekering"
            elif prefix == "BestAuto":
                category = "Bestelautoverzekering"
            elif prefix == "Brand":
               category = "Brandverzekeringen"
            elif prefix == "Cara":
                category = "Caravanverzekering"
            elif prefix == "EigVerv":
                category = "Eigen vervoer"
            elif prefix == "Fiets":
                category = "Fietsverzekering"
            elif prefix == "Geb":
                category = "Gebouwen"
            elif prefix == "GW":
                category = "Goed Werkgeverschap"
            elif prefix == "IB":
                category = "Inboedelverzekering"
            elif prefix == "Inv":
                category = "Inventaris"
            elif prefix == "Mot":
                category = "Motorverzekering"
            elif prefix == "RB":
                category = "Rechtsbijstandverzekering"
            elif prefix == "Reis":
                category = "Reisverzekering"
            elif prefix == "Scoot":
                category = "Scootmobielverzekering"
            elif prefix == "WEGAS":
                category = "WEGAS"
            elif prefix == "WerkMat":
                category = "Werk- en landbouwmaterieelverzekering"
            elif prefix == "WEGAM":
                category = "Werkgeversaansprakelijkheid Motorrijtuigen (WEGAM)"
            elif prefix == "Woon":
                category = "Woonhuisverzekering"
            else:
                category = "Overige"

        if category not in category_map:
            category_map[category] = []
        category_map[category].append(pdf)

    return category_map

def main():
    
    session_id = st.session_state._get_session_id()
    thread_id = f"thread_{session_id}"

    
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
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            embeddings = OpenAIEmbeddings()
            knowledge_bases[selected_pdf_path] = FAISS.from_texts(chunks, embeddings)

    knowledge_base = knowledge_bases[selected_pdf_path]

    custom_system_prompt = "Jij bent een expert in schadebehandelingen en het begrijpen en analyseren van polisvoorwaarden. Geef een duidelijke bronvermelding van pagina's, hoofdstukken of paragrafen. Start elke zin met HALLO"
    system_message_template = SystemMessagePromptTemplate.from_template(custom_system_prompt)
    
    user_question = st.text_input("Stel een vraag over de polisvoorwaarden")
    if user_question:
        response = submit_message(assistant_id, "unique_thread_id_for_user", user_question)

        if response:
            answer = response["choices"][0]["message"]["content"]
            st.write(answer)
        

       

      
      

if __name__ == '__main__':
    main()
