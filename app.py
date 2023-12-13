import openai
import streamlit as st
import uuid
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from hashlib import sha256
import os

# Set OpenAI API key from Streamlit secrets
api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = api_key
assistant_id = st.secrets["openai_assistant_id"]

# Set Streamlit page config
st.set_page_config(page_title="VA-Polisvoorwaardentool")

# Check password
hashed_password = st.secrets["hashed_password"]
password_input = st.text_input("Wachtwoord:", type="password")
if sha256(password_input.encode()).hexdigest() != hashed_password:
    st.error("Voer het juiste wachtwoord in.")
    st.stop()

# Function to start or get a thread
def start_or_get_thread():
    if 'thread_id' not in st.session_state:
        response = openai.thread.create(assistant_id=assistant_id)
        st.session_state['thread_id'] = response['data']['id']
        st.write(f"thread created: {st.session_state['thread_id']}")
    return st.session_state['thread_id']

# Function to send a message to the OpenAI thread and get a response
def send_message_get_response(thread_id, user_message):
    response = openai.message.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "user", "content": user_message}],
        assistant_id=assistant_id,
        thread_id=thread_id
    )
    st.write(f"API Response: {response}")
    return response['data']['content']

# Function to categorize PDFs
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

# Main function
def main():
    st.header("VA-Polisvoorwaardentool")

    # Load PDF files
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
    user_question = st.text_input("Stel een vraag over de polisvoorwaarden")

    if selected_pdf_path and user_question:
        with open(selected_pdf_path, "rb") as file:
            st.download_button(label="Download polisvoorwaarden", data=file, file_name=selected_pdf_name, mime="application/pdf")

        thread_id = start_or_get_thread()
        assistant_response = send_message_get_response(thread_id, user_question)
        st.write(assistant_response)

if __name__ == '__main__':
    main()
