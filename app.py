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

prompt_template = ChatPromptTemplate.from_template(
    "Je bent een expert in het interpreteren van verzekeringsdocumenten. "
    "Bij het beantwoorden van vragen, gebruik de informatie uit de polisvoorwaarden. "
    "Geef specifieke pagina- en paragraafnummers voor bronvermelding waar mogelijk. "
    "Vraag: {user_question}"
)


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
    st.header("VA-Polisvoorwaardentool")

    chat = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)
    chain = prompt_template | chat

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
    custom_system_prompt = "Jij bent een expert in schadebehandelingen en het begrijpen en analyseren van polisvoorwaarden. Geef een duidelijke bronvermelding van pagina's, hoofdstukken of paragrafen. Start elke zin met HALLO. Beantwoord de vraag: {user_question}" 
    system_message_template = SystemMessagePromptTemplate.from_template(custom_system_prompt)

    user_question = st.text_input("Stel een vraag over de polisvoorwaarden")
    docs = None
    
    if user_question:
        # Perform document similarity search
        docs = knowledge_base.similarity_search(user_question)

    # Check if documents are found and extract text from the first document
    if docs and len(docs) > 0:
        # Assuming the first document is the most relevant one
        most_relevant_doc = docs[0]

        # Extracting text based on your original implementation structure
        # Update this part based on how the 'Document' object stores text in your original code
        relevant_doc_content = getattr(most_relevant_doc, 'text', "Geen relevante inhoud gevonden in het document.")
    else:
        relevant_doc_content = "Geen relevante inhoud gevonden in het document."

    # Combine the relevant document content with the user question
    combined_input = {"user_question": relevant_doc_content + "\n\n" + user_question}

    # Invoke the chain with the combined input
    response = chain.invoke(combined_input)
    st.write(response.content)



if __name__ == '__main__':
    main()
