import os
import streamlit as st
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext, load_index_from_storage
from llama_index.llms import ChatOpenAI
from pathlib import Path

# Set page config
st.set_page_config(page_title="Polisvoorwaardentool")

# Check password
hashed_password = st.secrets["hashed_password"]
password_input = st.text_input("Wachtwoord:", type="password")

if sha256(password_input.encode()).hexdigest() != hashed_password:
    st.error("Voer het juiste wachtwoord in.")
    st.stop()

# Initialize the index
index_name = "./saved_index"
documents_folder = "./documents"

@st.cache_resource
def initialize_index(index_name, documents_folder):
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    service_context = ServiceContext.from_defaults(llm_predictor=chat_model)
    
    if os.path.exists(index_name):
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name),
            service_context=service_context,
        )
    else:
        documents = SimpleDirectoryReader(documents_folder).load_data()
        index = GPTVectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        index.storage_context.persist(persist_dir=index_name)

    return index

index = None
api_key = st.text_input("Enter your OpenAI API key here:", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    index = initialize_index(index_name, documents_folder)

if index is None:
    st.warning("Please enter your API key first.")

# Categorize PDFs
pdf_dir = "preloaded_pdfs/"
all_pdfs = [os.path.join(dp, f) for dp, dn, filenames in os.walk(pdf_dir) for f in filenames if f.endswith('.pdf')]
category_map = {}

for pdf in all_pdfs:
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

    category_map.setdefault(category, []).append(pdf)

categories = list(category_map.keys())
if not categories:
    st.warning("Geen polisvoorwaarden gevonden.")
    st.stop()

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

text = st.text_input("Query text:", value="What did the author do growing up?")

if st.button("Run Query") and text is not None:
    response = query_index(index, text)
    st.markdown(response)

    llm_col, embed_col = st.columns(2)
    with llm_col:
        st.markdown(
            f"LLM Tokens Used: {index.service_context.llm_predictor._last_token_usage}"
        )

    with embed_col:
        st.markdown(
            f"Embedding Tokens Used: {index.service_context.embed_model._last_token_usage}"
        )
