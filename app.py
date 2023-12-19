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
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser




# webapp setup

# titel enzovoorts
st.set_page_config(page_title="VA-Polisvoorwaardentool")

# wachtwoord
hashed_password = st.secrets["hashed_password"]
password_input = st.text_input("Wachtwoord:", type="password")

# wachtwoord check
if sha256(password_input.encode()).hexdigest() != hashed_password:
    st.error("Voer het juiste wachtwoord in.")
    st.stop()



# Global variable to cache embeddings to reduce repeated API calls
knowledge_bases = {}

# pdf categorieen

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

    # Set API key for OpenAI functionalities
    api_key = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = api_key

    # Load the PDFs and categorize them
    pdf_dir = "preloaded_pdfs/"  # Adjust the path as needed
    all_pdfs = [os.path.join(dp, f) for dp, dn, filenames in os.walk(pdf_dir) for f in filenames if f.endswith('.pdf')]
    category_map = categorize_pdfs(all_pdfs)

    # Let the user select a category and a specific PDF
    categories = list(category_map.keys())
    selected_category = st.selectbox("Kies een categorie:", categories)
    available_pdfs = category_map[selected_category]
    pdf_names = [os.path.basename(pdf) for pdf in available_pdfs]
    selected_pdf_name = st.selectbox("Welke polisvoorwaarden wil je raadplegen?", pdf_names)

    # Get the full path of the selected PDF
    selected_pdf_index = pdf_names.index(selected_pdf_name)
    selected_pdf_path = available_pdfs[selected_pdf_index]

    # Process the selected PDF
    with open(selected_pdf_path, "rb") as f:
        pdf_reader = PdfReader(f)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=4000, chunk_overlap=1000, length_function=len)
        chunks = text_splitter.split_text(text)

    # Load embeddings and vector store for the selected PDF
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # Define the chat model and prompt template
    custom_prompt = ChatPromptTemplate.from_template(
        "Je bent een expert in het interpreteren van verzekeringsdocumenten. "
        "Bij het beantwoorden van vragen, gebruik de informatie uit de polisvoorwaarden. "
        "De gebruiker is een schadebehandelaar, geef altijd zo nuttig mogelijk antwoord. "
        "Vraag: {user_question}"
    )
    chat_model = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.2)

    # Get user input and generate response
    user_question = st.text_input("Stel een vraag over de polisvoorwaarden")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        relevant_doc_content = "Geen relevante inhoud gevonden in het document."
        if docs and len(docs) > 0:
            most_relevant_doc = docs[0]
            relevant_doc_content = getattr(most_relevant_doc, 'text', relevant_doc_content)

        chain = (
            {"user_question": relevant_doc_content + "\n\n" + user_question}
            | custom_prompt
            | chat_model
            | StrOutputParser()
        )
        response = chain.invoke()
        st.write(response)

if __name__ == '__main__':
    main()
