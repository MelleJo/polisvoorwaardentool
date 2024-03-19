import streamlit as st
import os
import time
import pyperclip
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import AnalyzeDocumentChain
from langchain_community.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate



BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardenVA")

company_name_mapping = {
    "nn": "Nationale Nederlanden",
    "asr": "a.s.r.",
    "nlg": "NLG Verzekeringen",
    "avero": "Avéro Achmea",
    "Avero-p-r521": "Avéro Achmea",
    "europeesche": "Europeesche Verzekeringen",
    "aig": "AIG",
    "allianz": "Allianz",
    "bikerpolis": "Bikerpolis",
    "das": "DAS",
    "guardian": "Guardian",
    "noordeloos": "Noordeloos",
    "reaal": "Reaal",
    "unigarant": "Unigarant",
}

def get_all_documents():
    all_docs = []
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith('.pdf'):
                path = os.path.join(root, file)
                all_docs.append({'title': file, 'path': path})
    return all_docs

def get_insurance_companies(all_documents):
    companies = set()
    for doc in all_documents:
        parts = doc['title'].split('_')
        if len(parts) >= 2:
            company_key = parts[1].lower()
            company_name = company_name_mapping.get(company_key, company_key.capitalize())
            companies.add(company_name)
    return sorted(companies)

def get_categories():
    try:
        return sorted(next(os.walk(BASE_DIR))[1])
    except StopIteration:
        st.error("Fout bij het openen van de categorieën. Controleer of de map bestaat en niet leeg is.")
        return []

def get_documents(category):
    category_path = os.path.join(BASE_DIR, category)
    return sorted([doc for doc in os.listdir(category_path) if doc.endswith('.pdf')])

def extract_text_from_pdf_by_page(file_path):
    pages_text = []
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    return pages_text

def process_document(document_path, user_question):
    with st.spinner('Denken...'):
        # Extract text from the document
        document_pages = extract_text_from_pdf_by_page(document_path)
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(document_pages, embeddings)
        docs = knowledge_base.similarity_search(user_question)
        document_text = " ".join([doc.page_content for doc in docs])

        template = """
        Taakomschrijving: Als ervaren schadebehandelaar met grondige kennis van polisvoorwaarden, is het je primaire doel om vragen over dekkingen, uitsluitingen, en voorwaarden accuraat te beantwoorden, met directe verwijzingen naar de betreffende polisvoorwaardendocumenten. Je antwoorden moeten direct uit deze documenten komen en specifiek citeren waar de informatie te vinden is, inclusief paginanummers of sectienummers indien beschikbaar.

        Verwachtingen voor Antwoorden: Antwoord met precisie en bied directe citaten uit de polisvoorwaarden waar mogelijk. Vermeld altijd de locatie van de informatie (paginanummers of sectienummers). Als een vraag niet direct kan worden beantwoord met de beschikbare documentatie, geef dan duidelijk aan welke aanvullende informatie of verduidelijking van de gebruiker nodig is.

        Instructies:

        Identificeer snel dekkingen en uitsluitingen relevant voor de gebruikersvraag. Focus op kerninformatie en vermijd overbodige details.
        Citeer de relevante passage(s) uit de polisvoorwaarden die jouw bevindingen ondersteunen, met vermelding van de sectie- of paginanummers.
        Eindig je antwoord met een beknopte conclusie die de vraag direct beantwoordt, waarbij je duidelijk maakt of de situatie gedekt is, inclusief eventuele specifieke voorwaarden of maximale vergoedingen.
        Voorbeeld: Als de vraag is: 'Is schade door wateroverlast gedekt onder mijn polis?', zoek dan naar secties die wateroverlast behandelen. Je kunt bijvoorbeeld citeren: 'Wateroverlast wordt gedekt tot een maximum van €10.000 per gebeurtenis, zoals vermeld in Sectie 4.3 van uw polisvoorwaarden.' Sluit af met een duidelijke conclusie over de dekking.

        Zorg voor een duidelijke, beknopte conclusie na je analyse. Dit helpt de gebruiker om snel het antwoord op hun vraag te begrijpen.

        Gebruik de bovenstaande instructies om de vraag van de gebruiker te beantwoord: {user_question} op basis van {document_text}, je kennis komt enkel voort uit {document_text}. 

        """
        
        prompt = ChatPromptTemplate.from_template(template)

        
        # Perform similarity search
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-0125-preview", temperature=0, streaming=True)
        chain = prompt | llm | StrOutputParser() 
        return chain.stream({
            "document_text": document_text,
            "user_question": user_question,
        })
    


def display_search_results(search_results):
    if not search_results:
        st.write("Geen documenten gevonden.")
        return
    
    if isinstance(search_results[0], str):
        search_results = [{'title': filename, 'path': os.path.join(BASE_DIR, filename)} for filename in search_results]

    selected_title = st.selectbox("Zoekresultaten:", [doc['title'] for doc in search_results])
    selected_document = next((doc for doc in search_results if doc['title'] == selected_title), None)
    
    if selected_document:
        user_question = st.text_input("Stel een vraag over de polisvoorwaarden:")
        if user_question:
            # Call process_document and use its return value as the argument for st.write_stream
            document_stream = process_document(selected_document['path'], user_question)
            st.write_stream(document_stream)  # Correctly pass the generator/stream to st.write_stream

        # Download button for the selected PDF file
        with open(selected_document['path'], "rb") as file:
            btn = st.download_button(
                label="Download polisvoorwaarden",
                data=file,
                file_name=selected_document['title'],
                mime="application/pdf"
            )


    

def main():
    st.title("Polisvoorwaardentool - stabiele versie 1.2.3.")
    all_documents = get_all_documents()
    selection_method = st.radio("Hoe wil je de polisvoorwaarden selecteren?:", 
                                ['Zoeken', 'Categoriën', 'Per maatschappij'])

    if selection_method == 'Zoeken':
        search_query = st.text_input("Zoek naar een polisvoorwaardenblad:", "")
        if search_query:
            search_results = [doc for doc in all_documents if search_query.lower() in doc['title'].lower()]
            display_search_results(search_results)

    elif selection_method == 'Categoriën':
        categories = get_categories()
        if categories:
            selected_category = st.selectbox("Kies een categorie:", categories)
            document_filenames = get_documents(selected_category)  # Returns filenames
            # Construct full paths for documents in the selected category
            documents_with_paths = [{'title': filename, 'path': os.path.join(BASE_DIR, selected_category, filename)} for filename in document_filenames]
            display_search_results(documents_with_paths)

    elif selection_method == 'Per maatschappij':
        companies = get_insurance_companies(all_documents)
        selected_company = st.selectbox("Kies een maatschappij:", companies)
        if selected_company:
            original_keys = [key for key, value in company_name_mapping.items() if value.lower() == selected_company.lower()]
            if not original_keys:
                original_keys = [selected_company.lower()]
            company_documents = [doc for doc in all_documents if any(key for key in original_keys if key in doc['title'].lower())]
            display_search_results(company_documents)

if __name__ == "__main__":
    main()
