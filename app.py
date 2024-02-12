import streamlit as st
import os
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Set the base directory for preloaded PDFs
BASE_DIR = os.path.join(os.getcwd(), "preloaded_pdfs", "PolisvoorwaardentoolVA")

# Initialize tokenizer and model for embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Function to get categories from the base directory
def get_categories():
    return sorted(os.listdir(BASE_DIR))

# Function to get document names within a selected category
def get_documents(category):
    category_path = os.path.join(BASE_DIR, category)
    return sorted([doc for doc in os.listdir(category_path) if doc.endswith('.pdf')])

# Load FAISS index and document chunk map
def load_faiss_index(index_path, doc_map_path):
    index = faiss.read_index(index_path)
    with open(doc_map_path, 'rb') as f:
        doc_map = torch.load(f)
    return index, doc_map

# Generate embedding for query
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# Search FAISS index for relevant chunks
def search_index(index, query_embedding, doc_map, top_k=5):
    distances, indices = index.search(query_embedding, top_k)
    results = [(doc_map[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
    return results

# Function to extract text from a specific chunk in PDF
def extract_text_from_chunk(file_path, chunk_range):
    document_text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page_number in range(chunk_range[0], min(chunk_range[1] + 1, len(reader.pages))):
            text = reader.pages[page_number].extract_text()
            if text:
                document_text += text + "\n"
    return document_text

# Main Streamlit app function
def main():
    st.title("Polisvoorwaardentool - Testversie 1.0")
    
    # Debug mode toggle
    debug_mode = st.checkbox('Debugmodus', value=False)

    # Model version selection
    model_choice = st.selectbox("Kies model versie:", ["ChatGPT 3.5 Turbo", "gpt-4-turbo-preview"])
    model_version = "gpt-3.5-turbo" if model_choice == "ChatGPT 3.5 Turbo" else "gpt-4-turbo-preview"

    # Document selection UI
    categories = get_categories()
    selected_category = st.selectbox("Selecteer een categorie:", categories)
    documents = get_documents(selected_category)
    selected_document = st.selectbox("Selecteer een document:", documents)
        
    # Question input
    question = st.text_input("Stel een vraag over het document:")

    if st.button("Krijg Antwoord") and question:
        # Load FAISS index and document map
        index_path = "path_to_your_faiss_index"
        doc_map_path = "path_to_your_document_map"
        index, doc_map = load_faiss_index(index_path, doc_map_path)

        # Embed the question
        question_embedding = embed_text(question)

        # Search the index for relevant chunks
        search_results = search_index(index, question_embedding, doc_map)

        # Assuming the most relevant chunk is the first one
        most_relevant_chunk_info = search_results[0][0]  # (document_name, chunk_range)
        chunk_text = extract_text_from_chunk(os.path.join(BASE_DIR, selected_category, most_relevant_chunk_info[0]), most_relevant_chunk_info[1])

        # Initialize ChatOpenAI with the selected model
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model=model_version)

        # Prepare the messages for the chat
        messages = [
            SystemMessage(content="Jij bent een expert in het analyseren van polisvoorwaarden. De gebruiker is een schadebehandelaar en wil graag jouw hulp bij het vinden van specifieke en relevante informatie voor de schadebehandeling van een polis. Nauwkeurigheid is prioriteit nummer 1"),
            SystemMessage(content=chunk_text),
            HumanMessage(content=question)
        ]
        
        # Get the response
        try:
            response = llm.invoke(messages)
            if response:
                st.write(response.content)
                if debug_mode:
                    st.subheader("Debug Informatie")
                    st.write(f"Vraag: {question}")
                    if st.checkbox('Toon documenttekst'):
                        st.write(chunk_text)
            else:
                st.error("Geen antwoord gegenereerd.")
        except Exception as e:
            st.error(f"Er is een fout opgetreden: {e}")

if __name__ == "__main__":
    main()
