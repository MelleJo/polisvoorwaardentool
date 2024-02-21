import streamlit as st
from datetime import datetime
import hashlib
from langchain.chains import AnalyzeDocumentChain
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import openai

# Configuration details from Streamlit secrets or environment variables
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

# Pinecone client initialization
pc = Pinecone(api_key=PINECONE_API_KEY)

# Your Pinecone index specifics
index_name = "polisvoorwaardentoolindex"
dimension = 1536  # Your index dimension
metric = 'cosine'  # Metric used in your index

# Ensure the index exists or create it
#if index_name not in pc.list_indexes().names:
    #pc.create_index(
        #name=index_name,
      #  dimension=dimension,
       # metric=metric,
      #  spec=(
        #    cloud='gcp',
       #     region='us-central1',  # Iowa
      #  )
 #   )

# Obtain a handle to your index
index = pc.Index(index_name)

# LangChain OpenAI embeddings initialization
embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")

def get_md5_hash(text):
    """Generate MD5 hash for a given text."""
    return hashlib.md5(text.encode()).hexdigest()


# Document processing and vectorization using LangChain
def process_and_vectorize_document(file_path):
    analyze_document_chain = AnalyzeDocumentChain(embeddings=embeddings_model)
 # Initialize with necessary parameters
    document_chunks = analyze_document_chain.run(file_path)
    for chunk in document_chunks:
        vector = embeddings_model.get_embeddings(chunk)
        # Upsert each chunk to Pinecone with a unique ID and associated vector
        chunk_id = hashlib.md5(chunk.encode()).hexdigest()
        index.upsert(id=chunk_id, vector=vector, metadata={"last_modified": datetime.now().isoformat()})

# Query handling with MMR option
def query_document(question, use_mmr=False):
    question_vector = embeddings_model.get_embeddings(question)
    if use_mmr:
        results = index.query_mmr(question_vector, ...) # Pseudocode for MMR query
    else:
        results = index.query(vector=question_vector, top_k=1)
    return results

# Main application logic
def main():
    st.title("Polisvoorwaardentool - Enhanced with LangChain and Pinecone")
    document_path = st.file_uploader("Upload a document", type=["pdf"])
    if document_path:
        document_id = get_md5_hash(document_path.name)
        process_and_vectorize_document(document_path)

        question = st.text_input("Enter your query:")
        use_mmr = st.checkbox("Diversify results", value=False)
        if question and st.button("Search"):
            result_id = query_document(question, use_mmr=use_mmr)
            if result_id:
                st.success(f"Document ID {result_id} is the most relevant.")
            else:
                st.error("No relevant document found.")

if __name__ == "__main__":
    main()
