import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import faiss
import pickle
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configuring Google Gemini API key
genai.configure(api_key= GOOGLE_API_KEY)

# Function to save files to a specific directory
def save_uploaded_file(uploaded_file, upload_dir="uploaded_files"):
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path

# Function to process uploaded files
def process_files(files):
    documents = []
    for file in files:
        file_path = save_uploaded_file(file)  # Save the file and get its path
        file_extension = os.path.splitext(file.name)[1].lower()

        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path=file_path)
            documents.extend(loader.load())
        elif file_extension in [".docx", ".doc"]:
            loader = UnstructuredWordDocumentLoader(file_path=file_path)
            documents.extend(loader.load())
        elif file_extension in [".pptx", ".ppt"]:
            loader = UnstructuredPowerPointLoader(file_path=file_path)
            documents.extend(loader.load())
        else:
            st.warning(f"Unsupported file type: {file_extension}. Skipping.")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = [model.encode(chunk.page_content) for chunk in chunks]

    # Save embeddings and metadata
    embedded_data = [{"chunk": chunk.page_content, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)]
    embeddings_array = np.array(embeddings, dtype='float32')
    metadata = [{"id": f"chunk-{i}", "text": chunk.page_content} for i, chunk in enumerate(chunks)]
    faiss_index = faiss.IndexFlatL2(embeddings_array.shape[1])
    faiss_index.add(embeddings_array)

    # Save the index and metadata
    faiss.write_index(faiss_index, "faiss_index.bin")
    with open("faiss_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    return len(chunks)

def get_relevant_chunks(query, top_k=2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query).astype('float32')

    index = faiss.read_index("faiss_index.bin")
    D, I = index.search(np.array([query_embedding]), top_k)

    with open("faiss_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    relevant_chunks = [metadata[i]["text"] for i in I[0]]
    return relevant_chunks

def answer_question(query):
    chunks = get_relevant_chunks(query)
    context = "\n".join(chunks)

    model = genai.GenerativeModel(model_name='gemini-1.5-pro')
    response = model.generate_content(f"Answer the following questions using the provided context:\n\nQuestion: {query}\n\nContext: {context}")
    return response.text

def main():
    st.title("Doxpert.ai")

    uploaded_files = st.file_uploader("Upload your documents (PDF, Word, PPT, or Images)", type=["pdf", "docx", "doc", "ppt", "pptx"], accept_multiple_files=True)

    if uploaded_files:
        st.write("Processing uploaded files...")
        num_chunks = process_files(uploaded_files)
        st.success(f"Files processed successfully! {num_chunks} chunks created.")

    query = st.text_input("Enter your query:")
    chat_history = [] 
    if st.button("Go"):
        if query:
            with st.spinner("Fetching..."):
                answer = answer_question(query)
                chat_history.append({"user": query, "model": answer})
            st.subheader("Answer:")
            st.write(answer)
    if chat_history:
        st.subheader("Chat History")
        for i, chat in enumerate(chat_history, 1):
            st.write(f"**Q{i}:** {chat['user']}")  
            st.write(f"**A{i}:** {chat['model']}") 


if __name__ == "__main__":
    main()
