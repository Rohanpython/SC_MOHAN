import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Function to load FAISS index from the local disk
def load_embeddings_offline(faiss_index_path="faiss_index"):
    try:
        # Load FAISS index from local disk
        vectorstore = FAISS.load_local(
            faiss_index_path, 
            HuggingFaceEmbeddings(
                model_name="./models/all-MiniLM-L6-v2",  # Load from the local directory
                model_kwargs={'device': 'cpu'}  # Set to use CPU
            ),
            allow_dangerous_deserialization=True
        )
        return vectorstore
    except Exception as e:
        raise Exception(f"Error loading FAISS index: {e}")

# Function to extract text from PDF files, with error handling for corrupted files
def get_pdf_text_with_metadata(docs):
    chunks = []
    metadata = []
    for pdf in docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:  # Only append text if it exists
                    chunks.append(page_text)
                    metadata.append({
                        "source": os.path.basename(pdf),
                        "page": page_num + 1
                    })
            print(f"Successfully processed: {pdf}")
        except Exception as e:
            print(f"Error reading {pdf}: {e}")
            # Skip this file and continue with the others
            continue
    
    # Check if we have extracted any text
    if not chunks:
        print("No text extracted from any PDFs.")
    
    return chunks, metadata

# Function to split text into chunks with metadata
def get_chunks_with_metadata(text_list, metadata):
    # Join the list of text into a single string
    if isinstance(text_list, list):
        raw_text = "\n".join(text_list)  # Join with newlines between pages
    
    if not raw_text:
        raise ValueError("Raw text is empty, cannot create chunks.")
    
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(raw_text)
    
    # Map metadata to each chunk
    chunked_metadata = []
    for i, chunk in enumerate(chunks):
        chunked_metadata.append(metadata[i % len(metadata)])  # Map the metadata

    return chunks, chunked_metadata

# Function to create and save FAISS embeddings
def save_embeddings_to_faiss(chunks, chunked_metadata, embedding_model_name, faiss_index_path):
    embeddings = HuggingFaceEmbeddings(
        model_name="./models/all-MiniLM-L6-v2",  # Load from local folder
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=chunked_metadata)
    # Save the FAISS index to a file
    vectorstore.save_local(faiss_index_path)
    print(f"Embeddings saved to {faiss_index_path}")

# Load PDFs from a folder, create chunks, and save embeddings
def process_pdfs_in_folder_and_save_embeddings(folder_path, faiss_index_path):
    # Get all PDF files in the specified folder
    pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return
    
    # Step 1: Extract text and metadata from the PDFs
    raw_text, metadata = get_pdf_text_with_metadata(pdf_files)
    
    if not raw_text:
        print(f"No text found in the PDFs within {folder_path}")
        return

    # Step 2: Convert text into chunks with metadata
    text_chunks, chunked_metadata = get_chunks_with_metadata(raw_text, metadata)
    
    if not text_chunks:
        print("No text chunks created.")
        return
    
    # Step 3: Create and save FAISS embeddings with metadata
    save_embeddings_to_faiss(text_chunks, chunked_metadata, "sentence-transformers/all-MiniLM-L6-v2", faiss_index_path)

    # Final completion message
    print("All PDFs have been processed and embeddings have been saved successfully!")

if __name__ == "__main__":
    # Define the folder containing the PDF files (relative path to the 'data' folder)
    folder_path = "data"  # The 'data' folder should be in the same directory as the script
    
    # Define where to save the FAISS index
    faiss_index_path = "faiss_index"
    
    # Process the PDFs in the folder and create embeddings
    process_pdfs_in_folder_and_save_embeddings(folder_path, faiss_index_path)
