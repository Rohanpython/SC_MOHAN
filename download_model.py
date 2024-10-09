from sentence_transformers import SentenceTransformer

# Download the model and save it locally
def download_and_save_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model.save('./models/all-MiniLM-L6-v2')  # Save the model locally
    print("Model downloaded and saved locally at './models/all-MiniLM-L6-v2'")

if __name__ == "__main__":
    download_and_save_model()
