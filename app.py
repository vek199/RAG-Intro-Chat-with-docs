import os
from dotenv import load_dotenv
import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=gemini_api_key)

# Define a Gemini embedding function
class GeminiEmbeddingFunction:
    def __call__(self, texts):
        embeddings = []
        for text in texts:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            if "embedding" in response:
                embeddings.append(response["embedding"])
            else:
                raise ValueError(f"Unexpected response structure: {response}")
        return embeddings

# Initialize Gemini embedding function
gemini_ef = GeminiEmbeddingFunction()

# Initialize ChromaDB with Gemini embeddings
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=gemini_ef
)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-pro")


# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# Load documents from the directory
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)

print(f"Loaded {len(documents)} documents")
# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})


# Function to generate embeddings using Gemini API
def get_gemini_embedding(text):
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    if "embedding" in response:
        embedding = response["embedding"]
        print("==== Generating embeddings... ====")
        return embedding
    else:
        raise ValueError(f"Unexpected response format: {response}")


# Generate embeddings for the document chunks
for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_gemini_embedding(doc["text"])


# Upsert documents with embeddings into Chroma
for doc in chunked_documents:
    print("==== Inserting chunks into db ====")
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
    )


# Function to query documents
def query_documents(question, n_results=2):
    results = collection.query(query_texts=question, n_results=n_results)

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks


# Function to generate a response using Gemini AI
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an AI assistant for question-answering tasks. Use the following retrieved context "
        "to answer the question concisely. If you don't know the answer, say 'I don't know.' "
        "Limit your response to three sentences.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )

    response = model.generate_content(prompt)
    return response.text


# Example query and response generation
question = "tell me about Databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)
