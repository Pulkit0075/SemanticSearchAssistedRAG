import os
import hashlib
import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

def compute_folder_hash(folder):
    """Compute a hash of all files in the folder (names + contents)."""
    hash_md5 = hashlib.md5()
    for filename in sorted(os.listdir(folder)):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            hash_md5.update(filename.encode())
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Folder containing your documents
documents_folder = os.path.join(os.path.dirname(__file__), "documents")
hash_file = os.path.join(os.path.dirname(__file__), "documents.hash")
db_folder = os.path.join(os.path.dirname(__file__), "chroma_db")

# Check for changes in the documents folder
current_hash = compute_folder_hash(documents_folder)
previous_hash = None
if os.path.exists(hash_file):
    with open(hash_file, "r") as f:
        previous_hash = f.read().strip()

# If changed, delete the old DB
if current_hash != previous_hash and os.path.exists(db_folder):
    import shutil
    shutil.rmtree(db_folder)

# Save the new hash
with open(hash_file, "w") as f:
    f.write(current_hash)

# Load all documents from the folder
documents = []
document_names = []
for filename in os.listdir(documents_folder):
    filepath = os.path.join(documents_folder, filename)
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            documents.append(f.read())
        document_names.append(os.path.splitext(filename)[0])  # filename without extension

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  

# Initialize Chroma client with local persistence
client = PersistentClient(path=db_folder)

# Create (or get) the collection
collection = client.get_or_create_collection(name="my_documents")

# Prepare embeddings + metadata
ids = [f"doc_{i}" for i in range(len(documents))]
embeddings = embedding_model.encode(documents).tolist()
metadatas = [{"name": document_names[i]} for i in range(len(documents))]

# Add documents (skip if they are already stored to avoid duplication)
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids,
    metadatas=metadatas
)

# Read the query from query.text
query_file = os.path.join(os.path.dirname(__file__), "query.text")
with open(query_file, "r", encoding="utf-8") as f:
    query = f.read().strip()

query_embedding = embedding_model.encode([query]).tolist()[0]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=['metadatas', 'documents', 'distances']
)

# Print results: name + text
print(f"\n🔎 Query: {query}\n")
for metadata, document, distance in zip(results['metadatas'][0], results['documents'][0], results['distances'][0]):
    print(f"📄 Document Name: {metadata['name']}")
    print(f"📝 Text: {document}")
    print(f"📊 Distance: {distance:.4f}\n")