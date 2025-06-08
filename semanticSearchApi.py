import os
import hashlib
import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, documents_folder, db_folder=None):
        self.documents_folder = documents_folder
        self.db_folder = db_folder or os.path.join(documents_folder, "chroma_db")
        self.hash_file = os.path.join(documents_folder, "documents.hash")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._ensure_db()

    def _compute_folder_hash(self):
        hash_md5 = hashlib.md5()
        for filename in sorted(os.listdir(self.documents_folder)):
            filepath = os.path.join(self.documents_folder, filename)
            if os.path.isfile(filepath):
                hash_md5.update(filename.encode())
                with open(filepath, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _ensure_db(self):
        current_hash = self._compute_folder_hash()
        previous_hash = None
        if os.path.exists(self.hash_file):
            with open(self.hash_file, "r") as f:
                previous_hash = f.read().strip()
        if current_hash != previous_hash and os.path.exists(self.db_folder):
            import shutil
            shutil.rmtree(self.db_folder)
        with open(self.hash_file, "w") as f:
            f.write(current_hash)
        self._load_documents()
        self._init_chroma()

    def _load_documents(self):
        self.documents = []
        self.document_names = []
        for filename in os.listdir(self.documents_folder):
            filepath = os.path.join(self.documents_folder, filename)
            if os.path.isfile(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    self.documents.append(f.read())
                self.document_names.append(os.path.splitext(filename)[0])

    def _init_chroma(self):
        self.client = PersistentClient(path=self.db_folder)
        self.collection = self.client.get_or_create_collection(name="my_documents")
        ids = [f"doc_{i}" for i in range(len(self.documents))]
        embeddings = self.embedding_model.encode(self.documents).tolist()
        metadatas = [{"name": self.document_names[i]} for i in range(len(self.documents))]
        self.collection.add(
            documents=self.documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

    def query(self, query_text, n_results=3):
        query_embedding = self.embedding_model.encode([query_text]).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['metadatas', 'documents', 'distances']
        )
        # Return as a list of dicts
        return [
            {
                "name": metadata["name"],
                "text": document,
                "distance": distance
            }
            for metadata, document, distance in zip(
                results['metadatas'][0], results['documents'][0], results['distances'][0]
            )
        ]