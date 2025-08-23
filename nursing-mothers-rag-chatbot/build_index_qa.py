import os
import json
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

chunk_folder = "/workspaces/nursing-mothers-rag-chatbot/index/data/chunks"
index_folder = "/workspaces/nursing-mothers-rag-chatbot/index/embeddings"
'''os.makedirs(index_folder, exist_ok=True)'''

all_chunks = []
for filename in ["nhs_qa_strings.json", "nwh_qa_strings.json"]:
    filepath = os.path.join(chunk_folder, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue
    with open(filepath, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        all_chunks.extend(chunks)

print(f"Total Q&A chunks to embed: {len(all_chunks)}")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(
    all_chunks,
    embedding_model,
)
vectorstore.save_local(index_folder, index_name="breastfeeding_index")

print(f"FAISS vectorstore saved to {index_folder} (includes .faiss and .pkl files)")
