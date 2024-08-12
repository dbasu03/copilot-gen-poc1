import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def create_embeddings(text_list):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_list)
    return embeddings

def build_faiss_index(embeddings):
    embeddings_np = np.array(embeddings)
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    return index

def save_faiss_index(index, index_file_path):
    faiss.write_index(index, index_file_path)

def load_faiss_index(index_file_path):
    return faiss.read_index(index_file_path)

def search_index(index, input_embedding, k):
    distances, indices = index.search(np.array([input_embedding]), k)
    return distances, indices
