import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer,util
import faiss
import numpy as np

# Constants
DATA_FILE_PATH = 'OrderedWorkflows.csv'
INDEX_FILE_PATH = 'faiss_index.index'

# Functions from pyfile.py

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

# Streamlit app code

@st.cache
def initialize():
    df = load_data(DATA_FILE_PATH)
    embeddings = create_embeddings(df['Workflow'].tolist())
    index = build_faiss_index(embeddings)
    save_faiss_index(index, INDEX_FILE_PATH)
    return df, index

df, index = initialize()

st.title('Workflow Similarity Search')

user_input = st.text_input("Enter your query:")

if user_input:
    # Create embedding for user input
    input_embedding = create_embeddings([user_input])[0]

    # Perform similarity search
    k = 10
    distances, indices = search_index(index, input_embedding, k)

    # Display results
    st.write("Matching Workflows:")
    matching_workflows = df.iloc[indices[0]]['Workflow'].tolist()
    for workflow in matching_workflows:
        st.write(workflow)
