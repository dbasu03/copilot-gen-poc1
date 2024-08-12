import streamlit as st
import pandas as pd
from pyfile import load_data, create_embeddings, build_faiss_index, load_faiss_index, search_index

# Constants
DATA_FILE_PATH = 'OrderedWorkflows.csv'
INDEX_FILE_PATH = 'faiss_index.index'

# Load data and create FAISS index
@st.cache
def initialize():
    df = load_data(DATA_FILE_PATH)
    embeddings = create_embeddings(df['Workflow'].tolist())
    index = build_faiss_index(embeddings)
    build_faiss_index(embeddings)
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
