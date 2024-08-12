import pandas as pd

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  

file_path = '/content/OrderedWorkflows.csv'  
df = pd.read_csv(file_path)

embeddings = model.encode(df['Workflow'].tolist())


import faiss
import numpy as np

embeddings_np = np.array(embeddings)

dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings_np)

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

df = pd.read_csv('OrderedWorkflows.csv')

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(df['Workflow'].tolist())

embeddings_np = np.array(embeddings)

dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings_np)

faiss.write_index(index, 'faiss_index.index')

index = faiss.read_index('faiss_index.index')

user_input = input("Enter your text: ")

input_embedding = model.encode([user_input])[0]

k = 10  
distances, indices = index.search(np.array([input_embedding]), k)

matching_workflows = df.iloc[indices[0]]['Workflow'].tolist()

print("Matching Workflows:")
for workflow in matching_workflows:
    print(workflow)
