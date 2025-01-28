import streamlit as st

from datasets import load_dataset

# Replace with a valid dataset name and configuration
dataset = load_dataset("oscar", "unshuffled_deduplicated_te", split="train",trust_remote_code=True)
telugu_texts = [example["text"] for example in dataset]


from transformers import AutoTokenizer
from tqdm import tqdm  # For progress tracking

# Load a Telugu tokenizer
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("l3cube-pune/telugu-bert")
model = AutoModelForMaskedLM.from_pretrained("l3cube-pune/telugu-bert")
# Function to split text into chunks
def split_into_chunks(text, max_tokens=128):
    try:
        # Tokenize the text without truncation
        tokens = tokenizer.encode(text, truncation=False, return_tensors="pt")[0]
        # Split tokens into chunks
        chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
        # Decode chunks back to text
        return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
    except Exception as e:
        print(f"Error processing text: {text[:50]}... - {str(e)}")
        return []  # Return empty list if there's an error



# Split all Telugu texts into chunks with progress tracking
telugu_chunks = []
for text in tqdm(telugu_texts, desc="Processing Telugu texts"):
    chunks = split_into_chunks(text)
    telugu_chunks.extend(chunks)

# Save the chunks to a text file
with open("telugu_chunks.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(telugu_chunks))

print("Processing complete! Chunks saved to 'telugu_chunks.txt'.")

from sentence_transformers import SentenceTransformer
# Load a multilingual embedding model
embedding_model = SentenceTransformer ("paraphrase-multilingual-mpnet-base-v2" )
# Generate embeddings for Telugu chunks
chunk_embeddings = embedding_model.encode(telugu_chunks, show_progress_bar=True)

import faiss
import numpy as np
# Convert embeddings to a numpy array
embeddings_array = np.array(chunk_embeddings).astype("float32" )
# Create a FAISS index
dimension = embeddings_array.shape[1]
index = faiss.IndexFlatL2(dimension) # L2 distance for similarity search
index.add(embeddings_array)

api_key=st.secrets['groq']['api_key']

from groq import Groq

# Replace this with a valid API key
client = Groq(api_key=api_key)

def rag_system(query: str, top_k: int = 50) -> str:
  # Step 1: Convert query to embedding
  query_embedding = embedding_model.encode([query], show_progress_bar=False)
  query_embedding = np.array(query_embedding).astype("float32")
  # Step 2: Retrieve top-k matching chunks
  distances, indices = index.search(query_embedding, top_k)
  retrieved_chunks = [telugu_chunks[idx] for idx in indices[0]]
  # Step 3: Combine retrieved chunks into a single context
  context = " ".join(retrieved_chunks)
  # Step 4: Generate a response using Groq's API
  response = query_with_llm(query, context)
  return response

# Function to query the Groq API
def query_with_llm(query: str, context: str) -> str:
    prompt = f"ప్రశ్న: {query}\nసందర్భం: {context}"

    # Query the Groq API
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Ensure the model is accessible with your API key
        messages=[
            {"role": "system", "content": "You are a helpful assistant in Telugu language."},
            {"role": "user", "content": prompt}
        ]
    )

    # Return the generated response
    return response.choices[0].message.content.strip()

