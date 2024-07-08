import streamlit as st
import openai
import numpy as np
import torch
import requests
import urllib3
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

st.title("Text Similarity Checker")

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set your OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Monkey-patch requests to disable SSL verification for OpenAI
old_request = requests.Session.request

def new_request(*args, **kwargs):
    kwargs['verify'] = False  # Disable SSL verification
    return old_request(*args, **kwargs)

requests.Session.request = new_request

text1 = st.text_area("Enter first text:")
text2 = st.text_area("Enter second text:")

if st.button("Calculate"):
    try:
        # Function to get embeddings from OpenAI
        def get_embedding(text):
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"  # Change to the desired model
            )
            return np.array(response['data'][0]['embedding'])

        # Calculate embeddings
        embedding1 = get_embedding(text1)
        embedding2 = get_embedding(text2)

        # Calculate cosine similarity
        def cosine_similarity(vec1, vec2):
            vec1 = torch.tensor(vec1)
            vec2 = torch.tensor(vec2)
            dot_product = torch.dot(vec1, vec2).item()
            magnitude = torch.norm(vec1) * torch.norm(vec2)
            return dot_product / magnitude if magnitude != 0 else 0

        similarity = cosine_similarity(embedding1, embedding2)

        # Define your threshold
        SIMILARITY_THRESHOLD = 0.8

        if similarity > SIMILARITY_THRESHOLD:
            st.write("true")
        else:
            st.write("false")

    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API request failed: {e}")

    except ValueError:
        st.error("Invalid response from OpenAI API")
