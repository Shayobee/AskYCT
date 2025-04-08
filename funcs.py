import os
import base64
import re
import time
import numpy as np
from spellchecker import SpellChecker
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import streamlit as st


# Load documents with error handling

loader = TextLoader("program.txt")
docs = loader.load()
documents = [doc.page_content for doc in docs]

fallback_message = """
The requested question not in the knowledge base. 
Please visit https://yabatech.edu.ng/ for more information or contact the school via these means: 
- phone: +234-703-7431-055
- email 1: registrar@yabatech.edu.ng
- email 2: records@yabatech.edu.ng"""


# Text splitter and embeddings setup

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600,
                                               chunk_overlap=200,
                                               length_function=len)
chunks = text_splitter.split_documents(docs)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-MiniLM-L6-v3")
vectorstore = FAISS.from_documents(chunks, embedding_model)

# st.error(f"Error initializing vector store or embeddings: {e}")

# Initialize API client
API_TOKEN = st.secrets["hugging_face"]
model_name = "Qwen/Qwen2.5-72B-Instruct"
try:
    client = InferenceClient(model=model_name, token=API_TOKEN)
except Exception as e:
    st.error(f"Error initializing Hugging Face InferenceClient: {e}")

# Spellchecker setup
spell = SpellChecker()
custom_terms = ["hnd", "ond", "bsc", "codfel", "yct", "yaba", "yabatech"]
spell.word_frequency.load_words(custom_terms)


# Error handling for greeting detection
def detect_greeting(query):
    try:
        greeting_keywords = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        for keyword in greeting_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, query.lower()):
                time.sleep(1)
                return "Hello there! How can I assist you today?"
        return None
    except Exception as e:
        st.error(f"Error detecting greeting: {e}")
        return None


# Main query function with error handling
def queries(query):
    try:
        greeting_response = detect_greeting(query)
        if greeting_response:
            return greeting_response

        retriever = vectorstore.as_retriever(search_type="similarity",
                                             search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(query)
        context = "\n".join([doc.page_content.lower() for doc in retrieved_docs])

        if not context.strip():
            return fallback_message

        query_embedding = embedding_model.embed_query(query)
        context_embeddings = [embedding_model.embed_query(doc.page_content) for doc in retrieved_docs]
        cosine_sim = cosine_similarity([query_embedding], context_embeddings)
        max_similarity = np.max(cosine_sim)

        if max_similarity < 0.2:
            return fallback_message

        prompt = f"""
   
        Context:
        {context}
              
        You are a helpful and advanced educational assistant tasked with answering questions based on the provided 
        context. Follow these rules:

        1. Answer Only the Question: Respond directly to the question adding extra commentary, elaboration, 
        or personal thoughts ONLY when needed
        2. Exclude the Context: DO NOT include any part of the context in your response. Your answer should be 
        standalone and self-contained.
        3. Use Proper Formatting:
           - Use bullet points for lists where appropriate.
           - Give your Response to users only in ENGLISH.
        4. Friendly Yet Professional Tone: Maintain a polite and professional tone without being overly formal or 
        robotic. Avoid verbosity and stick to the facts.
        5. Follow Formatting Instructions: If the question specifies a format (e.g., sentence, list, etc.), adhere 
        to it precisely.
        6. No Meta-Comments: Do not include notes, disclaimers, or explanations about following the rules. 
        Simply provide the answer.
        7. Focus on Requirements: When answering questions, prioritize O'level subjects, UTME subjects, and cutoff 
        marks as they relate to OND admissions.


    
        Question: {query}
    """
        response = client.text_generation(prompt,
                                          max_new_tokens=200,
                                          temperature=0.2,
                                          repetition_penalty=1.5,
                                          top_p=0.9,)
        cleaned_response = response.split(":", 1)[-1].strip() if ":" in response else response.strip()
        print(cleaned_response)
        print(context)

        return cleaned_response.split('---', 1)[0]
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return fallback_message


def open_picture(image_name):
    try:
        cwd = os.path.dirname(__file__)
        image_path = os.path.join(cwd, "images", image_name)
        image_path = os.path.abspath(image_path)
        with open(image_path, "rb") as file:
            images = base64.b64encode(file.read()).decode()
        return images
    except FileNotFoundError:
        st.error(f"Image file {image_name} not found.")
    except Exception as e:
        st.error(f"Error loading image {image_name}: {e}")


def preprocess_input(user_input):
    try:
        user_input = user_input.lower()
        user_input = re.sub(r'\bthe school\b(?!\s+of)', 'YABATECH', user_input, flags=re.IGNORECASE)
        user_input = re.sub(r'\byct\b', 'YABATECH', user_input, flags=re.IGNORECASE)

        words = user_input.split()
        corrected_words = []
        for word in words:
            corrected_word = spell.correction(word) or word
            corrected_words.append(corrected_word)
        return " ".join(corrected_words)
    except Exception as e:
        st.error(f"Error preprocessing input: {e}")
        return user_input
