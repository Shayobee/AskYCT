import streamlit as st
from funcs import open_picture

st.set_page_config(page_title="Chat YCT",
                   page_icon="🤖",
                   layout="wide")

st.title("Chat YCT")

st.markdown(f"""
<img style="border: 2px solid powderblue" src="data:image/jpeg;base64,{open_picture("chat.webp")}" width="80%"><br>""",
            unsafe_allow_html=True)


st.text("Required for the fulfillment of the Higher National Diploma (HND) certification")


st.markdown("""
## Overview

This project is an **Admission Chatbot** designed to assist prospective students with the admission process for various
programs, schools, and departments. It leverages **state-of-the-art retrieval and generation techniques** to provide 
fast, accurate, and user-friendly answers.

## Project Goals

The main objectives of this chatbot are:
- To simplify the admission process by providing instant, accurate responses to common inquiries.
- To retrieve program-specific information such as O-Level requirements, program types, and school details.
- To deliver a seamless user experience via a conversational interface.

## Key Technologies

### FAISS (Facebook AI Similarity Search)  
[FAISS](https://faiss.ai/) is used for building a highly efficient **indexing and retrieval system**. It ensures that 
the chatbot can fetch relevant information quickly from a large dataset.

### Retrieval-Augmented Generation (RAG)  
[RAG](https://ai.facebook.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-assistants/) 
combines retrieval with natural language generation. It allows the chatbot to craft responses that incorporate relevant,
fact-based information retrieved from its knowledge base.

### Streamlit  
[Streamlit](https://streamlit.io/) powers the chatbot's user interface, making it interactive and user-friendly. 
It supports features like question submission, response display, and even future enhancements like analytics.

### Hugging Face Transformers  
[Hugging Face](https://huggingface.co/) provides the transformer models that power the chatbot’s natural language 
understanding and generation, ensuring high-quality conversational abilities.

## Project Workflow

1. **User Query Submission**: Users type a question into the chatbot interface.
2. **Retrieval Process**:
   - FAISS searches for the most relevant information in the knowledge base.
   - Relevant data is retrieved to form the basis of the response.
3. **Response Generation**:
   - The RAG model generates a comprehensive answer using the retrieved data.
4. **Answer Display**: The chatbot presents the answer to the user in a conversational format.

## Implementation Details

### Knowledge Base Design  
The chatbot's knowledge base is structured in **Text format**, which makes it easy to expand and update with new data 
about programs, requirements, and schools.

### Indexing with FAISS  
FAISS indexes the Text data, enabling rapid retrieval of relevant entries when users ask questions.

### Response Generation with RAG  
RAG combines retrieved data with generative capabilities to produce answers that are both factually correct and 
conversational.

### User Interface with Streamlit  
The chatbot interface, built using Streamlit, is simple and intuitive. Users can easily type questions, view answers, 
and navigate the application.

---

This chatbot brings together advanced technologies and thoughtful design to make your admission process easier than ever

""", unsafe_allow_html=True)
