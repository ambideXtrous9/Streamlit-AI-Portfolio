import streamlit as st
import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer

# ---------------------------
# Setup Cache Directories
# ---------------------------
HF_CACHE_DIR = "./hf_cache"  # Hugging Face model cache
FAISS_INDEX_PATH = "faiss_index"  # FAISS storage
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# ---------------------------
# Cached Model Loading using Streamlit
# ---------------------------
@st.cache_resource
def load_model():
    MODEL_NAME = "EleutherAI/gpt-neo-125M"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)

    # Optimize model execution
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return tokenizer, model

tokenizer, model = load_model()
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ---------------------------
# Cached FAISS Vector Store Loading
# ---------------------------
@st.cache_resource
def load_faiss_index():
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", 
                                      device=device, 
                                      cache_folder=HF_CACHE_DIR)
    
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    return vector_store

vector_store = load_faiss_index()
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# ---------------------------
# Setup LLM Pipeline
# ---------------------------
streamer = TextStreamer(
    tokenizer, skip_prompt=True, skip_special_tokens=True
)

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=256,
    temperature=0.001,
    top_p=0.95,
    repetition_penalty=1.15
)

llm = HuggingFacePipeline(pipeline=pipe)

# ---------------------------
# Prompt Template
# ---------------------------
prompt_template = """
                Use the following piece of context to answer the question in less than 30 words.

                Context : {context}

                Question : {question}

                Answer : """

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  
    retriever=retriever, 
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True,
    verbose=False
)

# ---------------------------
# Build Retrieval QA Chain
# ---------------------------
def predict(question):
    result = qa_chain(question)
    return result['result']
