import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import TextStreamer,pipeline,AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model_name1 = "sentence-transformers/all-MiniLM-L6-v2"

model_kwargs = {"device": device}

embeddings = HuggingFaceEmbeddings(model_name=model_name1, model_kwargs=model_kwargs)

vector_store = FAISS.load_local("faiss_index", 
                                embeddings,
                                allow_dangerous_deserialization=True)


streamer = TextStreamer(tokenizer,
                        skip_prompt = True,
                        skip_special_tokens = True)

pipe =  pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=256,
        temperature=0.001,
        top_p=0.95,
        repetition_penalty=1.15
    )

llm = HuggingFacePipeline(pipeline = pipe)


prompt_template = """
                Use following piece of context to answer the question in less than 30 words.

                Context : {context}

                Question : {question}

                Answer : """


PROMPT = PromptTemplate(
    template = prompt_template, 
    input_variables = ["context", "question"]
)


retriever = vector_store.as_retriever(search_kwargs = {"k": 5})

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff", # map_reduce, map_rerank, stuff, refine
    retriever = retriever, 
    chain_type_kwargs = {"prompt": PROMPT},
    return_source_documents = True,
    verbose = False
)


def predict(question):
    result = qa_chain(question)
    return result['result']