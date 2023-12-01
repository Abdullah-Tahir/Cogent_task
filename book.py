
import os
os.environ["OPENAI_API_KEY"] = " "

import tiktoken
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage

from dotenv import load_dotenv, find_dotenv         

import openai
import os  
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0)


from langchain.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
pdf_loader = DirectoryLoader("C:/Users/abdul/Downloads/", glob="**/*.pdf", loader_cls=PyPDFLoader)
#text_loader = DirectoryLoader("C:/Users/abdul/Downloads/dyaya/", glob="**/*.txt", loader_cls=TextLoader)
#word_loader = DirectoryLoader("C:/Users/abdul/Downloads/dyaya/", glob="./*.docx", loader_cls=UnstructuredWordDocumentLoader).load()



# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(

    chunk_size=2000,
    chunk_overlap=0,
    length_function=len)
docs = text_splitter.split_documents(pdf_loader)


from langchain.chains.summarize import load_summarize_chain
import textwrap
chain = load_summarize_chain(llm, 
                             chain_type="map_reduce")


output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100)
print(wrapped_text)


# for summarizing each part
chain.llm_chain.prompt.template

# for combining the parts
chain.combine_document_chain.llm_chain.prompt.template
chain = load_summarize_chain(llm, 
                             chain_type="map_reduce",
                             verbose=True
                             )


output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, 
                             width=100,
                             break_long_words=False,
                             replace_whitespace=False)
print(wrapped_text)
