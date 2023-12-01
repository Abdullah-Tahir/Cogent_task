import os
os.environ["OPENAI_API_KEY"] = "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"

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

from langchain.chains.summarize import load_summarize_chain

BULLET_POINT_PROMPT = PromptTemplate(template="• {text}")

def summarize_book(documents, map_prompt=BULLET_POINT_PROMPT, combine_prompt=BULLET_POINT_PROMPT, return_intermediate_steps=False, input_variables=None):
    # Load the list of documents from the directory
    pdf_loader = DirectoryLoader("C:/Users/abdul/Downloads/book", glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = pdf_loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0, length_function=len)
    docs = text_splitter.split_documents(documents)

    # Create and run the summarization chain
    chain = load_summarize_chain(llm,
                                 chain_type="map_reduce",
                                 map_prompt=map_prompt,
                                 combine_prompt=combine_prompt,
                                 return_intermediate_steps=return_intermediate_steps)

    # Pass input variables if provided
    if input_variables:
        output_summary = chain(docs, input_variables=input_variables)
    else:
        output_summary = chain(docs)

    # Process the output
    if return_intermediate_steps:
        # Access and analyze individual summaries for each document
        intermediate_summaries = output_summary["intermediate_summaries"]
        # ...

        # Combine the individual summaries into a final text
        final_summary = output_summary["output_text"]
    else:
        final_summary = output_summary

    # Wrap the text for better readability
    wrapped_text = textwrap.fill(final_summary,
                                 width=100,
                                 break_long_words=False,
                                 replace_whitespace=False)

    # Print the final summary
    print(wrapped_text)

# Usage example with default prompt
summarize_book(documents)

# Usage example with custom prompt and input variables
custom_prompt_template = PromptTemplate(template="Summarize the following text {text} into a concise narrative, focusing on key plot points, character motivations, and historical context:")
CUSTOM_PROMPT = PromptTemplate(template=custom_prompt_template, input_variables=["text"])
input_variables = {"text": "The text to be summarized"}
summarize_book(documents, map_prompt=CUSTOM_PROMPT, combine_prompt=CUSTOM_PROMPT, input_variables=input_variables)