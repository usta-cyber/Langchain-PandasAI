"""
Building a Document-based Question Answering System with 
LangChain, Pinecone, and LLMs like GPT-4 and ChatGPT



"""

import os
import openai
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

directory = 'testDir'
pinecone.init(
    api_key="83af22f9-ec60-484a-8eba-3519c69b251f",
    environment="eu-west4-gcp"
)

index_name = "test-tangr2"



def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

def get_similiar_docs(query, k=2, score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query, k=k)
  else:
    similar_docs = index.similarity_search(query, k=k)
  return similar_docs


documents = load_docs(directory)
print(len(documents))

docs = split_docs(documents)
print(len(docs))

embeddings = OpenAIEmbeddings(model_name="ada",openai_api_key='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')

query_result = embeddings.embed_query("Hello world")
print(len(query_result))
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)


model_name = "text-davinci-003"
# model_name = "gpt-3.5-turbo"
#model_name = "gpt-4"
llm = OpenAI(model_name=model_name,openai_api_key='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')
chain = load_qa_chain(llm, chain_type="stuff")

def get_answer(query):
  similar_docs = get_similiar_docs(query)
  answer = chain.run(input_documents=similar_docs, question=query)
  return answer




query = "give me SNL Finance"
answer = get_answer(query)
print(answer)

query = "How have relations between finance and the year improved?"
answer = get_answer(query)
print(answer)
