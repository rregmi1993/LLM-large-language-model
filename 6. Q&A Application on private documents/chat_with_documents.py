import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader


def load_documents(file):
  name, extension = os.path.splitext(file)

  if extension == '.pdf':
    print(f'loading a file .......{file}')
    loader = PyPDFLoader(file)

  elif extension == '.docx':
    print(f'loading a file .......{file}')
    loader = Docx2txtLoader(file)

  elif extension == '.txt':
    print(f'loading a file .......{file}')
    loader = TextLoader(file)

  else:
    print(f"give file {file} format is not supported by application")
    return None

  data = loader.load()
  print("completed ....")
  return data

#chunking the data
from langchain.text_splitter import RecursiveCharacterTextSplitter
def chunk_data(data, chunk_size=256, chunk_overlap = 30):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap =chunk_overlap)
  chunks = text_splitter.split_documents(data)
  return chunks

#create embedding
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
def ask_question_for_ans(vector_store, qns, k=3):
  llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
  retriver = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
  chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriver)
  answer = chain.run(qns)
  return answer

#check the cost of each embedding
import tiktoken
def calculate_embedding_cost(texts):
  enc = tiktoken.encoding_for_model('text-embedding-ada-002')
  total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
  #print(f"total tokens: {total_tokens}")
  #print(f"Embedding cost in USD: {total_tokens / 1000 * 0.004:.6f}")
  return total_tokens, total_tokens / 1000 * 0.004


if __name__=='__main__':
  load_dotenv(find_dotenv(), override=True)
  st.image('langchain_flow_llm.png')
  st.subheader(':::LLM Question Answering Application:::')

  with st.sidebar:
    api_key = st.text_input("Open API Key: ", type='password')
    if api_key:
      os.environ['OPEN_API_KEY'] = api_key

    upload_file = st.file_uploader('upload a file: ', type=['pdf', 'docx', 'txt'])
    chunk_size = st.number_input('chunk size:', min_value=100, max_value=2048, value=512)
    k = st.number_input('k', min_value=1, max_value=30, value=3)
    add_data = st.button('Add Data')


    if upload_file and add_data:
      with st.spinner('Reading chunking and embedding file....'):
        bytes_data = upload_file.read()
        file_name=os.path.join('./', upload_file.name)
        with open(file_name, 'wb') as f:
          f.write(bytes_data)
        
        data = load_documents(file_name)

        chunks = chunk_data(data, chunk_size=chunk_size)
        st.write(f'chunk size: {chunk_size} and chunks: {len(chunks)}')

        tokens, embedding_cost = calculate_embedding_cost(chunks)
        st.write(f'Embedding cost: ${embedding_cost} and tokens: {tokens}')

        vectore_store = create_embeddings(chunks)

        st.session_state.vs = vectore_store
        st.success("file uploaded, chunked and embedded sucessfully")

  qns = st.text_input("Ask question about the content of the uploaded file: ")
  if qns:
    if 'vs' in st.session_state:
      vectore_store = st.session_state.vs
      st.write(f'k : {k}')
      answer = ask_question_for_ans(vectore_store, qns, k)
      st.text_area('LLM answer : ', value=answer)


      
