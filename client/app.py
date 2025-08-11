import boto3
import streamlit as st
import os
import uuid
import shutil
#BEDROCK
#from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

##text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
## pdf loader
from langchain_community.document_loaders import PyPDFLoader
## FAISS
from langchain_community.vectorstores import FAISS




# Set up S3 client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME", "culinary-project101-bucket")



bedrock_client = boto3.client(service_name="bedrock-runtime",region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v2:0")

# Print Bedrock embeddings instance
print(bedrock_embeddings)

folder_path = f"/tmp/"

def get_unique_request_id():
    return str(uuid.uuid4())

  #split text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    return text_splitter.split_documents(pages)


import os
import shutil
import boto3
from langchain_community.vectorstores import FAISS

# Assuming bedrock_embeddings and BUCKET_NAME are already set
s3_client = boto3.client("s3")

def create_vector_store(request_id, documents):
    try:
        # 1. Create FAISS vector store in memory
        vector_store = FAISS.from_documents(documents, bedrock_embeddings)
        print(f"Vector store created in memory for request_id={request_id}")

        # 2. Save locally in a folder named after the request_id
        folder_path = f"/tmp/{request_id}"
        vector_store.save_local(folder_path)
        print(f"Vector store saved locally at {folder_path}")

        # 3. Upload each file in the folder separately to S3
        for filename in os.listdir(folder_path):
            local_path = os.path.join(folder_path, filename)
            s3_key = f"vector_stores/{request_id}/{filename}"  # Using a folder per request_id in S3

            s3_client.upload_file(local_path, BUCKET_NAME, s3_key)
            print(f"Uploaded {filename} to s3://{BUCKET_NAME}/{s3_key}")

        return True

    except Exception as e:
        print(f"Error creating/uploading vector store: {e}")
        return False
#load index
def load_index():
    s3_client.download_file(BUCKET_NAME, Key="vector_stores/c0185535-4a46-4801-9477-986db1cb8dac/index.faiss", Filename="/tmp/index.faiss")
    s3_client.download_file(BUCKET_NAME, Key="vector_stores/c0185535-4a46-4801-9477-986db1cb8dac/index.pkl", Filename="/tmp/index.pkl")

def get_llm():
    llm = Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_client,model_kwargs={"temperature": 0.1})
    return llm

#get response
def get_response(llm,vectorstore, question ):
    ## create prompt / template
    prompt_template = """

    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":question})
    return answer['result']

# Main function to handle PDF uploads and processing
def main():
    st.header("Culinary Project 101 Query Hub")
    load_index()
    dir_list=os.listdir(folder_path)
    #st.write(f"Available file and directories: {folder_path}")
    #st.write(dir_list)
   # Add clickable Instagram link
   
    if st.button("Go to The Culinary Project 101"):
        import webbrowser
        webbrowser.open_new_tab("https://www.instagram.com/theculinaryproject101/?hl=en")

    st.markdown(
    "Check out our Instagram page for available recipes: "
    "[The Culinary Project 101](https://www.instagram.com/theculinaryproject101/?hl=en)"
)



    #create index
    faiss_index = FAISS.load_local(index_name="index",folder_path=folder_path, embeddings=bedrock_embeddings,allow_dangerous_deserialization=True)
    st.write("FAISS index loaded successfully.")
    question = st.text_input("Enter your question:")
    if st.button("Ask Question"):
        with st.spinner("Getting response..."):
            llm=get_llm()
            response = get_response(llm, faiss_index, question)
            st.write("Response:")
            st.write(response)
            st.success("Response retrieved successfully.")

if __name__ == "__main__":
    main()