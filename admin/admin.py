import boto3
import streamlit as st
import os
import uuid
import shutil
#BEDROCK
#from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import BedrockEmbeddings
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


# Main function to handle PDF uploads and processing
def main():
    st.write("This is admin site for pdf demo")
    upload_file = st.file_uploader("Upload PDF", type=["pdf"])

    if upload_file is not None:

        request_id = get_unique_request_id()
        st.write(f"Request Id : {request_id}.")
        saved_file_name= f"{request_id}.pdf"
        with open(saved_file_name, "wb") as w:

            w.write(upload_file.getvalue())

        loader=PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()
        st.write(f"Total pages in pdf: {len(pages)}")
    
  

        splitted_docs = split_text(pages,1000,200)
        st.write(f"Total chunks created: {len(splitted_docs)}")
        st.write("===========================")
        st.write(splitted_docs[0])
        st.write("===========================")
        st.write(splitted_docs[1])

        st.write("Creating vector store")
        result=create_vector_store(request_id,splitted_docs)

        if result:
            st.write("Vector store created and uploaded to S3 successfully.")
        else:
            st.write("Failed to create vector store. Please check logs.")

if __name__ == "__main__":
    main()