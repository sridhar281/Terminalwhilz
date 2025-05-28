from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df= pd.read_csv("restaurant_reviews.csv")
embeddings=OllamaEmbeddings(model="llama3.2")
db_location="./chroma_db"
add_docs=not os.path.exists(db_location)
if add_docs:
    documents=[]
    ids=[]
    
    for i,row in df.iterrows():
        document=Document(
        page_content=row["title"]+""+row["Review"]
        metadata={"source":row["Restaurant"],"rating":row["Rating"],"date":row["Date"]},
        id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

vector_store=Chroma(
    collection_name="restaurant_reviews",
    embedding_function=embeddings,
    db_location=db_location,
)

if add_docs:
    vector_store.add_documents(documents=documents, ids=ids)
else:
    vector_store.load() 
    
retreival=vector_store.as_retriever(
    search_kwargs={"k":5},
)
