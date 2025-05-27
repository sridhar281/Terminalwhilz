from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model=OllamaLLM(model="llama3.2")
template= """You are an expert in answering questions about the restaurant
here are some relevant reviews:{reviews}
here is the question for you:{question}
"""
prompt= ChatPromptTemplate.from_template(template)
chain= prompt | model 

while True:
    print("\n-------------------------------------------------")
    question=input("Provide your query(q to quit):")
    if question=="q":
        break 
    
    result=chain.invoke({"reviews":[],"question":question})
    print(result)
