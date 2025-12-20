from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv


load_dotenv()

openai_client = OpenAI()

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)
vector_db = QdrantVectorStore.from_existing_collection(
    url= "http://localhost:6333",
    collection_name = "learning_rag",
    embedding = embedding_model 
)



def process_query(query:str):
    print("Searching Chunks", query)
    search_results = vector_db.similarity_search(query=query)
    
    
    context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number:{result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
                         for result in search_results])

    SYSTEM_PROMPT = f"""
You are a helpful AI Assistant who answers user query based on the available
context retrieved from a PDF file along with page_contents and page number.

You should only answer the user bases on following context and navigate the user to open the 
right page number to know more

Context :
{context}
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content":SYSTEM_PROMPT},
            {"role": "system", "content":query},
        ]
    )
    print(f"Response {response.choices[0].message.content}")
    return response.choices[0].message.content


# export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
# rq worker --with-scheduler   