from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model(
    model="gpt-4.1-mini",
    model_provider="openai"
)
class State(TypedDict):
    messages: Annotated[list, add_messages]
    
def chatbot(state: State):
    response = llm.invoke(state.get("messages"))
    return {"messages": [response]}


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

def compile_graph_checkpointer(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)

DB_URI = "mongodb://admin:admin@localhost:27017"
with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
    graph_with_checkpointer = compile_graph_checkpointer(checkpointer=checkpointer)
    config = {
        "configurable":{
        "thread_id": "Raj" #Always prefer to give username as thread id so that system can recognize you from your name.
        }
    }
    for chunk in graph_with_checkpointer.stream(
    State({"messages": ["What i am learning?"]}),
    config,
    stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()