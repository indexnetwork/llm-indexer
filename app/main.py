import os
import json

from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from llama_index.vector_stores.types import VectorStoreInfo, MetadataInfo


from llama_index import download_loader
from llama_index.llms import ChatMessage
from llama_index.vector_stores import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import VectorStoreIndex, ListIndex, LLMPredictor, ServiceContext
from llama_index.chat_engine.types import ChatMode
from llama_index.indices.composability import ComposableGraph
from llama_index.indices.query.query_transform import HyDEQueryTransform
from llama_index.query_engine.transform_query_engine import TransformQueryEngine
from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters

from langchain.chat_models import ChatOpenAI

import chromadb
from chromadb.config import Settings

import openai
import redis

load_dotenv()

redisClient = redis.Redis.from_url(os.environ["REDIS_CONNECTION_STRING"]);


origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.environ["OPENAI_API_KEY"]

local_directory = "chroma-indexes"
persist_directory = os.path.join(os.getcwd(), local_directory)

chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=persist_directory
))

embed_model = OpenAIEmbedding(model="text-embedding-ada-002", openai_api_key=os.environ["OPENAI_API_KEY"])
llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=os.environ["OPENAI_API_KEY"], streaming=True)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
hyde = HyDEQueryTransform(include_original=True)

UnstructuredURLLoader = download_loader("UnstructuredURLLoader")

vector_store_info = VectorStoreInfo(
    content_info="Content of a URL",
    metadata_info=[
        MetadataInfo(
            name="source",
            type="str",
            description="a web URL",
        ),
        MetadataInfo(
            name="index_id",
            type="str",
            description="Unique group of urls",
        ),
    ],
)

def get_collection():
    collection = chroma_client.get_or_create_collection(name="indexes")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context, vector_store_info=vector_store_info)
    
    return index

def get_query_engine(indexes):
    collection = get_collection()
    
    if len(indexes) > 1:
        index_filters =  {"$or": [{"index_id": {"$eq": i}} for i in indexes]}
    else:
        index_filters =  {"index_id": {"$eq": indexes[0]}}

    return collection.as_query_engine(streaming=True, verbose=True, vector_store_kwargs={"where": index_filters})

def get_index_chat_engine(indexes):
    collection = get_collection()
    
    if len(indexes) > 1:
        index_filters =  {"$or": [{"index_id": {"$eq": i}} for i in indexes]}
    else:
        index_filters =  {"index_id": {"$eq": indexes[0]}}

    return collection.as_chat_engine(chat_mode="react", similarity_top_k=21, response_mode="tree_summarize", vector_store_query_mode="default",  streaming=True, verbose=True, vector_store_kwargs={"where": index_filters})


class ChatStream(BaseModel):
    id: str
    did: Optional[str]
    indexes: Optional[List[str]]
    messages: List[ChatMessage]

class Link(BaseModel):
    url: str


def response_generator(response):
    yield json.dumps({
        "sources": [{"id": s.node.id_, "url": s.node.metadata.get("source")} for s in response.source_nodes]
    })
    yield "\n ###endjson### \n\n"
    for text in response.response_gen:
        yield text


@app.post("/index/{index_id}/links")
def add(index_id, link: Link):
    
    collection = get_collection()
    
    loader = UnstructuredURLLoader(urls=[link.url])

    try:
        kb_data = loader.load()
        if not kb_data:
            return JSONResponse(content={'message': 'No data loaded from the provided link'}, status_code=400)
    except Exception as e:
        return JSONResponse(content={'message': 'Url load error'}, status_code=400)
    
    if not kb_data:
        return JSONResponse(content={'message': 'No data loaded from the provided link'}, status_code=400)
    
    doc = kb_data[0]
    # doc.embedding = embed_model.get_text_embedding(doc.text)
    doc.metadata["index_id"] = index_id

    collection.insert(doc)
    chroma_client.persist()

    return JSONResponse(content={'message': 'Document added successfully'})

@app.delete("/index/{index_id}/links")
def remove(index_id: str, link: Link):
    return JSONResponse(content={"message": "Documents deleted successfully"})

@app.post("/chat_stream")
def chat_stream(params: ChatStream):
    
    if params.did:
        id_resp = redisClient.hkeys("user_indexes:by_did:" + params.did.lower())
        if not id_resp:
            return "You have no indexes"
        indexes = [item.decode('utf-8').split(':')[0] for item in id_resp]
    elif params.indexes:
        indexes = params.indexes

    index = get_index_chat_engine(indexes)

    messages = params.messages
    last_message = messages[-1]

    streaming_response = index.chat(message=last_message.content, chat_history=messages)
    print(streaming_response.response)
    return JSONResponse(content=streaming_response.response)

    def response_generator():
        for text in streaming_response.response_gen:
            yield text

#        yield '\n\nSources:'
#        for s in streaming_response.source_nodes:
#            yield '\n\n'
#            yield  s.node.metadata.get("source")
            
        
    return StreamingResponse(response_generator(), media_type='text/event-stream')

@app.get("/debug")
def debug():
    collection = chroma_client.get_or_create_collection(name="indexes")
    x = collection.get(include=["metadatas", "documents", "embeddings"])
    return JSONResponse(x)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
