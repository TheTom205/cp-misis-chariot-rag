from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import json
import re
import clickhouse_connect
from docx import Document
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from vectorestore_vllm import Vectorestore
from fastapi.responses import JSONResponse

app = FastAPI(title="RAG System API", description="API для загрузки и обработки .docx документов", version="1.0.0")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"]   
)

# Настройка клиента для подключения к ClickHouse
client = clickhouse_connect.get_client(
    host='0.0.0.0',
    port=8123,
    username='default',
    password=''
)

# Убедитесь, что таблица в ClickHouse существует
client.command("CREATE DATABASE IF NOT EXISTS rag_system")
client.command('''
    CREATE TABLE IF NOT EXISTS rag_system.docs_ya (
        metadata String,
        content String,
        embedding Array(Float32)
    ) ENGINE = MergeTree()
    ORDER BY tuple()
''')

# Настройка эмбеддингов
embeddings = YandexGPTEmbeddings(api_key='AQVNwwO6r_Ko914j7zAW3H5lPga-sVKoXCipbr8_', folder_id='b1gp8gibtjpdhh1rk46d')
vectorestore = Vectorestore(docx_path="./RAG-Documents/data.docx")

class QueryRequest(BaseModel):
    query: str
    chat_id: str
    user_id: str

class QueryResponse(BaseModel):
    answer: str

class CreateChatRequest(BaseModel):
    user_id: str
    document_id: str

class ChatInfo(BaseModel):
    chat_id: str
    user_id: str
    document_id: str

class UploadResponse(BaseModel):
    document_id: str

class MessageRequest(BaseModel):
    chat_id: str

class MessageInfo(BaseModel):
    user_id: str
    chat_id: str
    message: str
    timestamp: str

@app.post("/get_messages_by_chat_id", response_model=list[MessageInfo])
async def get_messages_by_chat_id(request: MessageRequest):
    try:
        query = f'''
            SELECT user_id, chat_id, message, timestamp
            FROM rag_system.user_memory
            WHERE chat_id = '{request.chat_id}'
            ORDER BY timestamp DESC
        '''
        results = client.query(query).result_rows
        messages = [
            {
                "user_id": row[0],
                "chat_id": row[1],
                "message": row[2],
                "timestamp": row[3].isoformat()
            } for row in results
        ]
        return JSONResponse(content=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении сообщений: {str(e)}")

# Остальные методы и функции

@app.post("/get_answer", response_model=QueryResponse)
async def get_answer(request: QueryRequest):
    try:
        answer = vectorestore.async_get_answer(query=request.query, user_id=request.user_id, chat_id=request.chat_id)
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении ответа: {str(e)}")

@app.get("/get_chats", response_model=list[ChatInfo])
async def get_chats():
    try:
        chats = vectorestore.get_last_messages_by_chat()
        return JSONResponse(content=chats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении чатов: {str(e)}")

@app.post("/create_chat", response_model=ChatInfo)
async def create_chat(request: CreateChatRequest):
    try:
        chat_info = vectorestore.create_chat(user_id=request.user_id, document_id=request.document_id)
        return JSONResponse(content=chat_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при создании чата: {str(e)}")

@app.post("/upload_doc", response_model=UploadResponse)
async def upload_doc(file: UploadFile = File(...)):
    try:
        doc = Document(file.file)
        chapters = extract_chapters_from_docx(doc)
        content_dict = split_document_by_headings(doc, chapters)
        rag_documents = create_rag_documents(content_dict)
        document_id = str(uuid.uuid4())
        data_to_insert = [
            (entry["metadata"], entry["content"], entry["embedding"])
            for entry in rag_documents
        ]
        client.insert('rag_system.docs_ya', data_to_insert, column_names=["metadata", "content", "embedding"])
        return {"document_id": document_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке документа: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8999)
