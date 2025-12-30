import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi.responses import StreamingResponse
import asyncio


from langchain_openai import ChatOpenAI

load_dotenv("../.env")

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "RAG backend running"}

# Globals (simple for now)
VECTOR_DB = {}
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    openai_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_path = f"../data/{file.filename}"

    os.makedirs("../data", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # Add metadata
    for c in chunks:
        c.metadata["source"] = file.filename
        c.metadata["page"] = c.metadata.get("page", None)

    vector_db = FAISS.from_documents(chunks, embeddings)
    os.makedirs("../vectorstore", exist_ok=True)
    vector_db.save_local(f"../vectorstore/{file.filename}")

    VECTOR_DB[file.filename] = vector_db

    return {"message": "PDF uploaded and indexed", "file": file.filename}
@app.post("/ask")
async def ask_question(question: str, file_name: str):
    if file_name not in VECTOR_DB:
        VECTOR_DB[file_name] = FAISS.load_local(
            f"../vectorstore/{file_name}",
            embeddings,
            allow_dangerous_deserialization=True
        )

    retriever = VECTOR_DB[file_name].as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Answer strictly from the context.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": [
            {"page": d.metadata.get("page"), "source": d.metadata.get("source")}
            for d in docs
        ]
    }

@app.post("/ask_stream")
async def ask_question_stream(question: str, file_name: str):
    if file_name not in VECTOR_DB:
        VECTOR_DB[file_name] = FAISS.load_local(
            f"../vectorstore/{file_name}",
            embeddings,
            allow_dangerous_deserialization=True
        )

    retriever = VECTOR_DB[file_name].as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Answer strictly from the context.

Context:
{context}

Question:
{question}
"""

    async def token_generator():
        for chunk in llm.stream(prompt):
            yield chunk.content
            await asyncio.sleep(0)  # allow async streaming

    return StreamingResponse(
        token_generator(),
        media_type="text/plain"
    )
