import uuid
import markdown2
import sqlite3
import os
import tempfile
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, Request, Form, UploadFile, File, Header
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain.storage import InMemoryStore
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI

from src.agent.agent import run_chat
from src.agent.factory import create_agent_graph
from src.services.document_indexer import DocumentIndexer
from src.services.vector_store import VectorStoreManager
from src.config import settings


# --- Pydantic model for JSON API requests ---
class QuestionRequest(BaseModel):
    question: str
    thread_id: str


# --- Lifespan and App Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application starting up...")

    db_conn = sqlite3.connect(
        settings.CHECKPOINTER_SQLITE_PATH, check_same_thread=False
    )
    checkpointer = SqliteSaver(conn=db_conn)

    # Initialize the stores that will be shared
    vector_store_manager = VectorStoreManager()
    doc_store = InMemoryStore()
    vector_store_conn = vector_store_manager.get_connection()

    agent_graph = create_agent_graph(
        checkpointer=checkpointer,
        vector_store=vector_store_conn,
        doc_store=doc_store,
    )
    app.state.agent_graph = agent_graph

    # Create the Document Indexer
    indexing_llm = ChatOpenAI(
        model=settings.RAG_LLM, temperature=settings.RAG_TEMPERATURE
    )

    app.state.indexer = DocumentIndexer(
        vector_store=vector_store_conn,
        doc_store=doc_store,
        llm_for_summarization=indexing_llm,
    )
    print("DocumentIndexer is initialized and ready.")

    yield

    print("Application shutting down...")
    if db_conn:
        db_conn.close()
        print("Checkpointer database connection closed.")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")
templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)


# --- Routes ---
@app.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url=f"/chat/{uuid.uuid4()}", status_code=303)


@app.get("/chat/{thread_id}", response_class=HTMLResponse)
async def get_chat_page(request: Request, thread_id: str):
    return templates.TemplateResponse(
        "index.html", {"request": request, "thread_id": thread_id}
    )


@app.post("/documents", response_class=JSONResponse)
async def upload_documents(request: Request, files: List[UploadFile] = File(...)):
    file_paths = []
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                file_paths.append(file_path)

            indexer = request.app.state.indexer
            result = indexer.process_files(file_paths)

        return JSONResponse(
            content={"message": "Documents processed successfully", **result}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": f"An error occurred: {e}"}
        )


@app.post("/send_message")
async def handle_question(
    request: Request,
    message: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
    hx_request: Optional[str] = Header(None),
):
    try:
        if hx_request:
            user_message, conv_id = message, thread_id
        else:
            body = await request.json()
            question_request = QuestionRequest(**body)
            user_message, conv_id = (
                question_request.question,
                question_request.thread_id,
            )

        if not user_message or not conv_id:
            return JSONResponse(
                status_code=400, content={"error": "Missing question or thread_id"}
            )

        # 1. Get the structured response
        agent_graph = request.app.state.agent_graph
        response_data = run_chat(user_message, conv_id, agent_graph)

        # 2. Format the data based on the request type
        if hx_request:
            # Format for HTML template
            html_from_markdown = markdown2.markdown(
                response_data["answer"], extras=["tables", "cuddled-lists", "breaks"]
            )
            context = {
                "request": request,
                "llm_response": html_from_markdown,
                "references": response_data["references"],
            }
            return templates.TemplateResponse(
                name="fragments/llm_response.html", context=context
            )
        else:
            # Format for JSON API response
            api_references = [ref["content"] for ref in response_data["references"]]
            return JSONResponse(
                content={
                    "answer": response_data["answer"],
                    "references": api_references,
                }
            )

    except Exception as e:
        print(f"Caught generic exception: {e}")
        error_message = "An unexpected error occurred. Please try again."
        if hx_request:
            return templates.TemplateResponse(
                name="fragments/llm_response.html",
                context={
                    "request": request,
                    "llm_response": error_message,
                    "references": [],
                },
            )
        else:
            return JSONResponse(status_code=500, content={"error": error_message})
