import os
import uuid
import shutil
import logging
import sqlite3
import tempfile

from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form, UploadFile, File, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from langgraph.graph.state import CompiledStateGraph
from langchain.storage import InMemoryStore
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import settings
from src.logging_config import setup_logging
from src.schemas import QuestionRequest
from src.agent.factory import create_agent_graph
from src.services.document_indexer import DocumentIndexer
from src.services.llm_provider import LLMProvider
from src.services.vector_store import VectorStoreManager
from src.services.chat_service import get_agent_response, format_response_for_htmx

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events."""
    setup_logging()
    logger.info("Application starting up...")

    db_conn = sqlite3.connect(
        settings.CHECKPOINTER_SQLITE_PATH, check_same_thread=False
    )
    checkpointer = SqliteSaver(conn=db_conn)

    # Insantiate the provider once at startup
    llm_provider = LLMProvider()

    # Initialize the stores that will be shared across the application
    vector_store_manager = VectorStoreManager()
    doc_store = InMemoryStore()
    vector_store_conn = vector_store_manager.get_connection()

    # Create the agent graph and add it to the app's state
    app.state.agent_graph = create_agent_graph(
        checkpointer=checkpointer,
        vector_store=vector_store_conn,
        doc_store=doc_store,
        llm_provider=llm_provider,
    )

    # Create the Document Indexer and add it to the app's state
    indexing_llm = llm_provider.create_llm_with_fallback(settings.RAG_LLM_PROFILE)
    app.state.indexer = DocumentIndexer(
        vector_store=vector_store_conn,
        doc_store=doc_store,
        llm_for_summarization=indexing_llm,
    )
    logger.info("DocumentIndexer is initialized and ready.")

    yield

    logger.info("Application shutting down...")
    if db_conn:
        db_conn.close()
        logger.info("Checkpointer database connection closed.")


# Initialize the FastAPI application with the lifespan manager
app = FastAPI(lifespan=lifespan)


# --- Dependency Injectors ---
def get_agent_graph() -> CompiledStateGraph:
    """Dependency injector for the agent graph."""
    return app.state.agent_graph


def get_indexer() -> DocumentIndexer:
    """Dependency injector for the document indexer."""
    return app.state.indexer


# Mount static files and configure Jinja2 templates
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")
templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)


# --- API Routes ---


@app.get("/", response_class=RedirectResponse)
async def root():
    """Redirects the root URL to a new chat session with a unique ID."""
    return RedirectResponse(url=f"/chat/{uuid.uuid4()}", status_code=303)


@app.get("/chat/{thread_id}", response_class=HTMLResponse)
async def get_chat_page(request: Request, thread_id: str):
    """Serves the main chat page."""
    return templates.TemplateResponse(
        "index.html", {"request": request, "thread_id": thread_id}
    )


@app.post("/documents", response_class=JSONResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    indexer: DocumentIndexer = Depends(get_indexer),
):
    """Handles PDF document uploads for indexing."""
    file_paths = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(file_path)
        try:
            result = await indexer.process_files(file_paths)
            return JSONResponse(
                content={"message": "Documents processed successfully", **result}
            )
        except Exception:
            logger.error("An error occurred during document indexing", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "message": "An unexpected error occurred during document processing."
                },
            )


@app.post("/question", response_class=HTMLResponse)
async def handle_htmx_message(
    request: Request,
    message: str = Form(...),
    thread_id: str = Form(...),
    agent_graph: CompiledStateGraph = Depends(get_agent_graph),
):
    """Handles chat messages submitted from the HTMX front-end."""
    response_data = get_agent_response(message, thread_id, agent_graph)
    response_context = format_response_for_htmx(response_data)

    return templates.TemplateResponse(
        name="fragments/llm_response.html",
        context={"request": request, **response_context},
    )


@app.post("/api/question", response_class=JSONResponse)
async def handle_api_question(
    payload: QuestionRequest,
    agent_graph: CompiledStateGraph = Depends(get_agent_graph),
):
    """Handles JSON requests for the chatbot API."""
    response_data = get_agent_response(payload.question, payload.thread_id, agent_graph)
    return JSONResponse(content=response_data)
