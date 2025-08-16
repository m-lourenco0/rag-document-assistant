from pydantic import BaseModel


class QuestionRequest(BaseModel):
    """Defines the JSON payload for a question sent to the API."""

    question: str
    thread_id: str
