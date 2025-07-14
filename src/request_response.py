from typing import Optional, List 
from pydantic import BaseModel, Field
import time

# --- Request and Response Models (Same as before) ---
class CompletionRequest(BaseModel):
    model: str
    prompt: str
    prompt_tokens: int = 0
    max_tokens: int = 50 #Field(alias="max_new_tokens", default=50)
    temperature: float = 0.7
    top_p: float = 0.9

class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    logprobs: Optional[float] = None
    finish_reason: Optional[str] = "length"

class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CompletionResponse(BaseModel):
    id: str = "cmpl-dummy_id"
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage