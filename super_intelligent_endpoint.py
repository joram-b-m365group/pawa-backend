"""
Pawa AI Backend with 2M Token Gemini Context
"""
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# Initialize FastAPI
app = FastAPI(title="Pawa AI Backend", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Models
class GeminiMessage(BaseModel):
    role: str
    content: str

class GeminiChatRequest(BaseModel):
    message: str
    conversation_history: List[GeminiMessage] = []
    model: str = "gemini-2.0-flash-exp"
    temperature: float = 0.7
    system_prompt: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    model_used: str = "gemini-2.0-flash-exp"

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Pawa AI Backend",
        "version": "2.0.0",
        "gemini_configured": bool(GOOGLE_API_KEY)
    }

# Gemini health check
@app.get("/gemini/health")
async def gemini_health():
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")

    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content("Hi")
        return {
            "status": "healthy",
            "model": "gemini-2.0-flash-exp",
            "test_response": response.text[:100]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

# Gemini chat endpoint
@app.post("/gemini/chat", response_model=ChatResponse)
async def gemini_chat(request: GeminiChatRequest):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")

    try:
        # Create model
        model = genai.GenerativeModel(
            model_name=request.model,
            generation_config={
                "temperature": request.temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )

        # Build conversation history
        history = []
        for msg in request.conversation_history:
            history.append({
                "role": msg.role if msg.role != "assistant" else "model",
                "parts": [msg.content]
            })

        # Start chat
        chat = model.start_chat(history=history)

        # Send message
        response = chat.send_message(request.message)

        return ChatResponse(
            response=response.text,
            model_used=request.model
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

# List available models
@app.get("/gemini/models")
async def list_gemini_models():
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")

    try:
        models = genai.list_models()
        gemini_models = [
            {
                "name": model.name,
                "display_name": model.display_name,
                "description": model.description
            }
            for model in models
            if "gemini" in model.name.lower() and "generateContent" in model.supported_generation_methods
        ]
        return {"models": gemini_models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "Pawa AI Backend",
        "version": "2.0.0",
        "features": [
            "2M token context with Gemini",
            "Multiple Gemini models",
            "Conversation history support"
        ],
        "endpoints": {
            "health": "/health",
            "gemini_health": "/gemini/health",
            "gemini_chat": "/gemini/chat (POST)",
            "gemini_models": "/gemini/models"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
