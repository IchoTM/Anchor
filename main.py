import os
from typing import Any, Dict, Optional
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from services.gemini_service import generate_grounding_message
from services.elevenlabs_service import generate_speech_audio
from services.photon_service import process_photon_message, parse_photon_payload

load_dotenv(find_dotenv(), override=True)

app = FastAPI(
    title="Anchor",
    description="Real-time iMessage dementia care assistant powered by Photon Spectrum and Gemini",
    version="0.1.0",
)


class IMessageWebhookRequest(BaseModel):
    sender_name: Optional[str] = Field(None, description="Name of the sender")
    relationship: Optional[str] = Field(None, description="Relationship to recipient")
    raw_message: Optional[str] = Field(None, description="Incoming message text")
    # Optional Photon fields
    event: Optional[str] = Field(None, description="Photon event type, e.g. message.received")
    data: Optional[Dict[str, Any]] = Field(None, description="Photon event data payload")
    text: Optional[str] = Field(None, description="Message text alternate")
    body: Optional[str] = Field(None, description="Message body alternate")
    sender: Optional[Any] = Field(None, description="Sender information")


class IMessageWebhookResponse(BaseModel):
    sender_name: str
    relationship: str
    raw_message: str
    grounded_message: str
    audio_generated: bool = Field(..., description="Success status indicating whether speech audio was generated")


@app.post("/webhook/imessage", response_model=IMessageWebhookResponse)
async def handle_imessage_webhook(request: Request):
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

    try:
        result = await process_photon_message(payload, generate_audio=True)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process message: {exc}",
        ) from exc

    print(f"[Anchor Grounded Message]: {result['grounded_message']}")

    return IMessageWebhookResponse(**result)


@app.post("/webhook/photon", response_model=IMessageWebhookResponse)
async def handle_photon_webhook(request: Request):
    """Dedicated endpoint for Photon / Spectrum framework webhooks."""
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

    try:
        result = await process_photon_message(payload, generate_audio=True)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process Photon webhook: {exc}",
        ) from exc

    print(f"[Anchor Grounded Message from Photon]: {result['grounded_message']}")

    return IMessageWebhookResponse(**result)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
