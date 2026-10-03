import os
from typing import Any, Dict, Optional
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from services.photon_service import process_photon_message, get_stored_audio
from services.presage_service import update_patient_biometrics, get_current_biometrics

load_dotenv(find_dotenv(), override=True)

app = FastAPI(
    title="Anchor",
    description="Real-time iMessage dementia care assistant powered by Photon, Gemini, Presage, and ElevenLabs",
    version="0.1.0",
)


class IMessageWebhookRequest(BaseModel):
    sender_name: Optional[str] = Field(None, description="Name of the sender")
    relationship: Optional[str] = Field(None, description="Relationship to recipient")
    raw_message: Optional[str] = Field(None, description="Incoming message text")
    # Presage biometric fields (optional overrides per message)
    anxiety_detected: Optional[bool] = Field(None, description="Presage anxiety detection flag")
    anxiety_score: Optional[float] = Field(None, description="Presage anxiety score from 0.0 to 1.0")
    presage: Optional[Dict[str, Any]] = Field(None, description="Presage biometric telemetry payload")
    # Optional Photon event fields
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
    anxiety_detected: bool = Field(..., description="Whether elevated anxiety was detected by Presage")
    anxiety_score: float = Field(..., description="Presage anxiety score between 0.0 and 1.0")
    audio_generated: bool = Field(..., description="Indicates whether ElevenLabs speech audio was triggered")
    audio_id: Optional[str] = Field(None, description="Unique ID for retrieving generated audio")
    audio_url: Optional[str] = Field(None, description="Relative URL to stream the audio MP3")


class PresageTelemetryRequest(BaseModel):
    anxiety_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Anxiety score between 0.0 and 1.0")
    anxiety_detected: Optional[bool] = Field(None, description="Direct flag for elevated anxiety")
    heart_rate: Optional[float] = Field(None, description="Current heart rate in BPM")
    respiration_rate: Optional[float] = Field(None, description="Breaths per minute")
    stress_index: Optional[float] = Field(None, description="Composite stress index from Presage")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional biometric sensor telemetry")


class PresageTelemetryResponse(BaseModel):
    status: str
    biometrics: Dict[str, Any]


@app.post("/webhook/imessage", response_model=IMessageWebhookResponse)
async def handle_imessage_webhook(request: Request):
    """Handles incoming iMessage webhooks with Presage anxiety-gated voice generation."""
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

    try:
        result = await process_photon_message(payload)
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
        result = await process_photon_message(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process Photon webhook: {exc}",
        ) from exc

    print(f"[Anchor Grounded Message from Photon]: {result['grounded_message']}")
    return IMessageWebhookResponse(**result)


@app.post("/telemetry/presage", response_model=PresageTelemetryResponse)
async def receive_presage_telemetry(payload: PresageTelemetryRequest):
    """Receives live biometric readings from the Presage emotional sensing SDK."""
    updated = update_patient_biometrics(
        anxiety_score=payload.anxiety_score,
        anxiety_detected=payload.anxiety_detected,
        heart_rate=payload.heart_rate,
        respiration_rate=payload.respiration_rate,
        stress_index=payload.stress_index,
        metadata=payload.metadata,
    )
    print(f"[Presage Telemetry Received]: Anxiety={updated['anxiety_score']:.2f}, High={updated['anxiety_detected']}")
    return PresageTelemetryResponse(status="success", biometrics=updated)


@app.get("/telemetry/presage", response_model=PresageTelemetryResponse)
async def get_presage_telemetry():
    """Retrieves the current patient biometric state."""
    return PresageTelemetryResponse(status="success", biometrics=get_current_biometrics())


@app.get("/audio/{audio_id}")
async def get_audio_stream(audio_id: str):
    """Streams generated ElevenLabs MP3 audio for high-anxiety grounding interventions."""
    audio_bytes = get_stored_audio(audio_id)
    if not audio_bytes:
        raise HTTPException(status_code=404, detail="Audio file not found or expired")
    return Response(content=audio_bytes, media_type="audio/mpeg")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
