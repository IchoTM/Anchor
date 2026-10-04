import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from services.photon_service import (
    process_photon_message,
    get_stored_audio,
    get_recent_events,
    get_latest_event,
)
from services.presage_service import update_patient_biometrics, get_current_biometrics

load_dotenv(find_dotenv(), override=True)

app = FastAPI(
    title="Anchor",
    description="Real-time iMessage dementia care assistant powered by Photon, Gemini, Presage, and ElevenLabs",
    version="0.1.0",
)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def render_template(filename: str, context: Optional[Dict[str, str]] = None) -> HTMLResponse:
    """Reads an HTML template from templates/ and substitutes any {{ key }} placeholders."""
    template_path = TEMPLATES_DIR / filename
    if not template_path.exists():
        raise HTTPException(status_code=500, detail=f"Template {filename} not found.")

    html = template_path.read_text(encoding="utf-8")
    if context:
        for key, val in context.items():
            html = html.replace(f"{{{{ {key} }}}}", str(val))
            html = html.replace(f"{{{{{key}}}}}", str(val))

    return HTMLResponse(content=html)


class IMessageWebhookRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    sender_name: Optional[str] = Field(None, description="Name of the sender")
    relationship: Optional[str] = Field(None, description="Relationship to recipient")
    raw_message: Optional[str] = Field(None, description="Incoming message text")
    anxiety_detected: Optional[bool] = Field(None, description="Presage anxiety detection flag")
    anxiety_score: Optional[float] = Field(None, description="Presage anxiety score from 0.0 to 1.0")
    presage: Optional[Dict[str, Any]] = Field(None, description="Presage biometric telemetry payload")
    event: Optional[str] = Field(None, description="Photon event type, e.g. message.received")
    data: Optional[Dict[str, Any]] = Field(None, description="Photon event data payload")
    text: Optional[str] = Field(None, description="Message text alternate")
    body: Optional[str] = Field(None, description="Message body alternate")
    sender: Optional[Any] = Field(None, description="Sender information")


class IMessageWebhookResponse(BaseModel):
    event_id: Optional[str] = Field(None, description="Unique event identifier")
    sender_name: str
    relationship: str
    raw_message: str
    grounded_message: str
    caregiver_reply: Optional[str] = Field(None, description="Reassuring auto-reply sent back to family member")
    anxiety_detected: bool = Field(..., description="Whether elevated anxiety was detected by Presage")
    anxiety_score: float = Field(..., description="Presage anxiety score between 0.0 to 1.0")
    audio_generated: bool = Field(..., description="Indicates whether ElevenLabs speech audio was triggered")
    audio_id: Optional[str] = Field(None, description="Unique ID for retrieving generated audio")
    audio_url: Optional[str] = Field(None, description="Relative URL to stream the audio MP3")


class PresageTelemetryRequest(BaseModel):
    anxiety_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Anxiety score between 0.0 to 1.0")
    anxiety_detected: Optional[bool] = Field(None, description="Direct flag for elevated anxiety")
    heart_rate: Optional[float] = Field(None, description="Current heart rate in BPM")
    respiration_rate: Optional[float] = Field(None, description="Breaths per minute")
    stress_index: Optional[float] = Field(None, description="Composite stress index from Presage")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional biometric sensor telemetry")


class PresageTelemetryResponse(BaseModel):
    status: str
    biometrics: Dict[str, Any]


def _detect_ngrok_tunnel_url() -> Optional[str]:
    """Queries ngrok's local client API to retrieve the active public tunnel URL."""
    try:
        req = urllib.request.Request("http://127.0.0.1:4040/api/tunnels", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=0.8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            tunnels = data.get("tunnels", [])
            for tunnel in tunnels:
                public_url = tunnel.get("public_url", "")
                if public_url.startswith("https://"):
                    return public_url.rstrip("/")
            if tunnels:
                return tunnels[0].get("public_url", "").rstrip("/")
    except Exception:
        pass
    return None


async def _parse_form_payload(request: Request) -> Dict[str, Any]:
    """Parses form-encoded or multipart incoming webhooks."""
    try:
        form_data = await request.form()
        return dict(form_data)
    except (AssertionError, ImportError):
        raw_body = await request.body()
        decoded = raw_body.decode("utf-8", errors="replace")
        parsed = urllib.parse.parse_qs(decoded, keep_blank_values=True)
        return {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}


async def _process_incoming_webhook(request: Request, source: str) -> Any:
    """Parses and grounds an incoming message from iMessage, Photon, or SMS gateway."""
    content_type = request.headers.get("content-type", "").lower()
    is_form = "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type
    payload: Dict[str, Any] = {}

    if is_form:
        payload = await _parse_form_payload(request)
        if "Body" in payload and "text" not in payload:
            payload["text"] = payload["Body"]
        if "From" in payload and "sender" not in payload:
            payload["sender"] = payload["From"]
    else:
        try:
            payload = await request.json()
        except Exception:
            try:
                payload = await _parse_form_payload(request)
                is_form = True
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Invalid request payload") from exc

    try:
        result = await process_photon_message(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process message from {source}: {exc}",
        ) from exc

    print(f"[Anchor Grounded Message from {source}]: {result['grounded_message']}")

    # If incoming via Twilio SMS, return TwiML XML so the sender gets an immediate text reply
    if is_form and ("From" in payload or "AccountSid" in payload):
        caregiver_msg = result.get("caregiver_reply", "Anchor: Message delivered and grounded.")
        twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{caregiver_msg}</Message>
</Response>"""
        return Response(content=twiml_response, media_type="application/xml")

    return IMessageWebhookResponse(**result)


@app.get("/api/tunnel-url")
async def get_tunnel_url(request: Request):
    """Returns the live public tunnel URL."""
    ngrok_url = _detect_ngrok_tunnel_url()
    if ngrok_url:
        return {"public_url": ngrok_url, "source": "ngrok_api"}

    env_url = os.getenv("PUBLIC_URL", os.getenv("NGROK_URL", "")).rstrip("/")
    if env_url:
        return {"public_url": env_url, "source": "environment"}

    origin = str(request.base_url).rstrip("/")
    return {"public_url": origin, "source": "origin"}


@app.post("/webhook/imessage")
async def handle_imessage_webhook(request: Request):
    """Handles incoming iMessage or SMS webhooks with Presage anxiety-gated voice generation."""
    return await _process_incoming_webhook(request, source="iMessage")


@app.post("/webhook/photon")
async def handle_photon_webhook(request: Request):
    """Dedicated endpoint for Photon / Spectrum framework webhooks."""
    return await _process_incoming_webhook(request, source="Photon")


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


@app.get("/api/events")
async def list_recent_events():
    """Returns recent message grounding events for the demo surface."""
    return {"events": get_recent_events()}


@app.get("/api/events/latest")
async def get_most_recent_event():
    """Returns the single latest event."""
    return {"event": get_latest_event()}


@app.get("/audio/{audio_id}")
async def get_audio_stream(audio_id: str):
    """Streams generated ElevenLabs MP3 audio for high-anxiety grounding interventions."""
    audio_bytes = get_stored_audio(audio_id)
    if not audio_bytes:
        raise HTTPException(status_code=404, detail="Audio file not found or expired")
    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.get("/text", response_class=HTMLResponse)
@app.get("/mobile", response_class=HTMLResponse)
async def serve_mobile_caregiver():
    """Serves the mobile Caregiver iMessage interface."""
    return render_template("mobile.html")


@app.get("/", response_class=HTMLResponse)
@app.get("/demo", response_class=HTMLResponse)
async def serve_demo_tablet():
    """Serves Grandma Eleanor's Bedside Station dashboard."""
    configured_public_url = os.getenv("PUBLIC_URL", os.getenv("NGROK_URL", "")).rstrip("/")
    return render_template("tablet.html", context={"public_url": configured_public_url})


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
