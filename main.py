import os
import urllib.parse
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


class IMessageWebhookRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    sender_name: Optional[str] = Field(None, description="Name of the sender")
    relationship: Optional[str] = Field(None, description="Relationship to recipient")
    raw_message: Optional[str] = Field(None, description="Incoming message text")
    # Presage biometric fields (optional overrides per message)
    anxiety_detected: Optional[bool] = Field(None, description="Presage anxiety detection flag")
    anxiety_score: Optional[float] = Field(None, description="Presage anxiety score from 0.0 to 1.0")
    presage: Optional[Dict[str, Any]] = Field(None, description="Presage biometric telemetry payload")
    # Optional Photon / SMS event fields
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


async def _parse_form_payload(request: Request) -> Dict[str, Any]:
    """
    Robust form parser that uses Starlette's request.form() when available,
    falling back to standard library urllib.parse for urlencoded data.
    """
    try:
        form_data = await request.form()
        return dict(form_data)
    except (AssertionError, ImportError):
        # python-multipart not installed; decode urlencoded data directly
        raw_body = await request.body()
        decoded = raw_body.decode("utf-8", errors="replace")
        parsed = urllib.parse.parse_qs(decoded, keep_blank_values=True)
        return {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}


async def _process_incoming_webhook(request: Request, source: str) -> Any:
    """
    Parses and grounds an incoming message from iMessage, Photon, or SMS gateway.
    Supports JSON as well as form-encoded payloads (Twilio, Telnyx).
    If incoming via Twilio SMS, returns TwiML XML to instantly reply to the sender's phone.
    """
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

    # If incoming from a live SMS gateway (e.g. Twilio), return TwiML XML so the sender gets an immediate text reply!
    if is_form and ("From" in payload or "AccountSid" in payload):
        caregiver_msg = result.get("caregiver_reply", "Anchor: Message delivered and grounded.")
        twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{caregiver_msg}</Message>
</Response>"""
        return Response(content=twiml_response, media_type="application/xml")

    return IMessageWebhookResponse(**result)


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


@app.get("/", response_class=HTMLResponse)
@app.get("/demo", response_class=HTMLResponse)
async def serve_demo_tablet():
    """Serves the interactive Bedside Tablet Dementia Care Simulator."""
    demo_phone_number = os.getenv("LIVE_DEMO_NUMBER", "")
    live_badge_html = ""
    if demo_phone_number:
        live_badge_html = f"""
        <div class="header-badge live-phone-badge">
          <span>📱 Text Live:</span>
          <strong>{demo_phone_number}</strong>
        </div>
        """

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Anchor • Dementia Care Bedside Station</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0c1117;
      --card-bg: #161e29;
      --card-border: rgba(255, 255, 255, 0.08);
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.25);
      --calm: #10b981;
      --calm-bg: rgba(16, 185, 129, 0.12);
      --panic: #f43f5e;
      --panic-bg: rgba(244, 63, 94, 0.12);
      --text: #f1f5f9;
      --text-muted: #94a3b8;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: radial-gradient(circle at 50% 0%, #172334 0%, var(--bg) 75%);
      color: var(--text);
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      min-height: 100vh;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 16px;
      flex-wrap: wrap;
      gap: 12px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand-icon {
      width: 40px;
      height: 40px;
      border-radius: 10px;
      background: linear-gradient(135deg, #0284c7, #38bdf8);
      display: grid;
      place-items: center;
      font-size: 22px;
      box-shadow: 0 0 20px var(--accent-glow);
    }

    .brand-title h1 {
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }

    .brand-title p {
      font-size: 13px;
      color: var(--text-muted);
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }

    .header-badge {
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(255, 255, 255, 0.05);
      padding: 6px 14px;
      border-radius: 999px;
      border: 1px solid var(--card-border);
      font-size: 13px;
    }

    .live-phone-badge {
      background: rgba(56, 189, 248, 0.12);
      border-color: rgba(56, 189, 248, 0.35);
      color: #7dd3fc;
    }

    .btn-sound-gate {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 13px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }

    .btn-sound-gate.active {
      background: var(--calm-bg);
      border-color: rgba(16, 185, 129, 0.4);
      color: var(--calm);
    }

    .pulse-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--calm);
      box-shadow: 0 0 10px var(--calm);
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }

    .container {
      display: grid;
      grid-template-columns: 360px 320px 1fr;
      gap: 20px;
      flex: 1;
    }

    @media (max-width: 1200px) {
      .container { grid-template-columns: 1fr 1fr; }
      .bedside-display { grid-column: span 2; }
    }

    @media (max-width: 840px) {
      .container { grid-template-columns: 1fr; }
      .bedside-display { grid-column: span 1; }
    }

    .panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 18px;
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }

    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .panel-title {
      font-size: 15px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
    }

    /* PHONE SIMULATOR */
    .imessage-shell {
      background: #0f1621;
      border-radius: 14px;
      border: 1px solid var(--card-border);
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .field-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    label {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
    }

    select, input, textarea {
      width: 100%;
      background: #182232;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 10px;
      padding: 10px 12px;
      color: var(--text);
      font-family: inherit;
      font-size: 14px;
      outline: none;
      transition: all 0.2s;
    }

    select:focus, input:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 10px var(--accent-glow);
    }

    .preset-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .chip {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 6px 10px;
      font-size: 11px;
      cursor: pointer;
      color: var(--text-muted);
      transition: 0.15s;
    }

    .chip:hover {
      background: rgba(255, 255, 255, 0.08);
      color: var(--text);
    }

    .btn-send {
      background: linear-gradient(135deg, #0284c7, #0ea5e9);
      color: white;
      border: none;
      border-radius: 10px;
      padding: 12px;
      font-weight: 600;
      font-size: 14px;
      cursor: pointer;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 8px;
      transition: all 0.2s;
    }

    .btn-send:hover {
      background: linear-gradient(135deg, #0369a1, #0284c7);
      box-shadow: 0 0 18px var(--accent-glow);
      transform: translateY(-1px);
    }

    /* PRESAGE BIOMETRIC RADAR */
    .metric-card {
      background: #0f1621;
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .metric-val {
      font-size: 26px;
      font-weight: 800;
      letter-spacing: -0.02em;
    }

    .metric-lbl {
      font-size: 12px;
      color: var(--text-muted);
    }

    .anxiety-toggle-box {
      background: #0f1621;
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .toggle-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .status-pill {
      font-size: 12px;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 999px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .status-calm {
      background: var(--calm-bg);
      color: var(--calm);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-panic {
      background: var(--panic-bg);
      color: var(--panic);
      border: 1px solid rgba(244, 63, 94, 0.3);
    }

    .slider {
      -webkit-appearance: none;
      width: 100%;
      height: 8px;
      border-radius: 4px;
      background: #253346;
      outline: none;
    }

    .slider::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 22px;
      height: 22px;
      border-radius: 50%;
      background: var(--accent);
      cursor: pointer;
      box-shadow: 0 0 10px var(--accent-glow);
    }

    /* BEDSIDE PATIENT TABLET DISPLAY */
    .bedside-display {
      background: radial-gradient(circle at 50% 20%, #1f2e42 0%, #101620 100%);
      border: 2px solid rgba(255, 255, 255, 0.12);
      border-radius: 24px;
      padding: 32px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      position: relative;
      overflow: hidden;
    }

    .bedside-display::after {
      content: "";
      position: absolute;
      top: -50%;
      right: -50%;
      width: 100%;
      height: 100%;
      background: radial-gradient(circle, var(--accent-glow) 0%, transparent 60%);
      pointer-events: none;
    }

    .bedside-clock {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      padding-bottom: 18px;
    }

    .time-large {
      font-size: 42px;
      font-weight: 800;
      letter-spacing: -0.03em;
    }

    .date-large {
      font-size: 16px;
      color: var(--text-muted);
      margin-top: 2px;
    }

    .patient-tag {
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--card-border);
      border-radius: 999px;
      padding: 6px 14px;
      font-size: 13px;
      font-weight: 600;
    }

    .grounding-hero {
      margin: 24px 0;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .sender-banner {
      display: flex;
      align-items: center;
      gap: 16px;
    }

    .sender-avatar {
      width: 58px;
      height: 58px;
      border-radius: 18px;
      background: linear-gradient(135deg, #0ea5e9, #38bdf8);
      display: grid;
      place-items: center;
      font-size: 26px;
      box-shadow: 0 6px 20px var(--accent-glow);
    }

    .sender-names h2 {
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }

    .sender-names span {
      font-size: 15px;
      color: var(--accent);
      font-weight: 600;
    }

    .grounded-quote {
      font-family: 'Newsreader', Georgia, serif;
      font-size: 27px;
      line-height: 1.45;
      color: #ffffff;
      background: rgba(255, 255, 255, 0.03);
      border-left: 4px solid var(--accent);
      border-radius: 0 16px 16px 0;
      padding: 22px;
      letter-spacing: -0.01em;
      transition: all 0.3s ease;
    }

    .raw-intercepted {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 13px;
      color: var(--text-muted);
      background: rgba(0, 0, 0, 0.2);
      padding: 8px 14px;
      border-radius: 10px;
      border: 1px dashed var(--card-border);
    }

    .caregiver-receipt {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      font-size: 12px;
      color: #38bdf8;
      background: rgba(56, 189, 248, 0.08);
      padding: 10px 14px;
      border-radius: 10px;
      border: 1px solid rgba(56, 189, 248, 0.25);
      line-height: 1.4;
    }

    .voice-player {
      background: #141d2a;
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 14px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    .voice-status {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 14px;
    }

    .audio-wave {
      display: flex;
      align-items: center;
      gap: 3px;
      height: 18px;
    }

    .bar {
      width: 3px;
      height: 100%;
      background: var(--accent);
      border-radius: 2px;
      animation: wave 1.2s ease-in-out infinite;
    }

    .bar:nth-child(2) { animation-delay: 0.2s; height: 60%; }
    .bar:nth-child(3) { animation-delay: 0.4s; height: 90%; }
    .bar:nth-child(4) { animation-delay: 0.1s; height: 40%; }

    @keyframes wave {
      0%, 100% { transform: scaleY(0.4); }
      50% { transform: scaleY(1); }
    }

    .btn-replay {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--card-border);
      color: white;
      border-radius: 10px;
      padding: 8px 16px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: 0.2s;
    }

    .btn-replay:hover {
      background: rgba(255, 255, 255, 0.15);
    }

    audio { display: none; }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-icon">⚓</div>
      <div class="brand-title">
        <h1>Anchor • Dementia Care Assistant</h1>
        <p>Real-Time Cognitive Grounding & Biometric Voice Intervention</p>
      </div>
    </div>
    <div class="header-actions">
      <!-- LIVE_PHONE_BADGE -->
      <button class="btn-sound-gate" id="soundToggleBtn" onclick="toggleAudioPermission()">
        <span>🔊 Enable Sound</span>
      </button>
      <div class="header-badge">
        <div class="pulse-dot"></div>
        <span id="gateway-status">Spectrum & Presage Gateway Active</span>
      </div>
    </div>
  </header>

  <div class="container">

    <!-- COLUMN 1: FAMILY iMESSAGE SIMULATOR -->
    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">Family iMessage Simulator</span>
        <span style="font-size: 18px;">📱</span>
      </div>

      <div class="imessage-shell">
        <div class="field-group">
          <label>Select Family Contact</label>
          <select id="contactSelect" onchange="onContactChange()">
            <option value="Alex|Grandson|👴">Alex (Grandson)</option>
            <option value="Sarah|Daughter|👩">Sarah (Daughter)</option>
            <option value="David|Son|👨">David (Son)</option>
            <option value="Maria|Caregiver|🩺">Maria (Caregiver)</option>
            <option value="Dr. Chen|Doctor|🩺">Dr. Chen (Doctor)</option>
            <option value="custom|Custom|💬">Custom Contact...</option>
          </select>
        </div>

        <div class="field-group" id="customContactGroup" style="display:none; flex-direction:column; gap: 8px;">
          <label>Custom Sender Info</label>
          <input type="text" id="customName" placeholder="Name (e.g. Emily)">
          <input type="text" id="customRelation" placeholder="Relationship to Patient (e.g. Niece)">
        </div>

        <div class="field-group">
          <label>Incoming Message</label>
          <textarea id="rawMessageInput" rows="3" placeholder="Type a message...">I will be there in 10 mins!</textarea>
        </div>

        <div class="field-group">
          <label>Quick Dementia Test Scenarios</label>
          <div class="preset-chips">
            <span class="chip" onclick="setScenario('Alex', 'Grandson', 'I will be there in 10 mins!')">⏳ 10 Mins (Panic)</span>
            <span class="chip" onclick="setScenario('Sarah', 'Daughter', 'Did you take your pills? Call me.')">💊 Pill Check</span>
            <span class="chip" onclick="setScenario('David', 'Son', 'I found your glasses on the table.')">👓 Lost Item</span>
            <span class="chip" onclick="setScenario('Dr. Chen', 'Doctor', 'Appointment confirmed for 3:00 PM today.')">📅 Appointment</span>
          </div>
        </div>

        <button class="btn-send" id="sendBtn" onclick="sendSimulatedMessage()">
          <span>Send Incoming iMessage</span>
          <span>➔</span>
        </button>
      </div>

      <div style="font-size: 12px; color: var(--text-muted); line-height: 1.5;">
        Intercepts incoming text messages, normalizes metadata through Photon/Twilio, and grounds cognitive disorientation via Gemini.
      </div>
    </div>

    <!-- COLUMN 2: PRESAGE BIOMETRIC RADAR -->
    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">Presage Biometrics</span>
        <span style="font-size: 18px;">🫀</span>
      </div>

      <div class="metric-card">
        <div>
          <div class="metric-lbl">HEART RATE</div>
          <div class="metric-val" id="hrDisplay">74 <span style="font-size: 14px; font-weight: 500; color: var(--text-muted);">BPM</span></div>
        </div>
        <div style="font-size: 26px;">❤️</div>
      </div>

      <div class="metric-card">
        <div>
          <div class="metric-lbl">RESPIRATION</div>
          <div class="metric-val" id="rrDisplay">16 <span style="font-size: 14px; font-weight: 500; color: var(--text-muted);">BrPM</span></div>
        </div>
        <div style="font-size: 26px;">🫁</div>
      </div>

      <div class="anxiety-toggle-box">
        <div class="toggle-row">
          <span style="font-size: 13px; font-weight: 700;">ANXIETY STATE</span>
          <span id="anxietyBadge" class="status-pill status-calm">CALM (TEXT ONLY)</span>
        </div>

        <input type="range" min="0" max="100" value="25" class="slider" id="anxietySlider" oninput="onAnxietySlider(this.value)">

        <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted);">
          <span>0% Baseline</span>
          <span id="anxietyPct">25%</span>
          <span>100% Panic Spike</span>
        </div>

        <div style="display: flex; gap: 8px; margin-top: 6px;">
          <button class="chip" style="flex:1; text-align:center;" onclick="setAnxiety(20)">Simulate Calm</button>
          <button class="chip" style="flex:1; text-align:center; color:#f43f5e;" onclick="setAnxiety(88)">Simulate Anxiety Spike</button>
        </div>
      </div>

      <div style="font-size: 12px; color: var(--text-muted); line-height: 1.5;">
        Presage emotional sensing monitors patient agitation. ElevenLabs TTS voice synthesis triggers <strong>only</strong> when anxiety exceeds threshold.
      </div>
    </div>

    <!-- COLUMN 3: BEDSIDE TABLET (PATIENT VIEW) -->
    <div class="bedside-display">
      <div class="bedside-clock">
        <div>
          <div class="time-large" id="clockTime">10:42 AM</div>
          <div class="date-large" id="clockDate">Tuesday, October 24</div>
        </div>
        <div class="patient-tag">Bedside Monitor • Room 3B</div>
      </div>

      <div class="grounding-hero">
        <div class="sender-banner">
          <div class="sender-avatar" id="avatarDisplay">👴</div>
          <div class="sender-names">
            <h2 id="senderNameDisplay">Alex</h2>
            <span id="relationshipDisplay">Your Grandson</span>
          </div>
        </div>

        <div class="grounded-quote" id="groundedDisplay">
          "Hi Grandma. Your grandson, Alex, just sent you a message. He wants you to know that he is coming over and will be at your house in 10 minutes."
        </div>

        <div class="raw-intercepted">
          <span>Intercepted raw message:</span>
          <strong id="rawInterceptedDisplay">"I will be there in 10 mins!"</strong>
        </div>

        <div class="caregiver-receipt" id="caregiverReceiptBox">
          <span>📲</span>
          <div>
            <strong>Caregiver Auto-Receipt Sent:</strong>
            <span id="caregiverReceiptText">Anchor: Message delivered and grounded for Rachel. Vitals are calm.</span>
          </div>
        </div>
      </div>

      <div class="voice-player">
        <div class="voice-status">
          <div class="audio-wave" id="audioWave" style="opacity: 0.3;">
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
          </div>
          <span id="audioStatusText" style="color: var(--text-muted);">Calm state • Voice synthesis idle</span>
        </div>

        <button class="btn-replay" id="replayBtn" style="display: none;" onclick="replayAudio()">
          <span>🔊 Play Voice</span>
        </button>
      </div>

      <audio id="audioElement"></audio>
    </div>

  </div>

  <script>
    let currentAudioUrl = null;
    let lastEventId = null;
    let audioUnlocked = false;

    // Browser audio policy unlock
    function toggleAudioPermission() {
      const audioEl = document.getElementById('audioElement');
      const soundBtn = document.getElementById('soundToggleBtn');
      audioEl.play().then(() => {
        audioEl.pause();
        audioUnlocked = true;
        soundBtn.classList.add('active');
        soundBtn.innerHTML = '<span>🔊 Sound Active</span>';
      }).catch(() => {
        audioUnlocked = true;
        soundBtn.classList.add('active');
        soundBtn.innerHTML = '<span>🔊 Sound Active</span>';
      });
    }

    // Auto unlock on first user interaction anywhere
    window.addEventListener('click', () => {
      if (!audioUnlocked) toggleAudioPermission();
    }, { once: true });

    // Clock
    function updateClock() {
      const now = new Date();
      document.getElementById('clockTime').textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      document.getElementById('clockDate').textContent = now.toLocaleDateString([], { weekday: 'long', month: 'long', day: 'numeric' });
    }
    setInterval(updateClock, 1000);
    updateClock();

    function onContactChange() {
      const val = document.getElementById('contactSelect').value;
      const customGroup = document.getElementById('customContactGroup');
      if (val.startsWith('custom')) {
        customGroup.style.display = 'flex';
      } else {
        customGroup.style.display = 'none';
      }
    }

    function setScenario(name, relationship, message) {
      document.getElementById('rawMessageInput').value = message;
      const select = document.getElementById('contactSelect');
      for (let i = 0; i < select.options.length; i++) {
        if (select.options[i].value.includes(name)) {
          select.selectedIndex = i;
          onContactChange();
          break;
        }
      }
    }

    function onAnxietySlider(val) {
      const pct = parseInt(val);
      document.getElementById('anxietyPct').textContent = pct + '%';
      const badge = document.getElementById('anxietyBadge');
      const hr = document.getElementById('hrDisplay');
      const rr = document.getElementById('rrDisplay');

      if (pct >= 60) {
        badge.className = 'status-pill status-panic';
        badge.textContent = 'ELEVATED ANXIETY (VOICE GATED ON)';
        hr.innerHTML = `${Math.round(80 + (pct * 0.35))} <span style="font-size:14px;color:var(--text-muted);">BPM</span>`;
        rr.innerHTML = `${Math.round(18 + (pct * 0.1))} <span style="font-size:14px;color:var(--text-muted);">BrPM</span>`;
      } else {
        badge.className = 'status-pill status-calm';
        badge.textContent = 'CALM (TEXT ONLY)';
        hr.innerHTML = `72 <span style="font-size:14px;color:var(--text-muted);">BPM</span>`;
        rr.innerHTML = `15 <span style="font-size:14px;color:var(--text-muted);">BrPM</span>`;
      }

      // Sync with Presage telemetry endpoint
      fetch('/telemetry/presage', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          anxiety_score: pct / 100.0,
          anxiety_detected: pct >= 60,
          heart_rate: parseFloat(hr.innerText),
          respiration_rate: parseFloat(rr.innerText)
        })
      }).catch(err => console.warn('Telemetry sync error:', err));
    }

    function setAnxiety(val) {
      document.getElementById('anxietySlider').value = val;
      onAnxietySlider(val);
    }

    async function sendSimulatedMessage() {
      const sendBtn = document.getElementById('sendBtn');
      const contactVal = document.getElementById('contactSelect').value;
      const rawMessage = document.getElementById('rawMessageInput').value.trim();
      if (!rawMessage) return;

      let senderName = "Alex";
      let relationship = "Grandson";
      let avatar = "👴";

      if (contactVal.startsWith('custom')) {
        senderName = document.getElementById('customName').value.trim() || "Family Member";
        relationship = document.getElementById('customRelation').value.trim() || "Family";
        avatar = "💬";
      } else {
        const parts = contactVal.split('|');
        senderName = parts[0];
        relationship = parts[1];
        avatar = parts[2] || "💬";
      }

      const anxietyScore = parseInt(document.getElementById('anxietySlider').value) / 100.0;
      const anxietyDetected = anxietyScore >= 0.60;

      sendBtn.disabled = true;
      sendBtn.innerHTML = '<span>Anchor Grounding...</span> <div class="pulse-dot"></div>';

      try {
        const response = await fetch('/webhook/imessage', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sender_name: senderName,
            relationship: relationship,
            raw_message: rawMessage,
            anxiety_detected: anxietyDetected,
            anxiety_score: anxietyScore
          })
        });

        const data = await response.json();
        if (data.event_id) {
          lastEventId = data.event_id;
        }
        updateBedsideDisplay(data, avatar);
      } catch (err) {
        alert("Failed to communicate with Anchor server: " + err);
      } finally {
        sendBtn.disabled = false;
        sendBtn.innerHTML = '<span>Send Incoming iMessage</span> <span>➔</span>';
      }
    }

    function updateBedsideDisplay(data, avatar) {
      if (data.event_id) {
        lastEventId = data.event_id;
      }

      document.getElementById('avatarDisplay').textContent = avatar || "💬";
      document.getElementById('senderNameDisplay').textContent = data.sender_name;
      document.getElementById('relationshipDisplay').textContent = data.relationship ? `Your ${data.relationship}` : "Family Member";
      document.getElementById('groundedDisplay').textContent = `"${data.grounded_message}"`;
      document.getElementById('rawInterceptedDisplay').textContent = `"${data.raw_message}"`;

      // Update Caregiver auto-reply receipt
      const receiptText = document.getElementById('caregiverReceiptText');
      if (data.caregiver_reply) {
        receiptText.textContent = data.caregiver_reply;
      }

      const wave = document.getElementById('audioWave');
      const statusText = document.getElementById('audioStatusText');
      const replayBtn = document.getElementById('replayBtn');
      const audioEl = document.getElementById('audioElement');

      if (data.audio_generated && data.audio_url) {
        const isNewAudio = (data.audio_url !== currentAudioUrl);
        currentAudioUrl = data.audio_url;
        wave.style.opacity = '1';
        statusText.textContent = "Rachel's voice intervention playing...";
        statusText.style.color = 'var(--accent)';
        replayBtn.style.display = 'inline-flex';

        // Avoid interrupting or restarting if this exact audio stream is already playing
        if (isNewAudio || audioEl.paused) {
          audioEl.src = data.audio_url;
          audioEl.currentTime = 0;
          audioEl.play().catch(e => {
            console.warn("Autoplay blocked by browser policy, click 'Play Voice' button:", e);
            statusText.textContent = "Voice ready (Click Play Voice to listen)";
          });
        }

        audioEl.onended = () => {
          wave.style.opacity = '0.3';
          statusText.textContent = "Voice intervention completed";
          statusText.style.color = 'var(--text-muted)';
        };
      } else {
        currentAudioUrl = null;
        wave.style.opacity = '0.3';
        statusText.textContent = "Calm state • Message delivered visually";
        statusText.style.color = 'var(--calm)';
        replayBtn.style.display = 'none';
      }
    }

    function replayAudio() {
      const audioEl = document.getElementById('audioElement');
      if (currentAudioUrl) {
        audioEl.src = currentAudioUrl;
        audioEl.currentTime = 0;
        audioEl.play();
        document.getElementById('audioWave').style.opacity = '1';
        document.getElementById('audioStatusText').textContent = "Replaying voice intervention...";
      }
    }

    // Background poller for live external webhooks (e.g. from Photon or curl)
    setInterval(async () => {
      try {
        const res = await fetch('/api/events/latest');
        const data = await res.json();
        if (data.event && data.event.event_id && data.event.event_id !== lastEventId) {
          lastEventId = data.event.event_id;
          updateBedsideDisplay(data.event, "💬");
        }
      } catch (e) {}
    }, 2000);
  </script>
</body>
</html>
"""
    rendered_html = html_template.replace("<!-- LIVE_PHONE_BADGE -->", live_badge_html)
    return HTMLResponse(content=rendered_html)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
