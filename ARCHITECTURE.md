# Project: Anchor (MHacks 2026)
# Mission: A real-time iMessage dementia care assistant.

## Tech Stack
- Backend: Python 3.12+ 
- Framework: FastAPI (with Uvicorn)
- APIs: 
  - Photon (Spectrum framework) for iMessage webhook interception.
  - Google Gemini API (`google-genai` SDK) for context analysis.
  - ElevenLabs API (`elevenlabs` SDK) for text-to-speech.
  - Presage SDK for biometric emotional sensing.

## The Core Loop
1. Photon intercepts an incoming iMessage to the patient.
2. The FastAPI server receives the webhook payload at `POST /webhook/imessage`.
3. The server queries the Gemini API with the message and a system prompt to "anchor" the patient.
4. If the frontend (Presage) detects high anxiety, the server triggers ElevenLabs to generate an MP3 of the Gemini response.

## Security Architecture: Cognitive Shield & Caregiver Dispatch

### 1. Zero-Disturbance Patient Bedside Shield
Any message flagged as predatory, financial extortion, or high-pressure is immediately suppressed from Eleanor's bedside station. No chime rings, no card displays, and no speech audio plays.

### 2. Dual-Tier Response Policy
- **Risky Message + Unverified / No Contact:**
  - **Zero Response to Sender:** `caregiver_reply = None`. Twilio/SMS sends `<Response></Response>` (empty). Bad actors receive silence and cannot probe AI filters or rules.
  - **Bedside Display:** Blocked.
  - **Active Caregiver Alert:** Dispatched immediately (SMS via Twilio, webhook, and dashboard).
- **Risky Message + Verified Contact:**
  - **Empathetic Safety Hold Receipt:** If a verified family contact sends a sensitive payment or urgent request (e.g. device stolen or account compromised):
    *"Anchor Notice: For Eleanor's peace of mind, messages concerning sensitive actions, payments, or urgent requests are held in Caregiver Review and will not appear on her bedside display. If this is [Name], please connect with Eleanor or her primary caregiver by phone."*
  - **Bedside Display:** Blocked.
  - **Active Caregiver Alert:** Dispatched immediately.

### 3. Family Members with New Numbers
- If a family member texts from an unverified or new phone number with a warm, genuine update (e.g., *"Hi Grandma, it's Tommy! Got a new phone, coming by Sunday"*):
  - The message contains NO extortion, wire requests, or panic prompts.
  - Gemini evaluates it as `is_malicious = False`.
  - Eleanor receives the calm grounded message on her tablet without false-positive blocks.

### 4. Active Out-of-Band Caregiver Alerting
When any malicious message is detected, `dispatch_caregiver_alert`:
- Dispatches an emergency SMS alert to `CAREGIVER_PHONE_NUMBER` via Twilio.
- Posts an alert to `CAREGIVER_WEBHOOK_URL` if configured.
- Logs full alert telemetry to `/api/caregiver/alerts` for real-time monitoring.

## Rules for Aider
- Write modular, asynchronous Python (`async def`).
- Use Pydantic models for all incoming and outgoing API payloads.
- Keep route handlers minimal in `main.py`; put business logic in `services/`.
- Use `python-dotenv` to load API keys from a `.env` file.
- Do not write placeholder `pass` or `TODO` blocks for core logic—implement the actual API calls.
