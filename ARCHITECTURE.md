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

## Security Architecture: Dual-Tier "Safety Hold" Protocol
To protect vulnerable patients from financial exploitation while avoiding the pitfalls of blunt blackholing or leaky error warnings:

1. **Patient Shielding (Bedside Zero-Disturbance):**
   Any message flagged as predatory, financial, or high-pressure is immediately suppressed from Eleanor's bedside station. No chime rings, no text displays, and no speech audio plays.

2. **Tier 1: Unverified / Stranger Numbers $\rightarrow$ Silent Blackhole:**
   - Messages from unknown numbers that fail safety checks are silently dropped.
   - **Zero In-Band Feedback:** No error message is returned (`caregiver_reply = None`). This stops bad actors from probing AI detection boundaries or learning that an automated filter exists.

3. **Tier 2: Known Family Contacts $\rightarrow$ Empathetic "Caregiver Review" Hold:**
   - If an authorized contact (e.g. Sarah or Alex) sends a message containing sensitive financial or urgent requests:
   - Instead of an accusatory scam alert, Anchor replies with a neutral policy hold receipt:
     *"Anchor Notice: For Eleanor's peace of mind, messages concerning sensitive actions, payments, or urgent requests are held in Caregiver Review and will not appear on her bedside display. If this is Sarah, please connect with Eleanor or her primary caregiver by phone."*
   - **Solves False Positives:** Legitimate family members immediately understand why the text was held and are directed to call.
   - **Neutralizes Compromised Devices:** An attacker who compromised Sarah's device learns that Eleanor cannot be reached via text, cannot bypass the lock without a live phone call, and gains no intel on AI filter rules.

4. **Out-of-Band Caregiver Telemetry:**
   All flagged events are transmitted with full context to the caregiver dashboard (`/api/caregiver/alerts`) for real-time monitoring and intervention.

## Rules for Aider
- Write modular, asynchronous Python (`async def`).
- Use Pydantic models for all incoming and outgoing API payloads.
- Keep route handlers minimal in `main.py`; put business logic in `services/`.
- Use `python-dotenv` to load API keys from a `.env` file.
- Do not write placeholder `pass` or `TODO` blocks for core logic—implement the actual API calls.
