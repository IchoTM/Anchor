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

## Rules for Aider
- Write modular, asynchronous Python (`async def`).
- Use Pydantic models for all incoming and outgoing API payloads.
- Keep route handlers minimal in `main.py`; put business logic in `services/`.
- Use `python-dotenv` to load API keys from a `.env` file.
- Do not write placeholder `pass` or `TODO` blocks for core logic—implement the actual API calls.
