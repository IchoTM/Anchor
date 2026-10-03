import os
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.gemini_service import generate_grounding_message
from services.elevenlabs_service import generate_speech_audio

load_dotenv(find_dotenv(), override=True)

app = FastAPI(
    title="Anchor",
    description="Real-time iMessage dementia care assistant",
    version="0.1.0",
)


class IMessageWebhookRequest(BaseModel):
    sender_name: str = Field(..., description="Name of the sender")
    relationship: str = Field(..., description="Relationship to the recipient (e.g. Grandson, Daughter)")
    raw_message: str = Field(..., description="The original incoming message text")


class IMessageWebhookResponse(BaseModel):
    sender_name: str
    relationship: str
    raw_message: str
    grounded_message: str
    audio_generated: bool = Field(..., description="Success status indicating whether speech audio was generated")


@app.post("/webhook/imessage", response_model=IMessageWebhookResponse)
async def handle_imessage_webhook(payload: IMessageWebhookRequest):
    try:
        grounded_message = await generate_grounding_message(
            sender_name=payload.sender_name,
            relationship=payload.relationship,
            raw_message=payload.raw_message,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate grounding message: {exc}",
        ) from exc

    print(f"[Anchor Grounded Message]: {grounded_message}")

    audio_generated = False
    try:
        audio_bytes = await generate_speech_audio(grounded_message)
        if audio_bytes and len(audio_bytes) > 0:
            audio_generated = True
            print(f"[Anchor TTS]: Successfully generated {len(audio_bytes)} bytes of speech audio.")
    except Exception as exc:
        print(f"[Anchor Warning] Failed to generate speech audio: {exc}")

    return IMessageWebhookResponse(
        sender_name=payload.sender_name,
        relationship=payload.relationship,
        raw_message=payload.raw_message,
        grounded_message=grounded_message,
        audio_generated=audio_generated,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
