import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.gemini_service import generate_grounding_message

load_dotenv()

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

    return IMessageWebhookResponse(
        sender_name=payload.sender_name,
        relationship=payload.relationship,
        raw_message=payload.raw_message,
        grounded_message=grounded_message,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
