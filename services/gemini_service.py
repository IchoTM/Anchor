import os
import asyncio
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types

load_dotenv(find_dotenv(), override=True)

SYSTEM_INSTRUCTION = """You are Anchor, an empathetic and highly patient cognitive assistant for an elderly person experiencing dementia. Your job is to intercept incoming text messages from their family members and rewrite them to provide gentle, grounding context.

People with dementia lose context. A text saying "I'll be there in 10 mins" can cause extreme panic because they don't remember who is texting or where they are supposed to be. 

Your goals:
1. Always state WHO the sender is and their RELATIONSHIP to the user.
2. Rephrase the message in a calm, clear, and warm tone.
3. Keep it brief. Do not overwhelm them with words.
4. Do NOT sound like an AI. Do not say "I am an AI assistant." Speak in the third person as a gentle narrator, or format it as a clear notification.

EXAMPLES:
Input: Sender: "Alex" | Relationship: "Grandson" | Message: "I'll be there in 10 mins!"
Output: "Hi Grandma. Your grandson, Alex, just sent you a message. He wants you to know that he is coming over and will be at your house in 10 minutes."

Input: Sender: "Sarah" | Relationship: "Daughter" | Message: "Did you take your pills? Call me."
Output: "Your daughter, Sarah, is checking in on you. She wants to know if you have taken your medication today, and she asked if you could give her a phone call."""

_client: genai.Client | None = None


def get_genai_client() -> genai.Client:
    global _client
    if _client is None:
        load_dotenv(find_dotenv(), override=True)
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in your .env file!")
        api_key = api_key.strip().strip("'\"")

        os.environ["GEMINI_API_KEY"] = api_key
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"

        _client = genai.Client(api_key=api_key, vertexai=False)
    return _client


async def generate_grounding_message(
    sender_name: str, relationship: str, raw_message: str
) -> str:
    client = get_genai_client()
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    user_prompt = f'Sender: "{sender_name}" | Relationship: "{relationship}" | Message: "{raw_message}"'

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.3,
    )

    try:
        if hasattr(client, "aio") and hasattr(client.aio, "models"):
            response = await client.aio.models.generate_content(
                model=model,
                contents=user_prompt,
                config=config,
            )
        else:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=user_prompt,
                config=config,
            )
    except Exception as exc:
        if model != "gemini-1.5-flash":
            fallback_model = "gemini-1.5-flash"
            if hasattr(client, "aio") and hasattr(client.aio, "models"):
                response = await client.aio.models.generate_content(
                    model=fallback_model,
                    contents=user_prompt,
                    config=config,
                )
            else:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=fallback_model,
                    contents=user_prompt,
                    config=config,
                )
        else:
            raise exc

    return response.text or ""
