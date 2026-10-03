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
_cached_working_model: str | None = None


def get_genai_client() -> genai.Client:
    global _client
    if _client is None:
        load_dotenv(find_dotenv(), override=True)
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in your .env file!")
        api_key = api_key.strip().strip("'\"")

        _client = genai.Client(api_key=api_key, vertexai=False)
    return _client


def _generate_sync(client: genai.Client, model: str, contents: str, config: types.GenerateContentConfig) -> str:
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    return getattr(response, "text", "") or ""


async def _call_gemini_single(client: genai.Client, model: str, user_prompt: str) -> str:
    full_prompt = f"{SYSTEM_INSTRUCTION}\n\nInput: {user_prompt}"

    # Try Interactions API first if supported
    if hasattr(client, "interactions"):
        try:
            interaction = await asyncio.wait_for(
                asyncio.to_thread(
                    client.interactions.create,
                    model=model,
                    input=full_prompt,
                ),
                timeout=10.0,
            )
            if interaction and getattr(interaction, "output_text", None):
                return interaction.output_text
        except Exception as exc:
            print(f"[Anchor Info] Interactions API call on {model} skipped ({exc}), trying generate_content...")

    # Fallback to generate_content
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.3,
    )

    return await asyncio.wait_for(
        asyncio.to_thread(_generate_sync, client, model, user_prompt, config),
        timeout=10.0,
    )


async def generate_grounding_message(
    sender_name: str, relationship: str, raw_message: str
) -> str:
    global _cached_working_model

    client = get_genai_client()
    user_prompt = f'Sender: "{sender_name}" | Relationship: "{relationship}" | Message: "{raw_message}"'

    preferred_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    candidates = []
    if _cached_working_model:
        candidates.append(_cached_working_model)
    if preferred_model not in candidates:
        candidates.append(preferred_model)
    for fallback in ["gemini-3.8-flash", "gemini-3.5-flash-lite"]:
        if fallback not in candidates:
            candidates.append(fallback)

    last_error: Exception | None = None

    for model_name in candidates:
        try:
            print(f"[Anchor Gemini] Querying model {model_name}...")
            text = await _call_gemini_single(client, model_name, user_prompt)
            if text:
                _cached_working_model = model_name
                print(f"[Anchor Gemini] Success using model {model_name}")
                return text
        except Exception as exc:
            last_error = exc
            print(f"[Anchor Warning] Gemini model {model_name} failed: {exc}")
            continue

    if last_error:
        raise last_error

    return ""
