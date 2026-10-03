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

        os.environ["GEMINI_API_KEY"] = api_key
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"

        _client = genai.Client(api_key=api_key, vertexai=False)
    return _client


def _find_available_model(client: genai.Client) -> str:
    """Find a supported 3.x flash model from client.models.list()."""
    candidate_names = []
    try:
        for m in client.models.list():
            name = getattr(m, "name", "")
            base_name = name.removeprefix("models/")
            candidate_names.append(base_name)

        for name in candidate_names:
            if "3.8-flash" in name:
                return name
        for name in candidate_names:
            if "3.5-flash" in name:
                return name
        for name in candidate_names:
            if "flash" in name and "gemini" in name:
                return name
        for name in candidate_names:
            if "gemini" in name:
                return name
        if candidate_names:
            return candidate_names[0]
    except Exception as exc:
        print(f"[Anchor Warning] Could not list models: {exc}")

    return "gemini-3.8-flash"


async def _call_gemini(client: genai.Client, model: str, user_prompt: str) -> str:
    """Attempt call via Interactions API first, fallback to generate_content."""
    full_prompt = f"{SYSTEM_INSTRUCTION}\n\nInput: {user_prompt}"

    # Try Interactions API if available on client
    if hasattr(client, "aio") and hasattr(client.aio, "interactions"):
        try:
            interaction = await client.aio.interactions.create(
                model=model,
                input=full_prompt,
            )
            if interaction and getattr(interaction, "output_text", None):
                return interaction.output_text
        except Exception:
            pass

    if hasattr(client, "interactions"):
        try:
            interaction = await asyncio.to_thread(
                client.interactions.create,
                model=model,
                input=full_prompt,
            )
            if interaction and getattr(interaction, "output_text", None):
                return interaction.output_text
        except Exception:
            pass

    # Fallback to generate_content
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.3,
    )

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

    return getattr(response, "text", "") or ""


async def generate_grounding_message(
    sender_name: str, relationship: str, raw_message: str
) -> str:
    global _cached_working_model

    client = get_genai_client()
    user_prompt = f'Sender: "{sender_name}" | Relationship: "{relationship}" | Message: "{raw_message}"'

    env_model = os.getenv("GEMINI_MODEL")
    candidates = []
    if _cached_working_model:
        candidates.append(_cached_working_model)
    if env_model and env_model not in candidates:
        candidates.append(env_model)
    for model_name in ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-pro-preview"]:
        if model_name not in candidates:
            candidates.append(model_name)

    last_error: Exception | None = None

    for model_name in candidates:
        try:
            text = await _call_gemini(client, model_name, user_prompt)
            if text:
                _cached_working_model = model_name
                return text
        except Exception as exc:
            last_error = exc
            continue

    # If predefined models fail, dynamically query available models from the account
    try:
        discovered_model = await asyncio.to_thread(_find_available_model, client)
        if discovered_model not in candidates:
            text = await _call_gemini(client, discovered_model, user_prompt)
            if text:
                _cached_working_model = discovered_model
                return text
    except Exception as exc:
        last_error = exc

    if last_error:
        raise last_error

    return ""
