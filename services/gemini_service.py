import os
import re
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
5. Provide ONLY the rephrased notification message. Do NOT repeat or echo "Input:" or "Output:" headers.

EXAMPLES:
Input: Sender: "Alex" | Relationship: "Grandson" | Message: "I'll be there in 10 mins!"
Output: "Hi Grandma. Your grandson, Alex, just sent you a message. He wants you to know that he is coming over and will be at your house in 10 minutes."

Input: Sender: "Sarah" | Relationship: "Daughter" | Message: "Did you take your pills? Call me."
Output: "Your daughter, Sarah, is checking in on you. She wants to know if you have taken your medication today, and she asked if you could give her a phone call." """

_client: genai.Client | None = None
_cached_working_model: str | None = None


def get_genai_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in your .env file!")
        api_key = api_key.strip().strip("'\"")

        _client = genai.Client(api_key=api_key, vertexai=False)
    return _client


def _clean_grounded_output(text: str) -> str:
    """Strips any echoed prompt headers (Input:/Output:) from the generated text."""
    cleaned = text.strip()
    if "Output:" in cleaned:
        cleaned = cleaned.split("Output:")[-1].strip()
    elif "output:" in cleaned.lower():
        pattern = re.compile(r"output\s*:\s*", re.IGNORECASE)
        parts = pattern.split(cleaned)
        if len(parts) > 1:
            cleaned = parts[-1].strip()

    # Remove enclosing quotes if the model wrapped the response in quotes
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()

    return cleaned


def _generate_sync(client: genai.Client, model: str, contents: str, config: types.GenerateContentConfig) -> str:
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    return getattr(response, "text", "") or ""


async def _call_gemini_single(client: genai.Client, model: str, user_prompt: str) -> str:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.3,
    )

    raw_text = await asyncio.wait_for(
        asyncio.to_thread(_generate_sync, client, model, user_prompt, config),
        timeout=8.0,
    )
    return _clean_grounded_output(raw_text)


def _build_fallback_grounding(sender_name: str, relationship: str, raw_message: str) -> str:
    """Provides a safe, rule-based grounding message when the LLM service is unavailable."""
    relation_text = f", your {relationship.lower()}," if relationship else ""
    return (
        f"Hi. {sender_name}{relation_text} sent you a message: "
        f"\"{raw_message}\". Everything is alright."
    )


async def generate_grounding_message(
    sender_name: str, relationship: str, raw_message: str
) -> str:
    global _cached_working_model

    client = get_genai_client()
    user_prompt = f'Sender: "{sender_name}" | Relationship: "{relationship}" | Message: "{raw_message}"'

    preferred_model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")

    # Priority candidate pool: tested active models first
    candidate_pool = [
        preferred_model,
        "gemini-flash-lite-latest",
        "gemini-flash-latest",
        "gemini-2.5-flash-lite",
    ]

    candidates: list[str] = []
    if _cached_working_model and _cached_working_model not in candidates:
        candidates.append(_cached_working_model)

    for cand in candidate_pool:
        if cand not in candidates:
            candidates.append(cand)

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
            err_msg = str(exc) or repr(exc)
            print(f"[Anchor Warning] Gemini model {model_name} failed: {err_msg}")
            continue

    # Fallback to discovering available models if hardcoded candidates fail
    try:
        print("[Anchor Gemini] Querying ModelService for available models...")
        models_page = await asyncio.to_thread(client.models.list)
        for m in models_page:
            model_id = getattr(m, "name", None) or getattr(m, "id", None)
            if model_id and ("gemini" in model_id.lower() or "flash" in model_id.lower()):
                clean_id = model_id.replace("models/", "")
                if clean_id in candidates:
                    continue
                try:
                    print(f"[Anchor Gemini] Attempting discovered model {clean_id}...")
                    text = await _call_gemini_single(client, clean_id, user_prompt)
                    if text:
                        _cached_working_model = clean_id
                        return text
                except Exception:
                    continue
    except Exception as list_exc:
        print(f"[Anchor Warning] Could not list models: {list_exc}")

    print("[Anchor Warning] All Gemini models exhausted. Using resilient fallback grounding message.")
    return _build_fallback_grounding(sender_name, relationship, raw_message)
