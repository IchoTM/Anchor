import os
import re
import json
import asyncio
from typing import Any, Dict, Tuple
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types

load_dotenv(find_dotenv(), override=True)

SYSTEM_INSTRUCTION = """You are Anchor, a real-time cognitive security and grounding assistant for an elderly person experiencing dementia (Eleanor).

Elderly individuals with dementia are exceptionally vulnerable to fraud, coercion, and sudden panic. They lack the cognitive capacity to verify urgency or identity. However, they also desperately need warmth and connection with their loved ones. Unnecessarily blocking genuine messages from children or grandchildren causes severe emotional distress.

For every incoming message, perform TWO tasks:

1. SECURITY SCAN (Scam / Malicious Detection):
Evaluate if the incoming message is malicious, fraudulent, predatory, or suspicious.

Flag as MALICIOUS (is_malicious = true) ONLY if the message involves:
- Emergency extortion or Grandparent Scams ("I'm in jail", "I was in an accident", "need bail money", "kidnapped")
- Demands for money transfers, wire transfers (Western Union, MoneyGram, Zelle, Venmo), cash, or cryptocurrency
- Gift card requests (Apple, Google Play, Target, Steam cards)
- Impersonation of law enforcement, government agencies (IRS, Social Security, Police, Court), or bank fraud departments threatening arrest, fines, or account suspension
- Demands for passwords, PINs, OTP codes, or Social Security numbers
- High-pressure secrecy ("Don't tell mom", "keep this secret", "hurry before time runs out")
- Suspicious external phishing links

Flag as SAFE (is_malicious = false) for:
- Legitimate family conversations, check-ins, affection, visits, schedule updates, medication reminders, or friendly notes.
- Family members letting Eleanor know they got a new phone or a new phone number (e.g., "Hey Ma, it's Jenna, I got a new phone...", "Hi Grandma, it's Tommy from my new number"). AS LONG AS they do NOT ask for money, cards, codes, or urgent payments, this is completely SAFE. Do NOT flag loving family updates as malicious!
- Friendly greetings, warm check-ins, or daily updates even if the sender's phone number is unverified or new, provided there is NO financial demand, threat, or extortion.

EXAMPLES OF EVALUATION:
- "Hey Ma, It's Jenna, I got a new phone and they gave me a new phone number for some reason. I love you" -> SAFE (is_malicious = false).
- "Hi Grandma, it's Tommy. I got a new phone! Coming by to see you Sunday." -> SAFE (is_malicious = false).
- "Hi Mum, I got a new phone. I lost my wallet and need you to wire $400 right now for rent" -> MALICIOUS (is_malicious = true).
- "URGENT: This is Chase Bank fraud department. Your account is locked. Wire $2000 to verify" -> MALICIOUS (is_malicious = true).
- "Grandma, I'm in jail and need bail money, please don't tell mom" -> MALICIOUS (is_malicious = true).
- "Thinking of you today Grandma, hope you had a good lunch!" -> SAFE (is_malicious = false).

2. PATIENT GROUNDING (Only if safe):
If the message is safe, rewrite it in a calm, gentle, patient-friendly tone for Eleanor:
- CRITICAL: ALWAYS state BOTH the sender's actual NAME and their RELATIONSHIP to Eleanor together.
  Examples:
  - "Your son, David, sent you a message..."
  - "Your granddaughter, Emma, wants you to know..."
  - "Your daughter, Sarah, is checking in..."
  - "Your grandson, Alex, just let you know..."
- DO NOT just say "Your son" or "Your granddaughter" without their name if their name is available! For someone with dementia, hearing both their specific name and their relationship anchors their memory and provides immediate reassurance.
- If the sender introduced themselves in the text (e.g., "It's Jenna", "It's Tommy"), always recognize their name and relationship ("Your daughter, Jenna", "Your grandson, Tommy").
- If the contact is known, use both their name and relationship.
- Rephrase clearly in warm, comforting, reassuring language.
- Keep it brief (1 to 3 sentences). Do not overwhelm her with words.
- Do NOT sound like an AI. Speak as a gentle bedside narrator.

OUTPUT FORMAT:
You MUST respond strictly with valid JSON conforming to this schema:
{
  "is_malicious": boolean,
  "malicious_reason": string, // Detailed explanation of why it was flagged if malicious; empty string if safe
  "grounded_message": string  // Gentle grounded message for Eleanor if safe; empty string if malicious
}
"""

# Heuristic patterns for rapid emergency scam detection
SCAM_HEURISTIC_PATTERN = re.compile(
    r"\b(bail\s*money|in\s*jail|wire\s*(?:money|\$)|western\s*union|gift\s*cards?|send\s*cash|"
    r"arrest\s*warrant|irs\s*audit|social\s*security\s*suspended|buy\s*crypto|bitcoin\s*atm|"
    r"send\s*\$\d+|send\s*money\s*right\s*now|don'?t\s*tell\s*mom|keep\s*this\s*(?:a\s*)?secret)\b",
    re.IGNORECASE,
)

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


def _heuristic_scam_check(raw_message: str) -> Tuple[bool, str]:
    """Fast-path heuristic scanner for high-risk predatory phrases."""
    match = SCAM_HEURISTIC_PATTERN.search(raw_message)
    if match:
        matched_phrase = match.group(0)
        return True, f"High-risk scam trigger detected: '{matched_phrase}'. Matches emergency financial extortion patterns."
    return False, ""


def _parse_gemini_json(raw_text: str) -> Dict[str, Any]:
    """Robustly extracts JSON from Gemini output."""
    cleaned = raw_text.strip()

    # Strip markdown code blocks if present
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return {
                "is_malicious": bool(data.get("is_malicious", False)),
                "malicious_reason": str(data.get("malicious_reason", "")).strip(),
                "grounded_message": str(data.get("grounded_message", "")).strip(),
            }
    except Exception:
        pass

    # Fallback regex search for JSON object inside response
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if isinstance(data, dict):
                return {
                    "is_malicious": bool(data.get("is_malicious", False)),
                    "malicious_reason": str(data.get("malicious_reason", "")).strip(),
                    "grounded_message": str(data.get("grounded_message", "")).strip(),
                }
        except Exception:
            pass

    return {
        "is_malicious": False,
        "malicious_reason": "",
        "grounded_message": cleaned,
    }


def _generate_sync(client: genai.Client, model: str, contents: str, config: types.GenerateContentConfig) -> str:
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    return getattr(response, "text", "") or ""


async def _call_gemini_analysis(client: genai.Client, model: str, user_prompt: str) -> Dict[str, Any]:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.2,
        response_mime_type="application/json",
    )

    raw_text = await asyncio.wait_for(
        asyncio.to_thread(_generate_sync, client, model, user_prompt, config),
        timeout=8.0,
    )
    return _parse_gemini_json(raw_text)


def _build_fallback_grounding(sender_name: str, relationship: str, raw_message: str) -> str:
    """Safe rule-based grounding message when the LLM service is unavailable."""
    name_clean = (sender_name or "").strip()
    if name_clean.lower().startswith("your "):
        who = name_clean
    elif relationship and relationship.lower() not in name_clean.lower():
        who = f"Your {relationship.lower()}, {name_clean},"
    elif name_clean:
        who = name_clean
    else:
        who = "A loved one"

    return (
        f"Hi Eleanor. {who} sent you a message: "
        f"\"{raw_message}\". Everything is alright."
    )


async def analyze_and_ground_message(
    sender_name: str, relationship: str, raw_message: str, sender_handle: str = ""
) -> Dict[str, Any]:
    """
    Analyzes an incoming message for predatory/scam behavior and generates patient grounding.
    Returns:
        {
            "is_malicious": bool,
            "malicious_reason": str,
            "grounded_message": str
        }
    """
    global _cached_working_model

    # 1. Fast-path heuristic check
    heuristic_flag, heuristic_reason = _heuristic_scam_check(raw_message)
    if heuristic_flag:
        print(f"[Anchor Security] Heuristic flagged message as malicious: {heuristic_reason}")
        return {
            "is_malicious": True,
            "malicious_reason": heuristic_reason,
            "grounded_message": "",
        }

    # 2. Query Gemini
    client = get_genai_client()
    sender_identifier = sender_name or sender_handle or "Unknown Sender"
    user_prompt = f'Sender Name: "{sender_name}" | Sender Identifier: "{sender_identifier}" | Relationship: "{relationship}" | Message: "{raw_message}"'

    preferred_model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
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

    for model_name in candidates:
        try:
            print(f"[Anchor Security] Scanning message with Gemini model {model_name}...")
            result = await _call_gemini_analysis(client, model_name, user_prompt)
            if result and (result.get("grounded_message") or result.get("is_malicious")):
                _cached_working_model = model_name
                print(f"[Anchor Security] Gemini scan complete. Malicious={result.get('is_malicious')}")
                return result
        except Exception as exc:
            err_msg = str(exc) or repr(exc)
            print(f"[Anchor Warning] Gemini model {model_name} failed during scan: {err_msg}")
            continue

    # Fallback if Gemini models fail
    return {
        "is_malicious": False,
        "malicious_reason": "",
        "grounded_message": _build_fallback_grounding(sender_name, relationship, raw_message),
    }


async def generate_grounding_message(
    sender_name: str, relationship: str, raw_message: str
) -> str:
    """Backwards compatibility wrapper for generating grounding message text."""
    result = await analyze_and_ground_message(sender_name, relationship, raw_message)
    return result.get("grounded_message", "") or _build_fallback_grounding(sender_name, relationship, raw_message)
