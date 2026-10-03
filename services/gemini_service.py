import os
import asyncio
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

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

_client = None

def get_genai_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in your .env file!")
        # Modern google-genai client initialization
        _client = genai.Client(api_key=api_key)
    return _client

async def generate_grounding_message(
    sender_name: str, relationship: str, raw_message: str
) -> str:
    client = get_genai_client()
    user_prompt = f'Sender: "{sender_name}" | Relationship: "{relationship}" | Message: "{raw_message}"'

    # Run content generation using the modern client & config
    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.3,
        ),
    )

    return response.text or ""
