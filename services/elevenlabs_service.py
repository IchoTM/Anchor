import os
import asyncio
from dotenv import load_dotenv, find_dotenv

try:
    from elevenlabs.client import ElevenLabs
except ImportError:
    from elevenlabs import ElevenLabs

load_dotenv(find_dotenv(), override=True)

_client: ElevenLabs | None = None

DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
DEFAULT_MODEL_ID = "eleven_multilingual_v2"


def get_elevenlabs_client() -> ElevenLabs:
    global _client
    if _client is None:
        load_dotenv(find_dotenv(), override=True)
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is not set in your .env file!")
        api_key = api_key.strip().strip("'\"")
        _client = ElevenLabs(api_key=api_key)
    return _client


def _generate_audio_sync(client: ElevenLabs, voice_id: str, model_id: str, text: str) -> bytes:
    if hasattr(client, "text_to_speech") and hasattr(client.text_to_speech, "convert"):
        result = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=model_id,
        )
    elif hasattr(client, "generate"):
        result = client.generate(
            text=text,
            voice=voice_id,
            model=model_id,
        )
    else:
        raise AttributeError("ElevenLabs client does not expose a supported text-to-speech conversion method.")

    if isinstance(result, (bytes, bytearray)):
        return bytes(result)

    chunks = []
    for chunk in result:
        if chunk:
            chunks.append(chunk)
    return b"".join(chunks)


async def generate_speech_audio(text: str) -> bytes:
    """Generate speech audio bytes for the given text using ElevenLabs."""
    client = get_elevenlabs_client()
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID)
    model_id = os.getenv("ELEVENLABS_MODEL_ID", DEFAULT_MODEL_ID)

    return await asyncio.to_thread(_generate_audio_sync, client, voice_id, model_id, text)
