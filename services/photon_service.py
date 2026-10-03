import os
import json
from typing import Any, Dict, Tuple
from dotenv import load_dotenv, find_dotenv

from services.gemini_service import generate_grounding_message
from services.elevenlabs_service import generate_speech_audio

load_dotenv(find_dotenv(), override=True)

# Default contact directory mapping phone numbers, handles, or names to relationship metadata
DEFAULT_CONTACTS: Dict[str, Dict[str, str]] = {
    "+15551234567": {"name": "Alex", "relationship": "Grandson"},
    "Alex": {"name": "Alex", "relationship": "Grandson"},
    "Sarah": {"name": "Sarah", "relationship": "Daughter"},
    "David": {"name": "David", "relationship": "Son"},
    "Maria": {"name": "Maria", "relationship": "Caregiver"},
    "Dr. Chen": {"name": "Dr. Chen", "relationship": "Doctor"},
}


def get_contact_directory() -> Dict[str, Dict[str, str]]:
    """Loads contact mappings with optional overrides from PHOTON_CONTACTS_JSON."""
    contacts = dict(DEFAULT_CONTACTS)
    custom_contacts_raw = os.getenv("PHOTON_CONTACTS_JSON")
    if custom_contacts_raw:
        try:
            custom_contacts = json.loads(custom_contacts_raw)
            if isinstance(custom_contacts, dict):
                contacts.update(custom_contacts)
        except Exception as exc:
            print(f"[Anchor Warning] Failed to parse PHOTON_CONTACTS_JSON: {exc}")
    return contacts


def parse_photon_payload(payload: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Extracts (sender_name, relationship, raw_message) from various Photon / Spectrum
    webhook payload structures (events, spaces, nested message objects, or direct fields).
    """
    contacts = get_contact_directory()

    # Unwrap event wrapper if present (e.g. {"event": "message.received", "data": {...}})
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload

    # 1. Extract message body / raw text
    raw_message = ""
    if "raw_message" in data and isinstance(data["raw_message"], str):
        raw_message = data["raw_message"]
    elif "text" in data and isinstance(data["text"], str):
        raw_message = data["text"]
    elif "body" in data and isinstance(data["body"], str):
        raw_message = data["body"]
    elif isinstance(data.get("message"), dict):
        msg_obj = data["message"]
        raw_message = msg_obj.get("text") or msg_obj.get("body") or msg_obj.get("raw_message") or ""
    elif isinstance(data.get("message"), str):
        raw_message = data["message"]

    # 2. Extract sender info
    sender_raw = data.get("sender") or data.get("from") or data.get("author") or {}
    metadata = data.get("metadata") or {}

    sender_name = ""
    sender_handle = ""
    relationship = ""

    if isinstance(sender_raw, dict):
        sender_name = sender_raw.get("name") or sender_raw.get("display_name") or ""
        sender_handle = sender_raw.get("handle") or sender_raw.get("phone_number") or sender_raw.get("id") or ""
        relationship = sender_raw.get("relationship") or ""
    elif isinstance(sender_raw, str):
        sender_name = sender_raw
        sender_handle = sender_raw

    # Fallback to top-level fields
    if not sender_name and "sender_name" in data:
        sender_name = str(data["sender_name"])
    if not sender_handle and "handle" in data:
        sender_handle = str(data["handle"])

    # 3. Resolve relationship from metadata or directory lookup
    if not relationship and isinstance(metadata, dict):
        relationship = metadata.get("relationship", "")

    if not relationship:
        # Check by handle
        if sender_handle in contacts and "relationship" in contacts[sender_handle]:
            contact_info = contacts[sender_handle]
            relationship = contact_info.get("relationship", "")
            if not sender_name:
                sender_name = contact_info.get("name", sender_handle)
        # Check by name
        elif sender_name in contacts and "relationship" in contacts[sender_name]:
            relationship = contacts[sender_name].get("relationship", "")

    # Clean up defaults
    if not sender_name:
        sender_name = sender_handle or "A family member"
    if not relationship:
        relationship = "Family Member"
    if not raw_message:
        raw_message = "(No text content)"

    return sender_name.strip(), relationship.strip(), raw_message.strip()


async def process_photon_message(
    payload: Dict[str, Any], generate_audio: bool = True
) -> Dict[str, Any]:
    """
    Full pipeline: Parse incoming Photon message, ground via Gemini,
    and optionally generate ElevenLabs speech audio.
    """
    sender_name, relationship, raw_message = parse_photon_payload(payload)

    # Ground message via Gemini
    grounded_message = await generate_grounding_message(
        sender_name=sender_name,
        relationship=relationship,
        raw_message=raw_message,
    )

    audio_generated = False
    if generate_audio:
        try:
            audio_bytes = await generate_speech_audio(grounded_message)
            if audio_bytes and len(audio_bytes) > 0:
                audio_generated = True
                print(f"[Anchor TTS]: Generated {len(audio_bytes)} bytes of speech audio.")
        except Exception as exc:
            print(f"[Anchor Warning] TTS generation failed: {exc}")

    return {
        "sender_name": sender_name,
        "relationship": relationship,
        "raw_message": raw_message,
        "grounded_message": grounded_message,
        "audio_generated": audio_generated,
    }
