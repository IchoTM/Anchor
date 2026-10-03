import os
import json
import uuid
import time
import asyncio
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv, find_dotenv

from services.gemini_service import generate_grounding_message
from services.elevenlabs_service import generate_speech_audio
from services.presage_service import evaluate_anxiety

load_dotenv(find_dotenv(), override=True)

DEFAULT_CONTACTS: Dict[str, Dict[str, str]] = {
    "+15551234567": {"name": "Alex", "relationship": "Grandson"},
    "+15550192": {"name": "Grandpa", "relationship": "Husband"},
    "Alex": {"name": "Alex", "relationship": "Grandson"},
    "Sarah": {"name": "Sarah", "relationship": "Daughter"},
    "David": {"name": "David", "relationship": "Son"},
    "Maria": {"name": "Maria", "relationship": "Caregiver"},
    "Grandpa": {"name": "Grandpa", "relationship": "Husband"},
    "Dr. Chen": {"name": "Dr. Chen", "relationship": "Doctor"},
}

# In-memory storage for audio bytes and processed message events
_audio_store: Dict[str, bytes] = {}
_recent_events: deque = deque(maxlen=50)


def get_stored_audio(audio_id: str) -> Optional[bytes]:
    """Retrieve raw audio bytes by audio ID."""
    return _audio_store.get(audio_id)


def get_recent_events(limit: int = 15) -> List[Dict[str, Any]]:
    """Retrieve the most recent processed Anchor events."""
    events = list(_recent_events)
    events.reverse()
    return events[:limit]


def get_latest_event() -> Optional[Dict[str, Any]]:
    """Retrieve the single most recent processed event."""
    return _recent_events[-1] if _recent_events else None


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
    webhook payload structures.
    """
    contacts = get_contact_directory()

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
        sender_handle = (
            sender_raw.get("identifier")
            or sender_raw.get("handle")
            or sender_raw.get("phone_number")
            or sender_raw.get("id")
            or ""
        )
        relationship = sender_raw.get("relationship") or ""
    elif isinstance(sender_raw, str):
        sender_name = sender_raw
        sender_handle = sender_raw

    # Fallback to top-level fields
    if not sender_name and "sender_name" in data:
        sender_name = str(data["sender_name"])
    if not sender_handle and "identifier" in data:
        sender_handle = str(data["identifier"])
    if not sender_handle and "handle" in data:
        sender_handle = str(data["handle"])

    # 3. Resolve relationship from metadata or directory lookup
    if not relationship and isinstance(metadata, dict):
        relationship = metadata.get("relationship", "")

    if not relationship:
        if sender_handle in contacts and "relationship" in contacts[sender_handle]:
            contact_info = contacts[sender_handle]
            relationship = contact_info.get("relationship", "")
            if not sender_name:
                sender_name = contact_info.get("name", sender_handle)
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
    payload: Dict[str, Any], force_audio: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Full pipeline:
    1. Parse incoming message and sender details.
    2. Ground message via Gemini.
    3. Evaluate patient anxiety state via Presage biometric emotional sensing.
    4. Gate ElevenLabs TTS: trigger voice generation only if anxiety is detected.
    5. Cache event for the Bedside Tablet display.
    """
    sender_name, relationship, raw_message = parse_photon_payload(payload)
    print(f"[Anchor Pipeline] Parsed: Sender='{sender_name}', Relation='{relationship}', Message='{raw_message}'")

    # Evaluate Presage anxiety state
    anxiety_detected, anxiety_score, biometrics = evaluate_anxiety(payload)
    print(f"[Anchor Presage] Anxiety Detected: {anxiety_detected} (Score: {anxiety_score:.2f})")

    # Determine whether audio should be generated (force override or anxiety-gated)
    should_generate_audio = anxiety_detected if force_audio is None else force_audio

    print("[Anchor Pipeline] Calling Gemini for grounding...")
    grounded_message = await generate_grounding_message(
        sender_name=sender_name,
        relationship=relationship,
        raw_message=raw_message,
    )
    print(f"[Anchor Pipeline] Grounded text: '{grounded_message}'")

    audio_generated = False
    audio_id = None
    audio_url = None

    if should_generate_audio:
        try:
            print("[Anchor Pipeline] Anxiety detected! Triggering ElevenLabs voice intervention...")
            audio_bytes = await asyncio.wait_for(
                generate_speech_audio(grounded_message),
                timeout=10.0,
            )
            if audio_bytes and len(audio_bytes) > 0:
                audio_generated = True
                audio_id = str(uuid.uuid4())
                _audio_store[audio_id] = audio_bytes
                audio_url = f"/audio/{audio_id}"
                print(f"[Anchor TTS]: Generated {len(audio_bytes)} bytes of speech audio (ID: {audio_id}).")
        except asyncio.TimeoutError:
            print("[Anchor Warning] ElevenLabs TTS timed out after 10.0s, proceeding without audio.")
        except Exception as exc:
            print(f"[Anchor Warning] TTS generation failed: {exc}")
    else:
        print("[Anchor Pipeline] Patient state is calm. Voice synthesis gated off.")

    result_event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "sender_name": sender_name,
        "relationship": relationship,
        "raw_message": raw_message,
        "grounded_message": grounded_message,
        "anxiety_detected": anxiety_detected,
        "anxiety_score": anxiety_score,
        "biometrics": biometrics,
        "audio_generated": audio_generated,
        "audio_id": audio_id,
        "audio_url": audio_url,
    }

    _recent_events.append(result_event)
    return result_event
