import os
import json
import uuid
import time
import asyncio
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv, find_dotenv

from services.gemini_service import analyze_and_ground_message
from services.elevenlabs_service import generate_speech_audio
from services.presage_service import evaluate_anxiety

load_dotenv(find_dotenv(), override=True)

DEFAULT_CONTACTS: Dict[str, Dict[str, str]] = {
    "+15551234567": {"name": "Alex", "relationship": "Grandson", "photo_url": "/static/avatars/alex.jpg"},
    "+15550192": {"name": "Grandpa", "relationship": "Husband", "photo_url": "/static/avatars/grandpa.jpg"},
    "Alex": {"name": "Alex", "relationship": "Grandson", "photo_url": "/static/avatars/alex.jpg"},
    "Sarah": {"name": "Sarah", "relationship": "Daughter", "photo_url": "/static/avatars/sarah.jpg"},
    "David": {"name": "David", "relationship": "Son", "photo_url": "/static/avatars/david.jpg"},
    "Maria": {"name": "Maria", "relationship": "Caregiver", "photo_url": "/static/avatars/maria.jpg"},
    "Grandpa": {"name": "Grandpa", "relationship": "Husband", "photo_url": "/static/avatars/grandpa.jpg"},
    "Dr. Chen": {"name": "Dr. Chen", "relationship": "Doctor", "photo_url": "/static/avatars/chen.jpg"},
}

# Heuristic name patterns mapped to photo files in /static/avatars/
AVATAR_NAME_PATTERNS: Dict[str, str] = {
    "alex": "/static/avatars/alex.jpg",
    "sarah": "/static/avatars/sarah.jpg",
    "david": "/static/avatars/david.jpg",
    "maria": "/static/avatars/maria.jpg",
    "grandpa": "/static/avatars/grandpa.jpg",
    "chen": "/static/avatars/chen.jpg",
}

# In-memory storage for audio bytes and safe message events for the bedside station
_audio_store: Dict[str, bytes] = {}
_recent_events: deque = deque(maxlen=50)


def get_stored_audio(audio_id: str) -> Optional[bytes]:
    """Retrieve raw audio bytes by audio ID."""
    return _audio_store.get(audio_id)


def get_recent_events(limit: int = 15) -> List[Dict[str, Any]]:
    """Retrieve recent safe processed Anchor events for the bedside display."""
    events = list(_recent_events)
    events.reverse()
    return events[:limit]


def get_latest_event() -> Optional[Dict[str, Any]]:
    """Retrieve the single most recent safe event."""
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


def resolve_contact_photo(
    sender_name: str,
    sender_handle: str,
    explicit_photo_url: Optional[str] = None,
) -> Optional[str]:
    """
    Resolves the avatar image URL for a sender using explicit payload values,
    the contact directory, or name pattern matching.
    """
    if explicit_photo_url and explicit_photo_url.strip():
        return explicit_photo_url.strip()

    contacts = get_contact_directory()

    # 1. Direct handle or name lookup in directory
    if sender_handle in contacts and contacts[sender_handle].get("photo_url"):
        return contacts[sender_handle]["photo_url"]
    if sender_name in contacts and contacts[sender_name].get("photo_url"):
        return contacts[sender_name]["photo_url"]

    # 2. Case-insensitive key lookup in directory
    sender_lower = (sender_name or "").lower().strip()
    handle_lower = (sender_handle or "").lower().strip()

    for key, info in contacts.items():
        if key.lower() in (sender_lower, handle_lower):
            if info.get("photo_url"):
                return info["photo_url"]

    # 3. Pattern match against known presets (e.g. "chen" -> chen.jpg)
    for pattern, path in AVATAR_NAME_PATTERNS.items():
        if pattern in sender_lower or pattern in handle_lower:
            return path

    return None


def parse_photon_payload(payload: Dict[str, Any]) -> Tuple[str, str, str, str, Optional[str]]:
    """
    Extracts (sender_name, relationship, raw_message, sender_handle, sender_photo_url)
    from various Photon / Spectrum / Twilio / SMS webhook payload structures.
    """
    contacts = get_contact_directory()

    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload

    # 1. Extract message body / raw text
    raw_message = ""
    if "raw_message" in data and isinstance(data["raw_message"], str):
        raw_message = data["raw_message"]
    elif "Body" in data and isinstance(data["Body"], str):
        raw_message = data["Body"]
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
    sender_raw = data.get("sender") or data.get("from") or data.get("From") or data.get("author") or {}
    metadata = data.get("metadata") or {}

    sender_name = ""
    sender_handle = ""
    relationship = ""
    explicit_photo = (
        data.get("sender_photo_url")
        or data.get("photo_url")
        or data.get("avatar_url")
    )

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
        if not explicit_photo:
            explicit_photo = sender_raw.get("photo_url") or sender_raw.get("avatar") or sender_raw.get("image")
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
    if not sender_handle and "From" in data:
        sender_handle = str(data["From"])

    # 3. Resolve relationship from metadata or directory lookup
    if not relationship and isinstance(metadata, dict):
        relationship = metadata.get("relationship", "")

    if not relationship:
        if sender_handle in contacts:
            contact_info = contacts[sender_handle]
            relationship = contact_info.get("relationship", "")
            if not sender_name or sender_name == sender_handle:
                sender_name = contact_info.get("name", sender_handle)
        elif sender_name in contacts:
            relationship = contacts[sender_name].get("relationship", "")

    # Format phone number for readability if sender has no explicit name
    if not sender_name or sender_name == sender_handle:
        if sender_handle.startswith("+") and len(sender_handle) >= 10:
            sender_name = f"Phone ({sender_handle})"
            if not relationship:
                relationship = "Unknown Contact"
        else:
            sender_name = sender_handle or "Unknown Contact"

    if not relationship:
        relationship = "Family Member"
    if not raw_message:
        raw_message = "(No text content)"

    # Resolve sender photo URL
    sender_photo_url = resolve_contact_photo(
        sender_name=sender_name,
        sender_handle=sender_handle,
        explicit_photo_url=explicit_photo,
    )

    return sender_name.strip(), relationship.strip(), raw_message.strip(), sender_handle.strip(), sender_photo_url


def generate_caregiver_reply(
    sender_name: str,
    anxiety_detected: bool,
    biometrics: Dict[str, Any],
    audio_generated: bool,
) -> str:
    """Constructs a comforting biometric confirmation sent back to family members."""
    patient_name = os.getenv("PATIENT_NAME", "Eleanor")
    hr = biometrics.get("heart_rate", 74.0)

    if anxiety_detected:
        intervention = "A soothing voice reminder was played at her bedside." if audio_generated else "A visual grounding card was displayed."
        return (
            f"Anchor Caregiver Update: We delivered and grounded your message for {patient_name}. "
            f"Current vitals show agitation (Heart Rate: {int(hr)} BPM). {intervention}"
        )
    else:
        return (
            f"Anchor Caregiver Update: Your message was grounded and delivered to {patient_name}'s bedside screen. "
            f"Her vitals are calm and stable (Heart Rate: {int(hr)} BPM)."
        )


def generate_malicious_caregiver_warning(
    sender_name: str,
    sender_handle: str,
    raw_message: str,
    malicious_reason: str,
) -> str:
    """
    Constructs a high-priority warning alert for caregivers containing:
    1. The message contents
    2. Who it is from (phone number if no contact info)
    3. Exactly why it was flagged as malicious
    """
    sender_identifier = sender_handle if sender_handle else sender_name
    if not sender_identifier or sender_identifier == "Unknown Contact":
        sender_identifier = "Unknown Phone Number"

    return (
        f"🚨 ANCHOR SHIELD ALERT: Malicious message intercepted and blocked from Eleanor.\n"
        f"• From: {sender_identifier}\n"
        f"• Message Content: \"{raw_message}\"\n"
        f"• Flagged Reason: {malicious_reason}\n"
        f"• Action Taken: Suppressed from Eleanor's bedside station. She was not alarmed."
    )


async def process_photon_message(
    payload: Dict[str, Any], force_audio: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Full pipeline:
    1. Parse incoming message, sender details, and multimodal contact photo.
    2. Scan for malicious / scam behavior via Gemini & security heuristics.
    3. If malicious:
       - BLOCK message from Eleanor's bedside tablet.
       - SUPPRESS voice audio generation.
       - Dispatch immediate warning alert to caregivers.
    4. If safe:
       - Ground message via Gemini for Eleanor.
       - Evaluate Presage anxiety state to gate ElevenLabs TTS voice synthesis.
       - Post safe grounded card with contact photo to Eleanor's tablet queue.
    """
    sender_name, relationship, raw_message, sender_handle, sender_photo_url = parse_photon_payload(payload)
    print(f"[Anchor Pipeline] Inbound: Sender='{sender_name}', Relation='{relationship}', Handle='{sender_handle}', Photo='{sender_photo_url}', Msg='{raw_message}'")

    # Evaluate Presage anxiety state
    anxiety_detected, anxiety_score, biometrics = evaluate_anxiety(payload)

    # Scan for malicious content and generate grounding
    print("[Anchor Pipeline] Calling Gemini cognitive security & grounding scan...")
    analysis = await analyze_and_ground_message(
        sender_name=sender_name,
        relationship=relationship,
        raw_message=raw_message,
        sender_handle=sender_handle,
    )

    is_malicious = bool(analysis.get("is_malicious", False))
    malicious_reason = str(analysis.get("malicious_reason", "")).strip()

    # MALICIOUS MESSAGE INTERVENTION BRANCH
    if is_malicious:
        print(f"[Anchor SHIELD]: Intercepted malicious message! Reason: {malicious_reason}")
        print("[Anchor SHIELD]: Suppressing patient tablet display & voice audio.")

        caregiver_warning = generate_malicious_caregiver_warning(
            sender_name=sender_name,
            sender_handle=sender_handle,
            raw_message=raw_message,
            malicious_reason=malicious_reason or "Suspicious predatory or extortion content detected.",
        )

        # Do NOT append to _recent_events so Eleanor's bedside screen remains unaffected
        result_event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "sender_name": sender_name,
            "relationship": relationship,
            "sender_handle": sender_handle,
            "sender_photo_url": sender_photo_url,
            "raw_message": raw_message,
            "grounded_message": "🛡️ Message shielded by Anchor Scam Defense.",
            "caregiver_reply": caregiver_warning,
            "anxiety_detected": anxiety_detected,
            "anxiety_score": anxiety_score,
            "biometrics": biometrics,
            "audio_generated": False,
            "audio_id": None,
            "audio_url": None,
            "is_malicious": True,
            "malicious_reason": malicious_reason or "Predatory or emergency financial scam detected.",
            "blocked_from_patient": True,
        }
        return result_event

    # SAFE MESSAGE BRANCH
    grounded_message = str(analysis.get("grounded_message", "")).strip()
    print(f"[Anchor Pipeline] Grounded text: '{grounded_message}'")

    should_generate_audio = anxiety_detected if force_audio is None else force_audio
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

    caregiver_reply = generate_caregiver_reply(
        sender_name=sender_name,
        anxiety_detected=anxiety_detected,
        biometrics=biometrics,
        audio_generated=audio_generated,
    )

    result_event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "sender_name": sender_name,
        "relationship": relationship,
        "sender_handle": sender_handle,
        "sender_photo_url": sender_photo_url,
        "raw_message": raw_message,
        "grounded_message": grounded_message,
        "caregiver_reply": caregiver_reply,
        "anxiety_detected": anxiety_detected,
        "anxiety_score": anxiety_score,
        "biometrics": biometrics,
        "audio_generated": audio_generated,
        "audio_id": audio_id,
        "audio_url": audio_url,
        "is_malicious": False,
        "malicious_reason": None,
        "blocked_from_patient": False,
    }

    # Only safe, grounded events are posted to Eleanor's tablet stream
    _recent_events.append(result_event)
    return result_event
