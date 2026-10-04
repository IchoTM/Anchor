import os
import io
import re
import json
import uuid
import time
import base64
import hashlib
import asyncio
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv, find_dotenv

from services.gemini_service import analyze_and_ground_message
from services.elevenlabs_service import generate_speech_audio
from services.presage_service import evaluate_anxiety

load_dotenv(find_dotenv(), override=True)

# Directory paths for avatar resolution and processed face-cropped images
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
PROCESSED_AVATARS_DIR = STATIC_DIR / "avatars" / "processed"
PROCESSED_AVATARS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONTACTS: Dict[str, Dict[str, Any]] = {
    "+15551234567": {"name": "Alex", "relationship": "Grandson", "phone_number": "+15551234567", "photo_url": "/static/avatars/alex.jpg"},
    "+15550192": {"name": "Grandpa", "relationship": "Husband", "phone_number": "+15550192", "photo_url": "/static/avatars/grandpa.jpg"},
    "+15559876543": {"name": "Maria", "relationship": "Caregiver", "phone_number": "+15559876543", "is_caregiver": True, "photo_url": "/static/avatars/maria.jpg"},
    "Alex": {"name": "Alex", "relationship": "Grandson", "phone_number": "+15551234567", "photo_url": "/static/avatars/alex.jpg"},
    "Sarah": {"name": "Sarah", "relationship": "Daughter", "photo_url": "/static/avatars/sarah.jpg"},
    "David": {"name": "David", "relationship": "Son", "photo_url": "/static/avatars/david.jpg"},
    "Emma": {"name": "Emma", "relationship": "Granddaughter", "photo_url": "/static/avatars/sarah.jpg"},
    "Maria": {"name": "Maria", "relationship": "Caregiver", "phone_number": "+15559876543", "is_caregiver": True, "photo_url": "/static/avatars/maria.jpg"},
    "Grandpa": {"name": "Grandpa", "relationship": "Husband", "phone_number": "+15550192", "photo_url": "/static/avatars/grandpa.jpg"},
    "Dr. Chen": {"name": "Dr. Chen", "relationship": "Doctor", "photo_url": "/static/avatars/chen.jpg"},
}

# Heuristic name patterns mapped to photo files in /static/avatars/
AVATAR_NAME_PATTERNS: Dict[str, str] = {
    "alex": "/static/avatars/alex.jpg",
    "sarah": "/static/avatars/sarah.jpg",
    "david": "/static/avatars/david.jpg",
    "emma": "/static/avatars/sarah.jpg",
    "maria": "/static/avatars/maria.jpg",
    "grandpa": "/static/avatars/grandpa.jpg",
    "chen": "/static/avatars/chen.jpg",
}

# In-memory storage for audio bytes and safe message events for the bedside station
_audio_store: Dict[str, bytes] = {}
_recent_events: deque = deque(maxlen=50)
_events_lock = asyncio.Lock()


def get_stored_audio(audio_id: str) -> Optional[bytes]:
    """Retrieve raw audio bytes by audio ID."""
    return _audio_store.get(audio_id)


def get_recent_events(limit: int = 15, chronological: bool = False) -> List[Dict[str, Any]]:
    """Retrieve recent safe processed Anchor events for the bedside display."""
    events = list(_recent_events)
    if not chronological:
        events.reverse()
        return events[:limit]
    return events[-limit:]


def get_latest_event() -> Optional[Dict[str, Any]]:
    """Retrieve the single most recent safe event."""
    return _recent_events[-1] if _recent_events else None


def get_contact_directory() -> Dict[str, Dict[str, Any]]:
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


def _normalize_identifier(identifier: str) -> str:
    """Strips phone punctuation to allow matching formatted and unformatted phone numbers."""
    if not identifier:
        return ""
    digits = re.sub(r"\D", "", str(identifier))
    return digits if digits else str(identifier).strip().lower()


def _is_phone_string(s: str) -> bool:
    """Checks whether a string represents a phone number rather than a person's name."""
    if not s:
        return False
    s_clean = s.strip()
    if s_clean.startswith("+"):
        return True
    digits = re.sub(r"\D", "", s_clean)
    non_phone_chars = re.sub(r"[\d\s\-\(\)\+\.]", "", s_clean)
    if len(non_phone_chars) == 0 and len(digits) >= 7:
        return True
    if s_clean.lower().startswith("phone"):
        return True
    return False


def _phones_match(p1: str, p2: str) -> bool:
    """Matches phone numbers across different national/international formats."""
    d1 = re.sub(r"\D", "", str(p1 or ""))
    d2 = re.sub(r"\D", "", str(p2 or ""))
    if not d1 or not d2:
        return False
    if d1 == d2:
        return True
    # Match 10-digit national number suffixes (e.g. +1 555-123-4567 vs 5551234567)
    if len(d1) >= 10 and len(d2) >= 10 and d1[-10:] == d2[-10:]:
        return True
    return False


def extract_name_from_text(text: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Extracts sender name and optional relationship if self-introduced in the text.
    Examples:
      - "Hey Grandma, it's Tommy, just landed..." -> ("Tommy", "Grandson")
      - "Hi Mom, it's David from my new phone" -> ("David", "Son")
      - "Hi Eleanor, it's your granddaughter, Emma!" -> ("Emma", "Granddaughter")
      - "Hey Nana, it's Sarah" -> ("Sarah", "Granddaughter")
    """
    if not text:
        return None

    # Pattern 1: "it's your (granddaughter|grandson|daughter|son),? ([A-Z][a-z]+)"
    m1 = re.search(
        r"\b(?:it's|its|this is|i'm|im)\s+your\s+(granddaughter|grandson|daughter|son|caregiver)[,\s]+([A-Z][a-z]+)\b",
        text,
        re.IGNORECASE,
    )
    if m1:
        rel = m1.group(1).capitalize()
        name = m1.group(2).capitalize()
        return name, rel

    # Pattern 2: "it's ([A-Z][a-z]+),? your (granddaughter|grandson|daughter|son)"
    m2 = re.search(
        r"\b(?:it's|its|this is|i'm|im)\s+([A-Z][a-z]+)[,\s]+your\s+(granddaughter|grandson|daughter|son)\b",
        text,
        re.IGNORECASE,
    )
    if m2:
        name = m2.group(1).capitalize()
        rel = m2.group(2).capitalize()
        return name, rel

    # Pattern 3: Greeting + title + "it's [Name]"
    m3 = re.search(
        r"\b(?:hey|hi|hello)\s+(grandma|grandpa|mom|mum|dad|nana|eleanor)[,!.]*\s+(?:it's|its|this is|i'm|im)\s+([A-Z][a-z]+)\b",
        text,
        re.IGNORECASE,
    )
    if m3:
        title = m3.group(1).lower()
        name = m3.group(2).capitalize()
        inferred_rel = None
        if title in ("grandma", "nana"):
            inferred_rel = "Grandchild"
        elif title in ("mom", "mum"):
            inferred_rel = "Child"
        return name, inferred_rel

    # Pattern 4: Simple "it's [Name]" / "this is [Name]"
    m4 = re.search(r"\b(?:it's|its|this is)\s+([A-Z][a-z]+)\b", text, re.IGNORECASE)
    if m4:
        name = m4.group(1).capitalize()
        stopwords = {
            "me", "time", "sunny", "cold", "hot", "great", "okay", "ok", "fine",
            "ready", "here", "just", "so", "all", "not", "too", "important", "urgent",
        }
        if name.lower() not in stopwords:
            return name, None

    return None


def verify_contact(sender_handle: str, sender_name: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Verifies whether a sender's handle, phone number, or name exists in the contact directory.
    """
    contacts = get_contact_directory()
    name_norm = (sender_name or "").strip().lower()

    # 1. Match by handle or phone number
    if sender_handle:
        for key, info in contacts.items():
            if _phones_match(sender_handle, key):
                return True, info
            contact_phone = info.get("phone_number", "")
            if contact_phone and _phones_match(sender_handle, contact_phone):
                return True, info
            if key.strip().lower() == sender_handle.strip().lower():
                return True, info

    # 2. Match by contact name (ignoring phone strings)
    if name_norm and not _is_phone_string(sender_name):
        clean_name = re.sub(r"^your\s+", "", name_norm)
        for key, info in contacts.items():
            key_clean = key.strip().lower()
            info_name = str(info.get("name", "")).strip().lower()
            if clean_name == key_clean or clean_name == info_name:
                return True, info

    return False, None


def get_caregiver_recipients(exclude_handle: str = "") -> List[Dict[str, str]]:
    """
    Retrieves all verified, non-compromised caregiver contact numbers.
    Strictly excludes the sender (exclude_handle) so that compromised contacts or attackers
    are NEVER sent the security alert.
    """
    excluded_norm = _normalize_identifier(exclude_handle)
    recipients: List[Dict[str, str]] = []
    seen_identifiers = set()

    # 1. Inspect verified contact directory for known caregivers
    contacts = get_contact_directory()
    for key, info in contacts.items():
        rel = str(info.get("relationship", "")).lower()
        is_cg = bool(info.get("is_caregiver", False)) or ("caregiver" in rel) or ("nurse" in rel)

        if not is_cg:
            continue

        name = info.get("name") or "Primary Caregiver"
        phone = info.get("phone_number") or (key if key.startswith("+") or any(c.isdigit() for c in key) else "")

        if not phone:
            continue

        phone_norm = _normalize_identifier(phone)
        if excluded_norm and (phone_norm == excluded_norm or _phones_match(phone, exclude_handle)):
            # SENDER IS COMPROMISED: Exclude them from caregiver alert dispatch!
            continue

        if phone_norm not in seen_identifiers:
            seen_identifiers.add(phone_norm)
            recipients.append({
                "name": name,
                "phone_number": phone,
                "relationship": info.get("relationship", "Caregiver"),
            })

    # 2. Collect from environment variables
    env_numbers = []
    for var_name in ("CAREGIVER_PHONE_NUMBER", "PRIMARY_CAREGIVER_PHONE", "CAREGIVER_PHONE_NUMBERS"):
        val = os.getenv(var_name, "").strip()
        if val:
            for part in val.split(","):
                part_clean = part.strip()
                if part_clean:
                    env_numbers.append(part_clean)

    for phone in env_numbers:
        phone_norm = _normalize_identifier(phone)
        if excluded_norm and (phone_norm == excluded_norm or _phones_match(phone, exclude_handle)):
            continue

        if phone_norm not in seen_identifiers:
            seen_identifiers.add(phone_norm)
            recipients.append({
                "name": "Designated Caregiver",
                "phone_number": phone,
                "relationship": "Primary Caregiver",
            })

    return recipients


def _load_image_bytes(photo_url: str) -> Optional[bytes]:
    """Loads raw image bytes from local static path, remote URL, or data URI."""
    if not photo_url:
        return None

    if photo_url.startswith("/static/"):
        relative_path = photo_url.replace("/static/", "", 1)
        file_path = STATIC_DIR / relative_path
        if file_path.exists() and file_path.is_file():
            try:
                return file_path.read_bytes()
            except Exception as exc:
                print(f"[Anchor Warning] Could not read local image {file_path}: {exc}")
                return None

    if photo_url.startswith("data:image/") and ";base64," in photo_url:
        try:
            _, b64_data = photo_url.split(";base64,", 1)
            return base64.b64decode(b64_data)
        except Exception as exc:
            print(f"[Anchor Warning] Failed to decode base64 avatar: {exc}")
            return None

    if photo_url.startswith("http://") or photo_url.startswith("https://"):
        try:
            req = urllib.request.Request(photo_url, headers={"User-Agent": "Anchor-Dementia-Station/1.0"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                return resp.read()
        except Exception as exc:
            print(f"[Anchor Warning] Could not fetch remote avatar {photo_url}: {exc}")
            return None

    return None


def crop_face_for_dementia_recognition(photo_url: str) -> str:
    """
    Applies clinical dementia-friendly facial detection and adaptive framing.
    Caches the 512x512 cropped portrait into /static/avatars/processed/ and returns its URL.
    """
    if not photo_url or not photo_url.strip():
        return photo_url

    if "/avatars/processed/" in photo_url:
        return photo_url

    try:
        from PIL import Image, ImageOps
        import cv2
        import numpy as np
    except ImportError:
        return photo_url

    raw_bytes = _load_image_bytes(photo_url.strip())
    if not raw_bytes:
        return photo_url

    image_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
    target_filename = f"face_{image_hash}.jpg"
    target_path = PROCESSED_AVATARS_DIR / target_filename
    target_url = f"/static/avatars/processed/{target_filename}"

    if target_path.exists():
        return target_url

    try:
        pil_raw = Image.open(io.BytesIO(raw_bytes))
        pil_img = ImageOps.exif_transpose(pil_raw).convert("RGB")
        width, height = pil_img.size

        np_img = np.array(pil_img)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(int(min(width, height) * 0.15), int(min(width, height) * 0.15)),
        )

        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            fx, fy, fw, fh = faces[0]

            fcx = fx + (fw / 2.0)
            fcy = fy + (fh / 2.0)

            target_box_size = max(fw, fh) * 1.95
            target_box_size = min(target_box_size, min(width, height))

            crop_cx = fcx
            crop_cy = fcy - (target_box_size * 0.06)

            left = crop_cx - (target_box_size / 2.0)
            top = crop_cy - (target_box_size / 2.0)
            right = left + target_box_size
            bottom = top + target_box_size

            if left < 0:
                right += abs(left)
                left = 0
            if top < 0:
                bottom += abs(top)
                top = 0
            if right > width:
                left -= (right - width)
                right = width
            if bottom > height:
                top -= (bottom - height)
                bottom = height

            left = max(0, int(left))
            top = max(0, int(top))
            right = min(width, int(right))
            bottom = min(height, int(bottom))

            cropped = pil_img.crop((left, top, right, bottom))
        else:
            square_size = min(width, height)
            left = max(0, int((width - square_size) / 2))
            top = max(0, int((height - square_size) * 0.25))
            right = left + square_size
            bottom = top + square_size
            cropped = pil_img.crop((left, top, right, bottom))

        resample_filter = getattr(Image, "Resampling", Image).LANCZOS
        cropped = cropped.resize((512, 512), resample=resample_filter)
        cropped.save(target_path, format="JPEG", quality=92, optimize=True)

        return target_url
    except Exception as exc:
        print(f"[Anchor Face Crop Exception]: {exc}")
        return photo_url


def resolve_contact_photo(
    sender_name: str,
    sender_handle: str,
    explicit_photo_url: Optional[str] = None,
    is_verified: bool = False,
) -> Optional[str]:
    """Resolves avatar image URL for a sender using explicit payload values or known contacts."""
    resolved_url: Optional[str] = None

    if explicit_photo_url and explicit_photo_url.strip():
        resolved_url = explicit_photo_url.strip()
    else:
        contacts = get_contact_directory()
        # Try handle match
        if sender_handle in contacts and contacts[sender_handle].get("photo_url"):
            resolved_url = contacts[sender_handle]["photo_url"]
        elif sender_name in contacts and contacts[sender_name].get("photo_url"):
            resolved_url = contacts[sender_name]["photo_url"]
        else:
            # Check heuristic avatar name patterns
            sender_lower = (sender_name or "").lower().strip()
            for pattern, path in AVATAR_NAME_PATTERNS.items():
                if pattern in sender_lower:
                    resolved_url = path
                    break

    if resolved_url:
        return crop_face_for_dementia_recognition(resolved_url)

    return None


def parse_photon_payload(payload: Dict[str, Any]) -> Tuple[str, str, str, str, Optional[str], bool]:
    """
    Extracts (sender_name, relationship, raw_message, sender_handle, sender_photo_url, is_verified_contact)
    with robust contact name resolution.
    """
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

    # 2. Extract sender info from dict or string
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
        if _is_phone_string(sender_raw):
            sender_handle = sender_raw
        else:
            sender_name = sender_raw
            sender_handle = sender_raw

    # Explicit top-level fields always take priority over raw string fallbacks
    if data.get("sender_name"):
        sender_name = str(data["sender_name"]).strip()
    elif data.get("name"):
        sender_name = str(data["name"]).strip()
    elif data.get("contact_name"):
        sender_name = str(data["contact_name"]).strip()

    if data.get("relationship"):
        relationship = str(data["relationship"]).strip()
    elif data.get("sender_relationship"):
        relationship = str(data["sender_relationship"]).strip()

    if not sender_handle:
        if data.get("identifier"):
            sender_handle = str(data["identifier"]).strip()
        elif data.get("handle"):
            sender_handle = str(data["handle"]).strip()
        elif data.get("From"):
            sender_handle = str(data["From"]).strip()
        elif data.get("phone"):
            sender_handle = str(data["phone"]).strip()
        elif data.get("phone_number"):
            sender_handle = str(data["phone_number"]).strip()

    if not relationship and isinstance(metadata, dict):
        relationship = str(metadata.get("relationship", "")).strip()

    # 3. Contact verification against directory
    is_verified, contact_info = verify_contact(sender_handle, sender_name)

    if is_verified and contact_info:
        relationship = relationship or contact_info.get("relationship", "")
        # If sender_name is missing or is just a phone number, use the verified directory name
        if not sender_name or _is_phone_string(sender_name):
            sender_name = contact_info.get("name", sender_name)
    else:
        # Cross-reference known contacts by relationship or name if handle was unverified
        contacts = get_contact_directory()
        if sender_name and not _is_phone_string(sender_name):
            clean_check = re.sub(r"^your\s+", "", sender_name.lower().strip())
            for key, info in contacts.items():
                if info.get("name", "").lower() == clean_check:
                    is_verified = True
                    contact_info = info
                    relationship = relationship or info.get("relationship", "")
                    sender_name = info.get("name", sender_name)
                    break

        # If sender_name is still a generic relationship string ("Son", "Granddaughter") or phone number:
        generic_relationship_names = {
            "son", "daughter", "grandson", "granddaughter", "husband", "caregiver", "doctor"
        }
        rel_to_check = relationship or (sender_name if sender_name.lower().strip() in generic_relationship_names else "")
        if (not sender_name or _is_phone_string(sender_name) or sender_name.lower().strip() in generic_relationship_names) and rel_to_check:
            for key, info in contacts.items():
                if str(info.get("relationship", "")).lower() == rel_to_check.lower():
                    sender_name = info.get("name", sender_name)
                    if not explicit_photo:
                        explicit_photo = info.get("photo_url")
                    break

    # 4. Check if the sender introduced themselves in the text (e.g. "Hey Grandma, it's Tommy")
    extracted = extract_name_from_text(raw_message)
    if extracted:
        extracted_name, extracted_rel = extracted
        if (
            not sender_name
            or _is_phone_string(sender_name)
            or sender_name.lower().startswith("your ")
            or sender_name.lower() in ("family or friend", "loved one", "son", "daughter", "grandson", "granddaughter")
        ):
            sender_name = extracted_name
        if extracted_rel and (not relationship or relationship.lower() in ("family member", "loved one", "family or friend")):
            relationship = extracted_rel

    # 5. Clean up any cases where sender_name was passed as "Your Son" or "Your Granddaughter"
    if sender_name:
        clean_lower = sender_name.lower().strip()
        if clean_lower.startswith("your "):
            potential_rel = clean_lower.replace("your ", "").strip().capitalize()
            if not relationship or relationship.lower() in ("family member", "loved one"):
                relationship = potential_rel
            contacts = get_contact_directory()
            for key, info in contacts.items():
                if str(info.get("relationship", "")).lower() == potential_rel.lower():
                    sender_name = info.get("name", sender_name)
                    break

    # 6. Fallback only if no human name could be found
    if not sender_name or _is_phone_string(sender_name):
        if relationship and "unknown" not in relationship.lower() and "loved one" not in relationship.lower():
            sender_name = f"Your {relationship}"
        else:
            sender_name = "Family or Friend"
            relationship = "Loved One"

    if not relationship:
        relationship = "Family Member"
    if not raw_message:
        raw_message = "(No text content)"

    sender_photo_url = resolve_contact_photo(
        sender_name=sender_name,
        sender_handle=sender_handle,
        explicit_photo_url=explicit_photo,
        is_verified=is_verified,
    )

    return (
        sender_name.strip(),
        relationship.strip(),
        raw_message.strip(),
        sender_handle.strip(),
        sender_photo_url,
        is_verified,
    )


def generate_caregiver_reply(
    sender_name: str,
    grounded_message: str,
    anxiety_detected: bool,
    biometrics: Dict[str, Any],
    audio_generated: bool,
) -> str:
    """Constructs a comforting biometric confirmation sent back to family members, including the grounded text."""
    patient_name = os.getenv("PATIENT_NAME", "Eleanor")
    hr = biometrics.get("heart_rate", 74.0)

    mode_note = (
        f"Spoken aloud to calm agitation ({int(hr)} BPM)"
        if audio_generated
        else f"Displayed on screen (Vitals stable, {int(hr)} BPM)"
    )

    greeting_target = sender_name if sender_name and not sender_name.lower().startswith("your ") else "there"
    return (
        f"Hi {greeting_target}, {patient_name} received your message.\n\n"
        f"Anchor grounded it as:\n"
        f'"{grounded_message}"\n\n'
        f"{mode_note}"
    )


def generate_safety_hold_notice(sender_name: str, relationship: str) -> str:
    """Generates an empathetic 'Caregiver Review' receipt for recognized family contacts."""
    contact_label = sender_name if sender_name and not sender_name.startswith("+") else "family"
    return (
        f"Anchor Notice: For Eleanor's peace of mind, messages concerning sensitive actions, "
        f"payments, or urgent requests are held in Caregiver Review and will not appear on her bedside display. "
        f"If this is {contact_label}, please connect with Eleanor or her primary caregiver by phone."
    )


def generate_malicious_caregiver_warning(
    sender_name: str,
    sender_handle: str,
    raw_message: str,
    malicious_reason: str,
    is_verified_contact: bool,
) -> str:
    """Constructs a high-priority warning alert for caregivers."""
    sender_identifier = sender_handle if sender_handle else sender_name
    contact_status = "Known/Verified Contact" if is_verified_contact else "UNVERIFIED / UNKNOWN SENDER"
    action_note = (
        "Zero response sent to sender. Held from Eleanor's bedside station."
        if not is_verified_contact
        else "Held in Caregiver Review. Eleanor's display was not disturbed."
    )

    return (
        f"🚨 ANCHOR SECURITY ALERT: Malicious/Predatory message blocked!\n"
        f"• From: {sender_identifier} ({contact_status})\n"
        f"• Message Content: \"{raw_message}\"\n"
        f"• Reason Flagged: {malicious_reason}\n"
        f"• Patient Action: {action_note}"
    )


def _send_twilio_alert_sync(
    account_sid: str, auth_token: str, from_number: str, to_number: str, body: str
) -> bool:
    """Dispatches SMS alert to primary caregiver via Twilio REST API."""
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        data = urllib.parse.urlencode({"From": from_number, "To": to_number, "Body": body}).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        auth_header = base64.b64encode(f"{account_sid}:{auth_token}".encode("utf-8")).decode("ascii")
        req.add_header("Authorization", f"Basic {auth_header}")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return resp.status in (200, 201)
    except Exception as exc:
        print(f"[Anchor Alert] Twilio SMS dispatch to {to_number} failed: {exc}")
        return False


def _send_webhook_alert_sync(webhook_url: str, payload: Dict[str, Any]) -> bool:
    """Dispatches webhook alert to external caregiver notification endpoint."""
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "Anchor-Security/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return resp.status in (200, 201, 204)
    except Exception as exc:
        print(f"[Anchor Alert] Caregiver webhook alert failed: {exc}")
        return False


async def dispatch_caregiver_alert(alert_payload: Dict[str, Any], sender_handle: str = "") -> List[str]:
    """
    Actively dispatches an emergency notification to non-compromised, known caregivers.
    Strictly skips the sender_handle so compromised accounts/scammers never receive the alert.
    """
    warning_text = alert_payload.get("caregiver_security_warning", "")
    caregivers = get_caregiver_recipients(exclude_handle=sender_handle)

    notified_list: List[str] = []

    print(f"\n=======================================================")
    print(f"🚨 [ANCHOR CAREGIVER DISPATCH - URGENT ACTION REQUIRED]")
    print(f"Targeting {len(caregivers)} non-compromised caregiver(s)...")
    if sender_handle:
        print(f"Excluded sender from caregiver dispatch: {sender_handle}")
    print(warning_text)
    print(f"=======================================================\n")

    twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_from = os.getenv("TWILIO_PHONE_NUMBER") or os.getenv("TWILIO_FROM_NUMBER")

    for cg in caregivers:
        phone = cg.get("phone_number")
        name = cg.get("name", "Caregiver")

        if not phone:
            continue

        if twilio_sid and twilio_token and twilio_from:
            print(f"[Anchor Alert] Dispatching outbound SMS to caregiver {name} at {phone}...")
            sms_sent = await asyncio.to_thread(
                _send_twilio_alert_sync, twilio_sid, twilio_token, twilio_from, phone, warning_text
            )
            if sms_sent:
                notified_list.append(f"{name} ({phone})")
                print(f"[Anchor Alert] Delivered emergency SMS to caregiver {name} ({phone}).")
        else:
            notified_list.append(f"{name} ({phone})")
            print(f"[Anchor Alert] Queued alert for caregiver {name} ({phone}) [Local/Dashboard Mode]")

    caregiver_webhook = os.getenv("CAREGIVER_WEBHOOK_URL")
    if caregiver_webhook:
        print(f"[Anchor Alert] Dispatching webhook alert to caregiver channel...")
        await asyncio.to_thread(_send_webhook_alert_sync, caregiver_webhook, alert_payload)
        notified_list.append("Caregiver Webhook Channel")

    return notified_list


async def process_photon_message(
    payload: Dict[str, Any], force_audio: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Full pipeline:
    1. Parse incoming message, sender name/relationship, and contact verification.
    2. Scan for malicious / scam behavior via Gemini & security heuristics.
    3. If malicious: Suppress bedside station & alert non-compromised caregivers.
    4. If safe: Ground message for Eleanor and gate ElevenLabs TTS via Presage vitals.
    """
    sender_name, relationship, raw_message, sender_handle, sender_photo_url, is_verified = parse_photon_payload(payload)
    print(
        f"[Anchor Pipeline] Inbound: Sender='{sender_name}', Relation='{relationship}', "
        f"Handle='{sender_handle}', Verified={is_verified}, Msg='{raw_message}'"
    )

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

    # MALICIOUS / RISKY MESSAGE BRANCH
    if is_malicious:
        print(f"[Anchor SHIELD]: Intercepted risky/malicious message! Reason: {malicious_reason}")
        print("[Anchor SHIELD]: Suppressing patient tablet display & voice audio.")

        caregiver_warning = generate_malicious_caregiver_warning(
            sender_name=sender_name,
            sender_handle=sender_handle,
            raw_message=raw_message,
            malicious_reason=malicious_reason or "Suspicious predatory or extortion content detected.",
            is_verified_contact=is_verified,
        )

        sender_reply = generate_safety_hold_notice(sender_name, relationship) if is_verified else None

        result_event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "sender_name": sender_name,
            "relationship": relationship,
            "sender_handle": sender_handle,
            "sender_photo_url": sender_photo_url,
            "is_verified_contact": is_verified,
            "raw_message": raw_message,
            "grounded_message": "",
            "caregiver_reply": sender_reply,
            "caregiver_security_warning": caregiver_warning,
            "anxiety_detected": anxiety_detected,
            "anxiety_score": anxiety_score,
            "biometrics": biometrics,
            "audio_generated": False,
            "audio_id": None,
            "audio_url": None,
            "is_malicious": True,
            "malicious_reason": malicious_reason or "Predatory or emergency financial scam detected.",
            "blocked_from_patient": True,
            "caregivers_notified": [],
        }

        notified = await dispatch_caregiver_alert(result_event, sender_handle=sender_handle)
        result_event["caregivers_notified"] = notified

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
        grounded_message=grounded_message,
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
        "is_verified_contact": is_verified,
        "raw_message": raw_message,
        "grounded_message": grounded_message,
        "caregiver_reply": caregiver_reply,
        "caregiver_security_warning": None,
        "anxiety_detected": anxiety_detected,
        "anxiety_score": anxiety_score,
        "biometrics": biometrics,
        "audio_generated": audio_generated,
        "audio_id": audio_id,
        "audio_url": audio_url,
        "is_malicious": False,
        "malicious_reason": None,
        "blocked_from_patient": False,
        "caregivers_notified": [],
    }

    async with _events_lock:
        _recent_events.append(result_event)

    return result_event
