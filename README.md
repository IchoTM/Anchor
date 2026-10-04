# ⚓ Anchor: Real-Time Cognitive Security & Grounding for Dementia Care

> **Bridging the context gap for elderly individuals experiencing dementia while protecting them from predatory financial fraud.**

---

## 💡 The Problem

1. **The Context Gap:**  
   Elderly individuals with dementia frequently experience anxiety and disorientation when receiving text messages. A brief message like *"I'll be there in 10 mins"* can trigger panic because they may not remember who sent it, why they are coming, or what time of day it is.

2. **The Fraud & Vulnerability Epidemic:**  
   Seniors lose over **$3 billion annually** to emergency grandparent scams, predatory wire fraud, and high-pressure impersonation tactics. Traditional blocking solutions also accidentally cut off genuine family members texting from new numbers.

---

## 🛡️ How Anchor Works

Anchor intercepts incoming messages (iMessage, SMS, or Spectrum/Photon webhooks) and routes them through a clinical care pipeline:

    [Inbound Message]
           │
           ▼
    [Gemini Security Scan & Contextual Grounding]
           │
           ├─────────────────────────────────┬────────────────────────────────┐
           │ (Flagged: Predatory / Scam)      │ (Safe: Genuine Family / Friend)│
           ▼                                 ▼                                │
    [Silent Bedside Shield]           [Presage Biometric Sensor Scan]          │
    • Screen stays quiet               • Heart Rate & Respiration             │
    • Zero acoustic startle           • Anxiety Threshold Evaluated            │
    • Out-of-band Caregiver Alert              │                              │
                                               ├──────────────┬───────────────┤
                                               │ (Calm State) │(Spike / Panic)│
                                               ▼              ▼               ▼
                                       [Visual Grounding] [ElevenLabs Voice Synthesis]
                                       • Portrait Framing • Soothing spoken bedside narration
                                       • Relationship Tag • Gentle earcon chime

1. **Cognitive Security Scan (Google Gemini):**  
   Analyzes every inbound text for extortion, urgent money demands, gift card requests, or impersonation.
   - **Predatory Scam:** Dropped silently (*Silent Blackhole*). Bedside screen does not light up. Immediate SMS alert dispatched to designated caregivers.
   - **Compromised Family Request:** If a known contact asks for sensitive banking info, a polite *Caregiver Review* hold receipt is returned to the sender.
   - **Safe Message:** Rewritten in gentle, grounding language explicitly identifying the sender and their relationship to Eleanor.

2. **Emotion-Gated Voice Narration (Presage SDK + ElevenLabs):**  
   - If Eleanor's vitals are calm, Anchor presents the message quietly on her bedside display.
   - If Presage detects elevated anxiety or distress (heart rate/respiration spike), Anchor generates a soothing, warm spoken narration using ElevenLabs.

3. **Clinical Bedside Orientation Station:**  
   - High-contrast, glare-resistant display using **Atkinson Hyperlegible** typography.
   - Time-of-day orientation anchor (e.g., *"Peaceful Morning"*, *"Safe at home in Room 3B"*).
   - Facial detection framing ensuring family portraits are centered and legible for aging eyes.

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.10+
- (Optional) `ngrok` for public mobile testing

### 2. Installation
    git clone https://github.com/your-org/anchor.git
    cd anchor
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

### 3. Environment Configuration
Copy `.env.example` to `.env` and fill in your API credentials:
    cp .env.example .env

Key variables:
- `GEMINI_API_KEY`: Google Gemini API key
- `ELEVENLABS_API_KEY`: ElevenLabs API key
- `CAREGIVER_PHONE_NUMBER`: Phone number for out-of-band security alerts

### 4. Running the Server
    python main.py

Anchor will start at `http://localhost:8000`.

---

## 📱 Interactive Interfaces

| Route | Interface | Description |
|---|---|---|
| `/` or `/tablet` | **Eleanor's Bedside Station** | Clinical ambient clock, orientation cues, and incoming grounded message stage. |
| `/text` or `/mobile` | **Family iMessage Simulator** | Native iOS-styled interface for texting Eleanor and uploading contact photos. |
| `/dev` or `/controller` | **Presenter Remote** | Biometric slider controls (Presage) and 1-click guided pitch sequence for judges. |
| `/docs` | **FastAPI Swagger API** | Interactive documentation for all REST and webhook endpoints. |

---

## 🧪 Guided 1-Click Pitch Scenarios

From the Presenter Remote (`/dev`), you can trigger four core acts:

1. **Act 1: The Context Gap (Calm • Text Only)**  
   Alex texts *"Be there in 10!"* -> Eleanor's vitals are calm (72 BPM). Anchor grounds the relationship quietly on screen.
2. **Act 2: Presage Biometric Voice Intervention (Panic • Voice On)**  
   Sarah texts *"Did you take your pills?"* -> Vitals spike to 88%. Anchor plays an orienting earcon and narrates the message soothingly via ElevenLabs.
3. **Act 3: The Silent Scam Shield (Predatory Blackhole)**  
   Unknown number demands an urgent wire transfer -> Bedside tablet remains completely undisturbed; emergency alert is routed to caregivers.
4. **Act 4: Compromised Contact Hold (Dual-Tier Protocol)**  
   A verified family phone asks for Eleanor's SSN or routing number -> Held in Caregiver Review; sender receives polite verification instructions.

---

## 📄 License
MIT License. Built for compassionate elder care and cognitive security.
