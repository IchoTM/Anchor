# Anchor: Real-Time Cognitive Security & Contextual Grounding for Dementia Care

> **An intelligent messaging interceptor and clinical display system designed to mitigate cognitive disorientation and prevent financial fraud in elderly patients.**

---

## System Overview & Problem Statement

1. **The Context Gap in Digital Communication:**  
   Patients experiencing dementia or cognitive decline frequently face anxiety when receiving standard text messages. A brief message such as *"I'll be there in 10 mins"* lacks necessary context, often causing distress as the patient may not recall the sender's identity, the scheduled event, or the current time of day.

2. **Targeted Financial Vulnerability:**  
   Elderly individuals are disproportionately targeted by financial fraud, losing billions annually to impersonation tactics, predatory wire fraud, and emergency scams. Standard blocking solutions are often insufficient and risk isolating the patient by accidentally filtering out legitimate communications from family members using new numbers.

---

## System Architecture

Anchor operates as a middleware layer, intercepting incoming messages (via iMessage, SMS, or webhooks) and routing them through a predefined clinical evaluation pipeline:

    [Inbound Message]
           │
           ▼
    [NLP Security & Context Analysis (Google Gemini)]
           │
           ├─────────────────────────────────┬────────────────────────────────┐
           │ (Flagged: Predatory / Scam)      │ (Safe: Verified Contact)       │
           ▼                                 ▼                                │
    [Out-of-Band Caregiver Alert]     [Biometric State Evaluation (Presage)]   │
    • Message dropped silently         • Evaluates heart rate & respiration   │
    • Device screen remains off        • Determines current anxiety baseline  │
                                               │                              │
                                               ├──────────────┬───────────────┤
                                               │ (Baseline)   │ (Elevated)    │
                                               ▼              ▼               ▼
                                       [Visual Grounding] [Audio Intervention]
                                       • Portrait Framing • Voice synthesis (ElevenLabs)
                                       • Sender ID Tag    • Auditory orientation cues

### Core Components

1. **Contextual NLP Engine (Google Gemini):**  
   Analyzes all inbound traffic for signs of extortion, financial demands, or impersonation.
   * **Threat Interception:** Malicious messages are silently dropped. The patient's display remains inactive to prevent startle responses, and an alert is automatically dispatched to the designated caregiver.
   * **Compromised Contact Protocol:** If a verified contact requests sensitive information (e.g., banking details, SSN), the system initiates a *Caregiver Review* hold and sends an automated verification prompt to the sender.
   * **Contextual Rewriting:** Safe messages are reformatted into gentle, grounding language that explicitly identifies the sender and their relationship to the patient.

2. **Biometric Integration (Presage SDK & ElevenLabs):**  
   * **Baseline State:** If the patient's vitals are stable, the system displays the grounded message silently.
   * **Elevated State:** If the Presage sensor detects an elevated heart rate or respiration spike indicative of anxiety, Anchor generates a soothing, synthesized voice narration via ElevenLabs to gently introduce the message.

3. **Patient-Facing Display Interface:**  
   * High-contrast, glare-resistant UI utilizing **Atkinson Hyperlegible** typography.
   * Persistent temporal and spatial orientation cues (e.g., *"Good Morning"*, *"You are safe at home"*).
   * Facial detection framing to ensure contact portraits are properly centered and clearly visible.

---

## Getting Started

### 1. Prerequisites
* Python 3.10+
* (Optional) `ngrok` for external mobile webhook testing

### 2. Installation
```bash
git clone [https://github.com/your-org/anchor.git](https://github.com/your-org/anchor.git)
cd anchor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt