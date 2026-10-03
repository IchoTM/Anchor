import os
import time
from typing import Any, Dict, Optional, Tuple
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

DEFAULT_ANXIETY_THRESHOLD = 0.60

# In-memory store for the latest biometric emotional telemetry from Presage
_patient_state: Dict[str, Any] = {
    "anxiety_score": 0.20,
    "anxiety_detected": False,
    "heart_rate": 72.0,
    "respiration_rate": 15.0,
    "stress_index": 0.25,
    "last_updated": time.time(),
}


def get_anxiety_threshold() -> float:
    """Returns the configured anxiety detection threshold."""
    try:
        return float(os.getenv("PRESAGE_ANXIETY_THRESHOLD", DEFAULT_ANXIETY_THRESHOLD))
    except ValueError:
        return DEFAULT_ANXIETY_THRESHOLD


def get_current_biometrics() -> Dict[str, Any]:
    """Returns the most recent biometric reading for the patient."""
    return dict(_patient_state)


def update_patient_biometrics(
    anxiety_score: Optional[float] = None,
    anxiety_detected: Optional[bool] = None,
    heart_rate: Optional[float] = None,
    respiration_rate: Optional[float] = None,
    stress_index: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Updates the active patient biometric state from Presage sensor telemetry.
    """
    threshold = get_anxiety_threshold()

    if anxiety_score is not None:
        _patient_state["anxiety_score"] = max(0.0, min(1.0, float(anxiety_score)))
        _patient_state["anxiety_detected"] = _patient_state["anxiety_score"] >= threshold
    elif anxiety_detected is not None:
        _patient_state["anxiety_detected"] = bool(anxiety_detected)
        _patient_state["anxiety_score"] = 0.85 if anxiety_detected else 0.20

    if heart_rate is not None:
        _patient_state["heart_rate"] = float(heart_rate)
    if respiration_rate is not None:
        _patient_state["respiration_rate"] = float(respiration_rate)
    if stress_index is not None:
        _patient_state["stress_index"] = float(stress_index)
    if metadata:
        _patient_state["metadata"] = metadata

    _patient_state["last_updated"] = time.time()
    return dict(_patient_state)


def evaluate_anxiety(payload: Optional[Dict[str, Any]] = None) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluates whether the patient is currently experiencing elevated anxiety.
    Checks the incoming webhook payload first for direct Presage telemetry,
    falling back to the current background sensor state.
    
    Returns:
        (is_anxiety_elevated: bool, anxiety_score: float, biometrics_snapshot: dict)
    """
    threshold = get_anxiety_threshold()

    if payload:
        # Check explicit top-level or nested presage fields in payload
        presage_data = payload.get("presage") or payload.get("biometrics")
        if isinstance(presage_data, dict):
            score = None
            detected = None
            if "anxiety_score" in presage_data:
                score = float(presage_data["anxiety_score"])
                detected = score >= threshold or bool(presage_data.get("anxiety_detected", False))
            elif "anxiety_detected" in presage_data:
                detected = bool(presage_data["anxiety_detected"])
                score = 0.85 if detected else 0.20

            if score is not None:
                updated = update_patient_biometrics(
                    anxiety_score=score,
                    anxiety_detected=detected,
                    heart_rate=presage_data.get("heart_rate"),
                    respiration_rate=presage_data.get("respiration_rate"),
                    stress_index=presage_data.get("stress_index"),
                    metadata=presage_data.get("metadata"),
                )
                return bool(detected), score, updated

        if "anxiety_score" in payload or "anxiety_detected" in payload:
            raw_score = payload.get("anxiety_score")
            raw_detected = payload.get("anxiety_detected")
            score = float(raw_score) if raw_score is not None else (0.85 if raw_detected else 0.20)
            detected = bool(raw_detected) if raw_detected is not None else (score >= threshold)
            updated = update_patient_biometrics(
                anxiety_score=score,
                anxiety_detected=detected,
            )
            return detected, score, updated

    # Fallback to current background biometric state
    current = get_current_biometrics()
    score = current.get("anxiety_score", 0.20)
    is_high = current.get("anxiety_detected", score >= threshold)
    return is_high, score, current
