import os
import time
from typing import Any, Dict, Optional, Tuple
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

DEFAULT_ANXIETY_THRESHOLD = float(os.getenv("PRESAGE_ANXIETY_THRESHOLD", "0.60"))

# In-memory store for the latest biometric emotional telemetry from Presage
_patient_state: Dict[str, Any] = {
    "anxiety_score": 0.20,
    "anxiety_detected": False,
    "heart_rate": 72.0,
    "respiration_rate": 15.0,
    "stress_index": 0.25,
    "last_updated": time.time(),
}


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
    threshold = float(os.getenv("PRESAGE_ANXIETY_THRESHOLD", str(DEFAULT_ANXIETY_THRESHOLD)))

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
    threshold = float(os.getenv("PRESAGE_ANXIETY_THRESHOLD", str(DEFAULT_ANXIETY_THRESHOLD)))

    if payload:
        # Check explicit top-level or nested presage fields in payload
        presage_data = payload.get("presage") or payload.get("biometrics") or {}
        if isinstance(presage_data, dict):
            if "anxiety_score" in presage_data:
                score = float(presage_data["anxiety_score"])
                is_high = score >= threshold or bool(presage_data.get("anxiety_detected", False))
                return is_high, score, presage_data
            if "anxiety_detected" in presage_data:
                is_high = bool(presage_data["anxiety_detected"])
                score = 0.85 if is_high else 0.20
                return is_high, score, presage_data

        if "anxiety_detected" in payload:
            is_high = bool(payload["anxiety_detected"])
            score = float(payload.get("anxiety_score", 0.85 if is_high else 0.20))
            return is_high, score, {"anxiety_detected": is_high, "anxiety_score": score}

        if "anxiety_score" in payload:
            score = float(payload["anxiety_score"])
            is_high = score >= threshold
            return is_high, score, {"anxiety_score": score, "anxiety_detected": is_high}

    # Fallback to current background biometric state
    current = get_current_biometrics()
    score = current.get("anxiety_score", 0.20)
    is_high = current.get("anxiety_detected", score >= threshold)
    return is_high, score, current
