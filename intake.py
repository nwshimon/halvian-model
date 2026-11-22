"""
intake.py -- turns a typed ED case (free text) into the structured fields the admit model needs.

An LLM reads the text and fills a fixed form. The result is then VALIDATED (allowed values, sane ranges), anything not
stated stays empty (never guessed), and every filled field carries the phrase it came from so a person can check it.
The app shows the filled form for the user to correct before anything is scored.

Not allowed in the form, by design: payer, race, language, name, record number. The LLM is told to ignore them.
"""
import re

import config as C
import llm_client as L

COMPLAINTS = list(C.COMPLAINT_MIX)
ORDERS = list(C.ORDER_BASE_PROB)
LEVELS = list(C.LEVEL_OF_CARE_SHARE)
VITALS = ["hr", "spo2", "sbp", "rr", "temp_c"]
RANGES = {"hr": (20, 250), "spo2": (50, 100), "sbp": (40, 260), "rr": (4, 60), "temp_c": (30, 43)}

SYSTEM = (
    "You fill in a structured form from a short emergency department case written by a clinician. Use ONLY what the text states. "
    "If something is not stated, use null. Never guess and never infer a diagnosis. Ignore names, record numbers, insurance, race and language.\n\n"
    f"age_band: one of {C.AGE_BANDS} (choose the band containing the stated age).\n"
    f"complaint_category: one of {COMPLAINTS} (the main presenting problem).\n"
    "esi: integer 1-5 if a triage/ESI level is stated.\n"
    "vitals: hr (beats/min), spo2 (%), sbp (systolic mmHg), rr (breaths/min), temp_c (Celsius; convert Fahrenheit). "
    "If several readings are given, use the worst: highest hr, rr and temperature, lowest spo2 and sbp.\n"
    f"orders: list of tests ordered so far; name must be one of {ORDERS}; resulted true/false; abnormal true/false/null (null if not resulted).\n"
    "minutes_since_arrival and minutes_since_last_reassessment: numbers if stated, else null.\n"
    "admit_decided: true only if the text says the patient will be admitted. "
    f"requested_level_of_care: one of {LEVELS} or null. isolation_required: true only if isolation is stated as needed.\n"
    "evidence: an object mapping each field you filled (use the field name, or vitals.hr etc.) to the exact words from the text it came from.\n\n"
    'Reply with JSON only: {"age_band":..,"complaint_category":..,"esi":..,"vitals":{"hr":..,"spo2":..,"sbp":..,"rr":..,"temp_c":..},'
    '"orders":[{"name":..,"resulted":..,"abnormal":..}],"minutes_since_arrival":..,"minutes_since_last_reassessment":..,'
    '"admit_decided":..,"requested_level_of_care":..,"isolation_required":..,"evidence":{..}}'
)


def _num(x):
    try:
        return None if x is None or isinstance(x, bool) else float(x)
    except (TypeError, ValueError):
        return None


def validate(raw, text):
    """Returns (case dict with None for unknowns, list of warnings)."""
    warn, raw = [], (raw if isinstance(raw, dict) else {})
    case = {"age_band": raw.get("age_band") if raw.get("age_band") in C.AGE_BANDS else None,
            "complaint_category": raw.get("complaint_category") if raw.get("complaint_category") in COMPLAINTS else None}
    esi = _num(raw.get("esi"))
    case["esi"] = int(esi) if esi is not None and 1 <= esi <= 5 and esi == int(esi) else None
    vit = raw.get("vitals") if isinstance(raw.get("vitals"), dict) else {}
    case["vitals"] = {}
    for k in VITALS:
        v = _num(vit.get(k))
        if v is not None and not (RANGES[k][0] <= v <= RANGES[k][1]):
            warn.append(f"{k}={v} is outside the believable range {RANGES[k]}; ignored")
            v = None
        case["vitals"][k] = v
    case["orders"] = []
    for o in raw.get("orders") or []:
        if isinstance(o, dict) and o.get("name") in ORDERS:
            res = bool(o.get("resulted")) and o["name"] not in C.ORDERS_WITHOUT_RESULT
            case["orders"].append({"name": o["name"], "resulted": res, "abnormal": bool(o.get("abnormal")) if res else False})
    for k in ("minutes_since_arrival", "minutes_since_last_reassessment"):
        v = _num(raw.get(k))
        case[k] = v if v is not None and 0 <= v <= 72 * 60 else None
    case["admit_decided"] = bool(raw.get("admit_decided"))
    lvl = raw.get("requested_level_of_care")
    case["requested_level_of_care"] = lvl if lvl in LEVELS else None
    case["isolation_required"] = bool(raw.get("isolation_required"))
    ev = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}
    case["evidence"] = {}
    for field, phrase in ev.items():
        phrase = str(phrase)
        if phrase and re.search(re.escape(re.sub(r"\s+", " ", phrase.strip())).replace(r"\ ", r"\s+"), text, flags=re.I):
            case["evidence"][str(field)] = phrase
        else:
            warn.append(f"evidence for {field!r} is not in the text; hidden")
    if case["admit_decided"] and not case["requested_level_of_care"]:
        warn.append("admit decided but no level of care stated: please choose one")
    return case, warn


def parse_case(text):
    """LLM intake. Raises llm_client.LLMUnavailable when there is no key/network/cache entry."""
    raw, meta = L.chat_json(SYSTEM, f"CASE:\n{text}")
    case, warn = validate(raw, text)
    return {"case": case, "warnings": warn, "source": meta["source"], "model": f"{meta['provider']}/{meta['model']}"}
