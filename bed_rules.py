"""
bed_rules.py -- the hand-written bed-matching rules (hard filters + weighted score), all numbers from config.py.

Used now to prove the demo data can produce the intended recommendations, and reused by the Step 8 matcher.
No machine learning: every recommendation comes with the reason codes that produced it. [SPEC]
"""
import pandas as pd

import config as C

LEVEL_RANK = {"med_surg": 0, "telemetry": 1, "icu": 2}      # a bed can serve its own level or any lower one


def _minutes_until_ready(bed, now):
    if bed["status"] == "ready":
        return 0.0
    t = bed["expected_ready_time"]
    return float("inf") if pd.isna(t) else max(0.0, (t - now).total_seconds() / 60)


def passes_hard_filters(patient, bed):
    """patient: dict with requested_level_of_care, isolation_required.  bed: row of the beds table."""
    can_serve = LEVEL_RANK[bed["bed_type"]] >= LEVEL_RANK[patient["requested_level_of_care"]]
    isolation_ok = bool(bed["isolation_capable"]) or not bool(patient["isolation_required"])
    freeing = bed["status"] in ("ready", "cleaning") or (bed["status"] == "occupied" and bool(bed["discharge_pending"]))
    return can_serve and isolation_ok and freeing


def score_bed(patient, bed, now):
    """Returns (score, reason_codes). Score = sum(weight x term) with the terms below."""
    w = C.MATCH_WEIGHTS
    terms = {
        "ready_now": 1.0 if bed["status"] == "ready" else 0.0,
        "minutes_until_ready": _minutes_until_ready(bed, now),
        "exact_level_of_care": 1.0 if bed["bed_type"] == patient["requested_level_of_care"] else 0.0,
        "preferred_unit": 1.0 if (patient.get("preferred_unit") and bed["unit"] == patient["preferred_unit"]) else 0.0,
        "isolation_bed_wasted": 1.0 if (bool(bed["isolation_capable"]) and not bool(patient["isolation_required"])) else 0.0,
    }
    score = sum(w[k] * v for k, v in terms.items())
    return score, [k for k, v in terms.items() if v]


def rank_beds(patient, beds, now, top_n=C.MATCH_TOP_N):
    """Top-N feasible beds, best first. Ties break on sooner-ready, then bed_id, so results are repeatable."""
    rows = []
    for _, bed in beds.iterrows():
        if passes_hard_filters(patient, bed):
            s, codes = score_bed(patient, bed, now)
            rows.append({"bed_id": bed["bed_id"], "score": round(s, 3), "reason_codes": codes,
                         "_ready": _minutes_until_ready(bed, now)})
    rows.sort(key=lambda r: (-r["score"], r["_ready"], r["bed_id"]))
    return [{k: v for k, v in r.items() if k != "_ready"} for r in rows[:top_n]]


def feasible_beds(patient, beds):
    return [b["bed_id"] for _, b in beds.iterrows() if passes_hard_filters(patient, b)]
