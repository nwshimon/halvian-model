"""
case_engine.py -- one typed case in, a fully explained answer out.

  score_case()      structured case -> admit probability, band, and the arithmetic behind it (from the trained model)
  match_beds()      level of care + isolation -> ranked beds with reason codes (hand-written rules in bed_rules.py)
  annotate_blockers()  for beds that are waiting on a discharge, read the latest inpatient note with the LLM extractor
  record_decision() the human confirms or overrides; written to bed_matches, tasks and audit_log

The AI proposes; a named role decides. Nothing here changes a bed on its own.
"""
import os

import numpy as np
import pandas as pd

import admit_model as AM
import bed_rules as BR
import config as C
import extract_blockers as EB
import llm_client as L
import schema as S

NOW = pd.Timestamp(C.DEMO_DATE) + pd.Timedelta(hours=int(C.DEMO_NOW_TIME[:2]), minutes=int(C.DEMO_NOW_TIME[3:]))
DEFAULT_OCCUPANCY = 0.87


def load_world():
    """The demo board, notes and trained model. Called once by the app."""
    beds = pd.read_csv(f"{C.DATA_DIR}/beds.csv", parse_dates=["status_updated_time", "expected_ready_time"])
    notes = pd.read_csv(f"{C.DATA_DIR}/inpatient_notes.csv", parse_dates=["note_time"])
    occ = pd.read_csv(f"{C.DATA_DIR}/occupancy_hourly.csv", parse_dates=["hour_start"])
    occupancy = float(occ.loc[occ["hour_start"] == NOW.floor("h"), "inpatient_occupancy"].iloc[0]) if (occ["hour_start"] == NOW.floor("h")).any() else DEFAULT_OCCUPANCY
    return {"beds": beds, "notes": notes, "occupancy": occupancy, "model": AM.AdmitModel.load()}


def abnormal_flags(vitals):
    out = {}
    for k, (lo, hi) in C.VITAL_ABNORMAL.items():
        v = vitals.get(k)
        out[k] = v is not None and ((lo is not None and v < lo) or (hi is not None and v > hi))
    return out


def complete_case(case):
    """Fill what the user did not give with population-typical values, and say exactly which ones were assumed."""
    c, assumed = dict(case), []
    c["vitals"] = dict(case.get("vitals") or {})
    for k in ("hr", "spo2", "sbp", "rr", "temp_c"):
        if c["vitals"].get(k) is None:
            c["vitals"][k] = C.VITAL_BASELINE[k][0]; assumed.append(f"{k} (typical value {C.VITAL_BASELINE[k][0]})")
    if c.get("esi") is None:
        c["esi"] = 3; assumed.append("ESI (3)")
    if c.get("minutes_since_arrival") is None:
        c["minutes_since_arrival"] = 60.0; assumed.append("minutes since arrival (60)")
    if c.get("minutes_since_last_reassessment") is None:
        c["minutes_since_last_reassessment"] = max(0.0, c["minutes_since_arrival"] - 8); assumed.append("minutes since last check (arrival minus 8)")
    return c, assumed


def build_features(case, occupancy):
    """One row with exactly config.MODEL_FEATURES. Payer and other prohibited fields never appear here."""
    v, orders = case["vitals"], case.get("orders", [])
    flags = abnormal_flags(v)
    has_result = [o for o in orders if o["name"] not in C.ORDERS_WITHOUT_RESULT]
    row = {"age_band": case["age_band"], "esi": int(case["esi"]), "complaint_category": case["complaint_category"],
           "max_hr_so_far": v["hr"], "min_spo2_so_far": v["spo2"], "min_sbp_so_far": v["sbp"], "max_rr_so_far": v["rr"],
           "max_temp_so_far": v["temp_c"], "abnormal_vital_count_so_far": int(sum(flags.values())),
           "orders_so_far": len(orders), "ct_orders_so_far": sum(o["name"] == "CT" for o in orders),
           "abnormal_results_so_far": sum(bool(o.get("resulted")) and bool(o.get("abnormal")) for o in orders),
           "pending_results_so_far": sum(not o.get("resulted") for o in has_result),
           "mins_since_arrival": float(case["minutes_since_arrival"]), "mins_since_last_reassessment": float(case["minutes_since_last_reassessment"]),
           "inpatient_occupancy_at_snapshot": float(occupancy)}
    assert set(row) == set(C.MODEL_FEATURES) and not set(row) & set(C.PROHIBITED_FEATURES)
    return pd.DataFrame([row])[C.MODEL_FEATURES]


def score_case(case, world):
    """Admit probability plus the arithmetic: baseline log-odds + each feature's push = total log-odds -> probability."""
    am, X = world["model"], build_features(case, world["occupancy"])
    p = float(am.predict_proba(X)[0])
    contrib = am.contributions(X).iloc[0]
    baseline = float(am.lr.intercept_[0] + (am.base_logit_means * am.lr.coef_[0]).sum())
    rows = [{"feature": AM.FEATURE_LABELS[f], "value": X.iloc[0][f], "push": float(contrib[f])} for f in contrib.index]
    rows.sort(key=lambda r: -abs(r["push"]))
    total = baseline + sum(r["push"] for r in rows)
    band = str(AM.AdmitModel.category([p])[0])
    return {"p": p, "band": band, "baseline_logodds": baseline, "baseline_p": float(AM.sigmoid(baseline)), "total_logodds": total,
            "rows": rows, "top3": rows[:C.TOP_FACTORS_SHOWN], "features": X.iloc[0].to_dict(),
            "calibrated": am.calibration != "none"}


def patient_for_matching(case):
    pref = C.PREFERRED_UNIT_BY_COMPLAINT.get(case["complaint_category"])
    return {"requested_level_of_care": case["requested_level_of_care"], "isolation_required": bool(case["isolation_required"]), "preferred_unit": pref}


def match_beds(case, beds, now=NOW, top_n=C.MATCH_TOP_N):
    """Ranked beds with reason codes, plus how many beds passed the hard filters."""
    pat = patient_for_matching(case)
    if pat["requested_level_of_care"] not in C.LEVEL_OF_CARE_SHARE:
        return {"patient": pat, "ranked": [], "n_feasible": 0, "n_beds": len(beds), "needs_level_of_care": True}
    b = beds.set_index("bed_id")
    ranked = []
    for i, r in enumerate(BR.rank_beds(pat, beds, now, top_n), start=1):
        bed = b.loc[r["bed_id"]]
        mins = 0.0 if bed["status"] == "ready" else max(0.0, (bed["expected_ready_time"] - now).total_seconds() / 60)
        ranked.append({"rank": i, "bed_id": r["bed_id"], "score": r["score"], "reasons": r["reason_codes"], "unit": bed["unit"],
                       "bed_type": bed["bed_type"], "status": bed["status"], "isolation_capable": bool(bed["isolation_capable"]),
                       "discharge_pending": bool(bed["discharge_pending"]), "minutes_until_ready": round(mins)})
    return {"patient": pat, "ranked": ranked, "n_feasible": len(BR.feasible_beds(pat, beds)), "n_beds": len(beds), "needs_level_of_care": False}


def annotate_blockers(ranked, beds, notes, use_llm=True):
    """For each recommended bed that is occupied-and-waiting-on-discharge, extract blockers from its latest note."""
    out = {}
    b = beds.set_index("bed_id")
    for r in ranked:
        bed = b.loc[r["bed_id"]]
        if not (bed["status"] == "occupied" and bool(bed["discharge_pending"])):
            continue
        n = notes[notes["bed_id"] == r["bed_id"]].sort_values("note_time")
        if n.empty:
            continue
        latest = n.iloc[-1]
        try:
            res = EB.extract(latest["note_text"]) if use_llm else EB.rules_extract(latest["note_text"])
        except L.LLMUnavailable as e:
            res = EB.rules_extract(latest["note_text"]); res["unavailable_reason"] = str(e)
        out[r["bed_id"]] = {"note_id": latest["note_id"], "note_time": latest["note_time"], "note_text": latest["note_text"],
                            "visit_id": bed["current_visit_id"], **res}
    return out


def _append(path, df, table):
    for c in S.cols(table):
        if c not in df:
            df[c] = pd.NA
    df = df[S.cols(table)]
    os.makedirs(C.DATA_DIR, exist_ok=True)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def record_decision(case_id, ranked, chosen_bed, role, override_reason, blockers, now=NOW, data_dir=C.DATA_DIR):
    """Writes the human decision. An override MUST carry a reason. Returns what was written."""
    chosen = next((r for r in ranked if r["bed_id"] == chosen_bed), None)
    overridden = chosen is None or chosen is not ranked[0]
    if overridden and not (override_reason or "").strip():
        raise ValueError("an override needs a written reason")
    ts, mid = pd.Timestamp(now), f"M-{pd.Timestamp.now().strftime('%H%M%S%f')}"
    rows = []
    for r in ranked:
        status = "accepted" if (chosen is not None and r["bed_id"] == chosen_bed and not overridden) else ("rejected" if overridden else "proposed")
        rows.append({"match_id": f"{mid}-{r['rank']}", "visit_id": case_id, "bed_id": r["bed_id"], "created_time": ts, "rank": r["rank"],
                     "match_score": r["score"], "reason_codes": str(r["reasons"]), "status": status,
                     "decided_by_role": role if status in ("accepted", "rejected") else "ai", "override_reason": override_reason if status == "rejected" else None})
    if overridden:
        rows.append({"match_id": f"{mid}-0", "visit_id": case_id, "bed_id": chosen_bed, "created_time": ts, "rank": pd.NA,
                     "match_score": chosen["score"] if chosen else pd.NA, "reason_codes": str(chosen["reasons"]) if chosen else "[]",
                     "status": "overridden", "decided_by_role": role, "override_reason": override_reason})
    matches = pd.DataFrame(rows)
    matches["rank"] = matches["rank"].astype("Int64")
    tasks = pd.DataFrame([{"task_id": f"T-{mid}-{i}", "visit_id": blockers[chosen_bed]["visit_id"], "created_time": ts, "task_type": f["blocker_type"],
                           "owner_role": f["owner_role"], "due_time": ts + pd.Timedelta(hours=2), "status": "open", "source": "note_extraction"}
                          for i, f in enumerate(blockers.get(chosen_bed, {}).get("findings", []))])
    audit = pd.DataFrame([
        {"log_id": f"L-{mid}-1", "log_time": ts, "actor": "ai", "action": "proposed_beds", "entity_type": "case", "entity_id": case_id,
         "detail": str([{"bed": r["bed_id"], "score": r["score"]} for r in ranked])},
        {"log_id": f"L-{mid}-2", "log_time": ts, "actor": role, "action": "overrode_ai" if overridden else "confirmed_ai", "entity_type": "case",
         "entity_id": case_id, "detail": str({"bed": chosen_bed, "reason": override_reason or None})}])
    _append(f"{data_dir}/bed_matches.csv", matches, "bed_matches")
    if len(tasks):
        _append(f"{data_dir}/tasks.csv", tasks, "tasks")
    _append(f"{data_dir}/audit_log.csv", audit, "audit_log")
    return {"matches": len(matches), "tasks": len(tasks), "audit": len(audit), "overridden": overridden}
