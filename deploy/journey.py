"""
journey.py -- replay one demo patient's ED stay event by event, rescoring with the trained model at each step.

For every step it gives: what happened, the facts known so far (what changed), the admit score and why, and how that moves
the whole ED's expected admits (the sum of every undecided patient's score at that moment). Uses only demo patients
(split == "demo") and the real model; nothing here is hand-typed.
"""
import pandas as pd

import admit_model as AM
import config as C

FIELDS = list(C.MODEL_FEATURES)
LABEL = AM.FEATURE_LABELS


def load(model):
    d = C.DATA_DIR
    visits = pd.read_csv(f"{d}/visits.csv", parse_dates=["arrival_time", "disposition_time", "left_ed_time"])
    visits = visits[visits["split"] == "demo"].set_index("visit_id")
    events = pd.read_csv(f"{d}/events.csv", parse_dates=["event_time"])
    events = events[events["visit_id"].isin(visits.index)].sort_values(["visit_id", "event_time"])
    snap = pd.read_csv(f"{d}/snapshots.csv", parse_dates=["snapshot_time"])
    snap = snap[snap["split"] == "demo"].sort_values(["visit_id", "snapshot_time"]).reset_index(drop=True)
    snap["p"] = model.predict_proba(snap)
    contrib = model.contributions(snap)
    baseline = float(model.lr.intercept_[0] + (model.base_logit_means * model.lr.coef_[0]).sum())
    decided = visits["disposition_time"].fillna(visits["left_ed_time"])
    return {"visits": visits, "events": events, "snap": snap, "contrib": contrib, "baseline": baseline, "decided": decided}


def patients(J):
    """Demo patients that have at least two scores (so there is a journey to watch)."""
    n = J["snap"].groupby("visit_id").size()
    return [v for v in J["visits"].index if n.get(v, 0) >= 2]


def sentence(ev, snap_row):
    t = ev["event_type"]
    name = str(ev["order_name"]).replace("_", " ") if pd.notna(ev["order_name"]) else ""
    if t == "arrival":
        return "Patient arrives at the ED."
    if t == "triage":
        return f"Triage nurse assesses the patient and assigns ESI level {int(snap_row['esi'])}." if snap_row is not None else "Triage."
    if t == "provider_seen":
        return "The doctor sees the patient. (No new data yet, so no new score.)"
    if t == "order":
        return f"The doctor orders: {name}."
    if t == "result":
        return f"Result returns: {name} is " + ("ABNORMAL." if ev["result_abnormal"] in (True, "True") else "normal.")
    if t == "reassessment":
        return "Nurse rechecks the vitals. (They feed the next score.)"
    if t == "admit_decision":
        return "The doctor decides to ADMIT. The score is now a fact, and a bed request opens."
    return t.replace("_", " ").capitalize()


def steps(J, vid):
    """Events up to and including the admit decision (or all, if never admitted), each with its latest score."""
    ev = J["events"][J["events"]["visit_id"] == vid]
    stop = ev[ev["event_type"] == "admit_decision"]["event_time"]
    if len(stop):
        ev = ev[ev["event_time"] <= stop.iloc[0]]
    else:
        ev = ev[~ev["event_type"].isin(["bed_requested", "bed_matched", "bed_ready", "report_given", "transported"])]
    snap = J["snap"][J["snap"]["visit_id"] == vid]
    out = []
    for _, e in ev.iterrows():
        upto = snap[snap["snapshot_time"] <= e["event_time"]]
        row = upto.iloc[-1] if len(upto) else None
        new_score = bool(len(snap[snap["trigger_event_id"] == e["event_id"]]))
        out.append({"event": e, "snap_idx": None if row is None else int(row.name), "snap": row, "new_score": new_score,
                    "text": sentence(e, row)})
    return out


def facts(row):
    return {LABEL[f]: row[f] for f in FIELDS}


def ed_total(J, t):
    """Expected admits among everyone in the ED who has not yet had a disposition decision at time t."""
    v, s = J["visits"], J["snap"]
    live = v[(v["arrival_time"] <= t) & (J["decided"] > t)].index
    p = 0.0
    n = 0
    for vid in live:
        sub = s[(s["visit_id"] == vid) & (s["snapshot_time"] <= t)]
        if len(sub):
            p += float(sub["p"].iloc[-1]); n += 1
    return p, n


def split(total):
    return {k: total * sh for k, sh in C.LEVEL_OF_CARE_SHARE.items()}