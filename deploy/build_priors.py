"""
build_priors.py -- small lookup tables built ONCE from the TRAINING split (admitted visits only) so the early-prep loop can guess,
before the doctor decides, (a) which level of care a likely admit will need, (b) how long until the decision, (c) whether isolation is needed.

  level_priors          P(level of care | complaint, ESI)        (falls back to the overall share when n < PRIOR_MIN_N)
  decision_time_priors  minutes from a snapshot to the admit decision, by results still pending (0, 1, 2, 3+): p25 / p50 / p75
  isolation_priors      P(isolation required | complaint, ESI)

Never uses payer or any other prohibited field. Output: data/priors.json.   Run:  python build_priors.py
"""
import json

import numpy as np
import pandas as pd

import config as C

PRIOR_MIN_N = 30
LEVELS = ["med_surg", "telemetry", "icu"]


def build():
    v = pd.read_csv(f"{C.DATA_DIR}/visits.csv", parse_dates=["disposition_time"])
    tr = v[(v["split"] == "train") & (v["disposition"] == "admit")].copy()
    overall = tr["requested_level_of_care"].value_counts(normalize=True)
    overall_iso = float(tr["isolation_required"].astype(float).mean())
    level_rows, iso_rows = [], []
    for (c, e), g in tr.groupby(["complaint_category", "esi"]):
        n = len(g)
        sh = g["requested_level_of_care"].value_counts(normalize=True) if n >= PRIOR_MIN_N else overall
        level_rows.append({"complaint_category": c, "esi": int(e), "n": int(n), **{lv: round(float(sh.get(lv, 0.0)), 4) for lv in LEVELS},
                           "fallback": bool(n < PRIOR_MIN_N)})
        iso_rows.append({"complaint_category": c, "esi": int(e), "n": int(n),
                         "share": round(float(g["isolation_required"].astype(float).mean()) if n >= PRIOR_MIN_N else overall_iso, 4)})
    snap = pd.read_csv(f"{C.DATA_DIR}/snapshots.csv", parse_dates=["snapshot_time"])
    snap = snap[(snap["split"] == "train") & (snap["label_admit"] == 1)].merge(tr[["visit_id", "disposition_time"]], on="visit_id")
    snap["mins_to_decision"] = (snap["disposition_time"] - snap["snapshot_time"]).dt.total_seconds() / 60
    snap = snap[snap["mins_to_decision"] > 0]
    snap["bucket"] = snap["pending_results_so_far"].clip(upper=3)
    dt_rows = []
    for b, g in snap.groupby("bucket"):
        q = g["mins_to_decision"].quantile([.25, .5, .75])
        dt_rows.append({"pending_results": int(b), "n": int(len(g)), "p25": round(float(q.iloc[0]), 1), "p50": round(float(q.iloc[1]), 1),
                        "p75": round(float(q.iloc[2]), 1)})
    out = {"built_from": "train split, admitted visits", "level_priors": level_rows, "decision_time_priors": dt_rows,
           "isolation_priors": iso_rows, "overall_level_share": {k: round(float(x), 4) for k, x in overall.items()}}
    with open(f"{C.DATA_DIR}/priors.json", "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    o = build()
    print("decision-time priors (minutes from snapshot to decision):")
    print(pd.DataFrame(o["decision_time_priors"]).to_string(index=False))
    lp = pd.DataFrame(o["level_priors"])
    print("\nlevel priors, selected rows:")
    print(lp[(lp.complaint_category == "abdominal_pain") & (lp.esi == 3)].to_string(index=False))
    print("overall level share:", o["overall_level_share"])
    print(f"wrote {C.DATA_DIR}/priors.json ({len(lp)} complaint x ESI cells)")
