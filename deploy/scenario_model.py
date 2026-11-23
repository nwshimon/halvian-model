"""
scenario_model.py -- how much boarding time the early-prep loop could save across all admits. A SCENARIO, never a measurement.

Method (test split only, admitted visits):
  * lead time = minutes from the first time the admit score reaches the expedite threshold (0.60) to the doctor's decision; patients never flagged get 0.
  * head start = usable share (0.8) of the lead. It is spent first on the decision-to-bed-request stage (the request goes out at 60% instead of after the decision),
    then on the bed-ready wait (the discharge chain starts earlier). Savings are capped by the stage each one acts on.
  * r = share of prompts staff act on, s = share of those where a suitable bed exists. Saving is applied at r x s.
  * f = trim of the bed-ready wait for ALL admits from forecast-driven discharge work (pooled holds). The literature supports reductions of this size
    (UCLA 49% in discharge order to departure; Hopkins command center 30% fewer ED patients waiting for a bed).
  Scenarios: conservative (r 0.5, s 0.5, f 15%), base (0.75, 0.6, 25%), optimistic (1.0, 0.8, 40%).
Nobody is assumed to work faster: stage durations are the simulated ones (stage split from the literature, total from the brief).
Writes data/scenario_results.json.   Run:  python scenario_model.py
"""
import json

import numpy as np
import pandas as pd

import admit_model as AM
import config as C

USABLE = 0.8
SCENARIOS = {"conservative": (0.5, 0.5, 0.15), "base": (0.75, 0.6, 0.25), "optimistic": (1.0, 0.8, 0.40)}


def run():
    am = AM.AdmitModel.load()
    v = pd.read_csv(f"{C.DATA_DIR}/visits.csv", parse_dates=["arrival_time", "disposition_time", "bed_requested_time", "bed_matched_time", "bed_ready_time", "left_ed_time"])
    sn = pd.read_csv(f"{C.DATA_DIR}/snapshots.csv", parse_dates=["snapshot_time"])
    te = v[(v["split"] == "test") & (v["disposition"] == "admit")].set_index("visit_id")
    s = sn[(sn["split"] == "test") & sn["visit_id"].isin(te.index)].copy()
    s["p"] = am.predict_proba(s)
    s["dec"] = te["disposition_time"].reindex(s["visit_id"]).to_numpy()
    s = s[s["snapshot_time"] < s["dec"]]
    lead_stats = {}
    for th in (C.PREP_SOFT_THRESHOLD, C.PREP_EXPEDITE_THRESHOLD, 0.80):
        f = s[s["p"] >= th].groupby("visit_id")["snapshot_time"].min()
        lead = (te["disposition_time"].reindex(f.index) - f).dt.total_seconds() / 60
        lead_stats[str(th)] = {"share_of_admits_flagged": round(len(f) / len(te), 3), "median_lead_min": round(float(lead.median())),
                               "p25": round(float(lead.quantile(.25))), "p75": round(float(lead.quantile(.75)))}
    f6 = s[s["p"] >= C.PREP_EXPEDITE_THRESHOLD].groupby("visit_id")["snapshot_time"].min()
    lead = ((te["disposition_time"].reindex(f6.index) - f6).dt.total_seconds() / 60).reindex(te.index).fillna(0).to_numpy()
    mins = lambda a, b: (te[a] - te[b]).dt.total_seconds().to_numpy() / 60
    s1, s2, s3, s4 = (mins("bed_requested_time", "disposition_time"), mins("bed_matched_time", "bed_requested_time"),
                      mins("bed_ready_time", "bed_matched_time"), mins("left_ed_time", "bed_ready_time"))
    tot = s1 + s2 + s3 + s4
    H = USABLE * lead
    out = {"n_admits_test": int(len(te)), "baseline_median_h": round(float(np.median(tot) / 60), 2), "lead_time_by_threshold": lead_stats, "scenarios": {}}
    for name, (r, sv, fr) in SCENARIOS.items():
        save1 = np.minimum(H, s1)
        save3 = np.minimum(np.maximum(H - s1, 0), s3)
        new = tot - r * sv * (save1 + save3)
        new = new - fr * np.maximum(s3 - r * sv * save3, 0)
        out["scenarios"][name] = {"median_boarding_h": round(float(np.median(new) / 60), 2), "change": round(float(np.median(new) / np.median(tot) - 1), 3),
                                  "r": r, "s": sv, "f": fr}
    with open(f"{C.DATA_DIR}/scenario_results.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    o = run()
    print(f"baseline median boarding on the test split: {o['baseline_median_h']} h  ({o['n_admits_test']} admits)")
    for k, x in o["lead_time_by_threshold"].items():
        print(f"threshold {k}: flagged {x['share_of_admits_flagged']:.0%} of admits, median lead {x['median_lead_min']} min (P25 {x['p25']}, P75 {x['p75']})")
    for k, x in o["scenarios"].items():
        print(f"{k:13s} {x['median_boarding_h']} h  ({x['change']:+.0%})   r={x['r']} s={x['s']} f={x['f']}")
