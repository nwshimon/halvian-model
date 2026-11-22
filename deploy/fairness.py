"""
fairness.py -- does the system treat payer groups differently?  (test split only; demo rows excluded)

Payer is never a model input (config.PROHIBITED_FEATURES), so any gap here is a gap in the *data the model sees*
or in how the ED runs, not a rule we wrote. We still measure it, because leaving a feature out does not guarantee fair results.

Metrics per payer group (config.FAIRNESS_GROUPS), each compared with the overall value:
  mean_admit_score       time-weighted mean predicted admit probability
  calibration_gap        mean score minus actual admit rate (a group the model over/under-predicts for)
  false_negative_rate    admitted visits whose LAST score before the decision was "unlikely" (<0.20)
  median_boarding_hours  admitted visits, observed
  lwbs_rate, median_door_to_provider_min   ED access, observed (not model output)

Flag rule (config): abs(group / overall - 1) > FAIRNESS_TOLERANCE.  Calibration gap uses the absolute gap in points.
Each flag also gets a 95% bootstrap interval (resampling visits) so a flag on a small group is marked "uncertain".
Run:  python fairness.py
"""
import json

import numpy as np
import pandas as pd

import admit_model as AM
import config as C

MIN_N = 30  # fewer than this many visits (or admits for admit-only metrics) in a group: show but never call it a finding


def per_visit_table(data_dir=C.DATA_DIR):
    snap = pd.read_csv(f"{data_dir}/snapshots.csv", parse_dates=["snapshot_time"])
    visits = pd.read_csv(f"{data_dir}/visits.csv", parse_dates=["disposition_time", "left_ed_time"])
    snap = snap[snap["split"] == "test"].copy()
    model = AM.AdmitModel.load()
    snap["p"] = model.predict_proba(snap)
    snap["w"] = AM.time_weights(snap, visits)
    snap = snap.sort_values(["visit_id", "snapshot_time"], kind="stable")
    g = snap.groupby("visit_id")
    pv = pd.DataFrame({"mean_score": g.apply(lambda d: np.average(d["p"], weights=d["w"]), include_groups=False),
                       "last_score": g["p"].last()})
    v = visits[visits["split"] == "test"].set_index("visit_id")
    pv = pv.join(v[["payer", "admitted", "lwbs", "door_to_provider_min", "boarding_hours"]], how="inner")
    pv["admitted"] = pv["admitted"].astype(bool)
    pv["lwbs"] = pv["lwbs"].astype(bool)
    pv["miss"] = pv["admitted"] & (pv["last_score"] < C.CATEGORY_THRESHOLDS["possible"])
    return pv


def _metrics(d):
    adm = d[d["admitted"]]
    return {"visits": len(d), "admits": len(adm),
            "mean_admit_score": d["mean_score"].mean(),
            "calibration_gap": d["mean_score"].mean() - d["admitted"].mean(),
            "false_negative_rate": adm["miss"].mean() if len(adm) else np.nan,
            "median_boarding_hours": adm["boarding_hours"].median() if len(adm) else np.nan,
            "lwbs_rate": d["lwbs"].mean(),
            "median_door_to_provider_min": d["door_to_provider_min"].median()}


ABS_METRICS = {"calibration_gap"}          # compared as an absolute difference in points, not a ratio
ADMIT_ONLY = {"false_negative_rate", "median_boarding_hours"}


def _diff(group, overall, metric):
    if metric in ABS_METRICS:
        return group - overall
    return group / overall - 1 if overall else np.nan


def _flag(diff, metric):
    tol = C.FAIRNESS_TOLERANCE if metric not in ABS_METRICS else 0.02   # [ASSUMPTION] 2 points for the calibration gap
    return bool(abs(diff) > tol)


def run(n_boot=300, seed=0):
    pv = per_visit_table()
    overall = _metrics(pv)
    rng = np.random.default_rng(seed)
    rows = []
    for grp in C.FAIRNESS_GROUPS:
        d = pv[pv["payer"] == grp]
        m = _metrics(d)
        boots = {k: [] for k in overall if k not in ("visits", "admits")}
        idx_all, idx_g = np.arange(len(pv)), np.flatnonzero((pv["payer"] == grp).to_numpy())
        for _ in range(n_boot):
            bs = pv.iloc[rng.choice(idx_all, len(idx_all))]
            bg = bs[bs["payer"] == grp]
            if len(bg) < 5:
                continue
            mo, mg = _metrics(bs), _metrics(bg)
            for k in boots:
                boots[k].append(_diff(mg[k], mo[k], k))
        for k in boots:
            diff = _diff(m[k], overall[k], k)
            lo, hi = np.nanpercentile(boots[k], [2.5, 97.5]) if boots[k] else (np.nan, np.nan)
            n_ok = m["admits"] if k in ADMIT_ONLY else m["visits"]
            flagged = _flag(diff, k)
            certain = bool(flagged and (lo > 0 or hi < 0) and n_ok >= MIN_N)   # interval excludes "no difference"
            rows.append({"group": grp, "metric": k, "group_value": m[k], "overall_value": overall[k], "difference": diff,
                         "ci_low": lo, "ci_high": hi, "n": n_ok, "flag": flagged, "finding": certain,
                         "status": ("FINDING" if certain else "uncertain" if flagged else "ok")})
    return pd.DataFrame(rows), overall, pv


def twin_check():
    """The demo's payer twins (same patient, different payer) must get identical scores."""
    try:
        man = json.load(open(f"{C.DATA_DIR}/demo_manifest.json"))
    except FileNotFoundError:
        return None
    return man.get("payer_twins")


def main():
    df, overall, pv = run()
    print(f"test visits {overall['visits']:,}  admits {overall['admits']:,}  (tolerance {C.FAIRNESS_TOLERANCE:.0%})\n")
    for k, g in df.groupby("metric", sort=False):
        print(f"== {k}   overall {g['overall_value'].iloc[0]:.3f}")
        for _, r in g.iterrows():
            print(f"   {r['group']:<14} {r['group_value']:7.3f}  diff {r['difference']:+7.1%}  "
                  f"[{r['ci_low']:+.1%}, {r['ci_high']:+.1%}]  n={int(r['n']):>6}  {r['status']}")
    df.to_csv(f"{C.DATA_DIR}/fairness_report.csv", index=False)
    print(f"\nwrote {C.DATA_DIR}/fairness_report.csv")


if __name__ == "__main__":
    main()
