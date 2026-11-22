"""
build_snapshots.py -- Step 2: turn visits + events + vitals + occupancy into the model's training rows.

One snapshot row = "what the ED knew about one patient at one moment".
A snapshot is taken at every event in config.SCORE_ON_EVENTS (triage, order, result).

KNOWN-AS-OF RULE: every feature uses only events and vitals with time <= snapshot_time.
The label (visits.admitted) is copied onto every snapshot of the visit and is never a feature.
Run:  python build_snapshots.py      (reads ./data/*.csv, writes ./data/snapshots.csv, prints a report)
"""
import os

import numpy as np
import pandas as pd

import config as C
import schema as S

VITAL_MAX = {"hr": "max_hr_so_far", "rr": "max_rr_so_far", "temp_c": "max_temp_so_far"}
VITAL_MIN = {"spo2": "min_spo2_so_far", "sbp": "min_sbp_so_far"}


def build_snapshots(visits, events, vitals, occupancy):
    # ---- A. running totals over events (sorted by time; ties resolved by merge_asof taking the LAST tied row,
    #         so every event at or before snapshot_time is counted and nothing after it is)
    ev = events.sort_values(["event_time", "event_id"], kind="stable").copy()
    is_order = ev["event_type"].eq("order")
    is_result = ev["event_type"].eq("result")
    ev["f_orders"] = is_order
    ev["f_ct"] = is_order & ev["order_name"].eq("CT")
    ev["f_orders_w_result"] = is_order & ~ev["order_name"].isin(C.ORDERS_WITHOUT_RESULT)
    ev["f_results"] = is_result
    ev["f_abn"] = is_result & ev["result_abnormal"].eq(True)
    for c in ("f_orders", "f_ct", "f_orders_w_result", "f_results", "f_abn"):
        ev["c_" + c] = ev.groupby("visit_id")[c].cumsum()
    ev["_assess"] = ev["event_time"].where(ev["event_type"].isin(["triage", "reassessment"]))
    ev["last_assess_time"] = ev.groupby("visit_id")["_assess"].ffill()
    running = ev[["visit_id", "event_time", "c_f_orders", "c_f_ct", "c_f_orders_w_result",
                  "c_f_results", "c_f_abn", "last_assess_time"]].rename(columns={"event_time": "_et"})

    # ---- B. the snapshot rows themselves
    trig = ev[ev["event_type"].isin(C.SCORE_ON_EVENTS)][["event_id", "visit_id", "event_time", "event_type"]]
    trig = trig.rename(columns={"event_id": "trigger_event_id", "event_time": "snapshot_time",
                                "event_type": "trigger_type"}).sort_values("snapshot_time", kind="stable")
    snap = pd.merge_asof(trig, running.sort_values("_et", kind="stable"), left_on="snapshot_time", right_on="_et",
                         by="visit_id", direction="backward")

    # ---- C. running vitals extremes and "has this vital ever been abnormal"
    vt = vitals.sort_values("recorded_time", kind="stable").copy()
    flag_cols = []
    for v, (lo, hi) in C.VITAL_ABNORMAL.items():
        bad = pd.Series(False, index=vt.index)
        if lo is not None:
            bad |= vt[v] < lo
        if hi is not None:
            bad |= vt[v] > hi
        vt["ab_" + v] = bad
        flag_cols.append("ab_" + v)
    keep = ["visit_id", "recorded_time"]
    g = vt.groupby("visit_id")
    for v, name in VITAL_MAX.items():
        vt[name] = g[v].cummax(); keep.append(name)
    for v, name in VITAL_MIN.items():
        vt[name] = g[v].cummin(); keep.append(name)
    for c in flag_cols:
        vt["ever_" + c] = vt.groupby("visit_id")[c].cummax().astype(int)
    vt["abnormal_vital_count_so_far"] = vt[["ever_" + c for c in flag_cols]].sum(axis=1)
    keep.append("abnormal_vital_count_so_far")
    snap = pd.merge_asof(snap.sort_values("snapshot_time", kind="stable"), vt[keep].sort_values("recorded_time", kind="stable"),
                         left_on="snapshot_time", right_on="recorded_time", by="visit_id", direction="backward")

    # ---- D. visit-level facts known at triage, hospital context, label
    vis = visits[["visit_id", "arrival_time", "age_band", "esi", "complaint_category", "admitted", "split"]]
    snap = snap.merge(vis, on="visit_id", how="left", validate="many_to_one")
    occ = occupancy.rename(columns={"hour_start": "_hour", "inpatient_occupancy": "inpatient_occupancy_at_snapshot"})
    snap["_hour"] = snap["snapshot_time"].dt.floor("h")
    snap = snap.merge(occ, on="_hour", how="left", validate="many_to_one")

    snap["orders_so_far"] = snap["c_f_orders"].astype(int)
    snap["ct_orders_so_far"] = snap["c_f_ct"].astype(int)
    snap["abnormal_results_so_far"] = snap["c_f_abn"].astype(int)
    snap["pending_results_so_far"] = (snap["c_f_orders_w_result"] - snap["c_f_results"]).astype(int)
    snap["mins_since_arrival"] = (snap["snapshot_time"] - snap["arrival_time"]).dt.total_seconds() / 60
    snap["mins_since_last_reassessment"] = (snap["snapshot_time"] - snap["last_assess_time"]).dt.total_seconds() / 60
    snap["label_admit"] = snap["admitted"].astype(bool)
    snap["esi"] = snap["esi"].astype("Int64")

    snap = snap.sort_values(["visit_id", "snapshot_time", "trigger_event_id"], kind="stable").reset_index(drop=True)
    snap["snapshot_id"] = [f"S-{i + 1:07d}" for i in range(len(snap))]
    return snap[S.cols("snapshots")]


def verify_known_as_of(snap, visits, events, vitals, occupancy, n=400, seed=7):
    """Recompute features for a random sample of snapshots the slow, obvious way (filter to time <= t, then
    count) and compare. A mismatch means a feature peeked at the future or miscounted."""
    rng = np.random.default_rng(seed)
    rows = snap.iloc[rng.choice(len(snap), size=min(n, len(snap)), replace=False)]
    v = visits.set_index("visit_id")
    occ = occupancy.set_index("hour_start")["inpatient_occupancy"]
    ev_by, vt_by = dict(tuple(events.groupby("visit_id"))), dict(tuple(vitals.groupby("visit_id")))
    bad = []
    for r in rows.itertuples(index=False):
        e = ev_by[r.visit_id]; e = e[e["event_time"] <= r.snapshot_time]
        t = vt_by[r.visit_id]; t = t[t["recorded_time"] <= r.snapshot_time]
        orders = e[e["event_type"] == "order"]
        results = e[e["event_type"] == "result"]
        assess = e[e["event_type"].isin(["triage", "reassessment"])]["event_time"].max()
        n_abn = 0
        for x, (lo, hi) in C.VITAL_ABNORMAL.items():
            low = lo is not None and bool((t[x] < lo).any())
            high = hi is not None and bool((t[x] > hi).any())
            n_abn += int(low or high)
        want = {"max_hr_so_far": t["hr"].max(), "min_spo2_so_far": t["spo2"].min(), "min_sbp_so_far": t["sbp"].min(),
                "max_rr_so_far": t["rr"].max(), "max_temp_so_far": t["temp_c"].max(),
                "abnormal_vital_count_so_far": n_abn, "orders_so_far": len(orders),
                "ct_orders_so_far": int((orders["order_name"] == "CT").sum()),
                "abnormal_results_so_far": int(results["result_abnormal"].eq(True).sum()),
                "pending_results_so_far": int((~orders["order_name"].isin(C.ORDERS_WITHOUT_RESULT)).sum()) - len(results),
                "mins_since_arrival": (r.snapshot_time - v.loc[r.visit_id, "arrival_time"]).total_seconds() / 60,
                "mins_since_last_reassessment": (r.snapshot_time - assess).total_seconds() / 60,
                "inpatient_occupancy_at_snapshot": occ[r.snapshot_time.floor("h")]}
        for k, w in want.items():
            if not np.isclose(getattr(r, k), w, atol=1e-6):
                bad.append((r.snapshot_id, k, getattr(r, k), w))
    return bad


def report(snap, visits, events, vitals, occupancy):
    problems = S.validate(snap, "snapshots") + S.check_snapshots(snap, visits, events)
    for f in C.MODEL_FEATURES:
        if snap[f].isna().any():
            problems.append(f"snapshots.{f}: has nulls")
    mism = verify_known_as_of(snap, visits, events, vitals, occupancy)
    if mism:
        problems.append(f"known-as-of recompute disagreed on {len(mism)} values, e.g. {mism[:3]}")

    nv = len(visits)
    print(f"\n=== Step 2 report: {len(snap):,} snapshots from {nv:,} visits ({len(snap) / nv:.2f} per visit) ===")
    print("by trigger:", snap["trigger_type"].value_counts().to_dict())
    per_visit = snap.groupby("visit_id").size()
    print(f"snapshots per visit: median {per_visit.median():.0f}, max {per_visit.max()}")
    by_split = snap["split"].value_counts()
    print("snapshots by split:", {k: int(by_split[k]) for k in C.SPLIT},
          "| visits by split:", visits["split"].value_counts().reindex(list(C.SPLIT)).to_dict())
    print(f"admit label: {snap['label_admit'].mean():.1%} of snapshot rows vs {visits['admitted'].mean():.1%} of visits "
          "(admits have more orders and results, so more rows; the split is by visit so this is fine)")
    print("\nfeature ranges (min / median / max):")
    for f in C.MODEL_FEATURES:
        if FEATURE_NUMERIC(f):
            print(f"   {f:34s} {snap[f].min():8.2f} {snap[f].median():8.2f} {snap[f].max():8.2f}")
    print(f"\nknown-as-of brute-force recompute of 400 random snapshots: "
          f"{'all 12 numeric features identical' if not mism else str(len(mism)) + ' mismatches'}")

    # sanity: does the model get better as information arrives? Fit on TRAIN, score on VALIDATION (test stays untouched).
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        cat = ["age_band", "complaint_category", "esi"]
        num = [f for f in C.MODEL_FEATURES if f not in cat]
        X = snap[C.MODEL_FEATURES].copy(); X["esi"] = X["esi"].astype(str)
        pipe = make_pipeline(ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat),
                                                ("n", StandardScaler(), num)]),
                             LogisticRegression(C=C.LOGREG_C, max_iter=3000))
        tr, va = (snap["split"] == "train").to_numpy(), (snap["split"] == "validation").to_numpy()
        pipe.fit(X[tr], snap.loc[tr, "label_admit"])
        p = pipe.predict_proba(X[va])[:, 1]
        y = snap.loc[va, "label_admit"].to_numpy()
        tt = snap.loc[va, "trigger_type"].to_numpy()
        print(f"\nAUC sanity (fit on train, scored on validation; the real model is Step 3):")
        print(f"   all validation snapshots   {roc_auc_score(y, p):.3f}   expected range {C.EXPECTED_AUC_RANGE}")
        for t in C.SCORE_ON_EVENTS:
            print(f"   {t:8s} snapshots only    {roc_auc_score(y[tt == t], p[tt == t]):.3f}   (n={int((tt == t).sum()):,})")
        last = snap.loc[va].groupby("visit_id").tail(1).index
        lm = snap.index.isin(last)[va]
        print(f"   last snapshot of each visit {roc_auc_score(y[lm], p[lm]):.3f}")
    except Exception as e:
        print("AUC check skipped:", e)

    ex = snap[snap["visit_id"] == "V-000014"]
    show = ["snapshot_id", "trigger_type", "snapshot_time", "mins_since_arrival", "orders_so_far",
            "abnormal_results_so_far", "pending_results_so_far", "min_spo2_so_far", "label_admit"]
    print("\nexample: all snapshots of admitted patient V-000014")
    print(ex[show].to_string(index=False))
    print("\nPROBLEMS:", problems if problems else "none")
    return problems


def FEATURE_NUMERIC(f):
    return S.FEATURE_DTYPES[f] in ("int", "float") and f != "esi"


def load():
    d = C.DATA_DIR
    visits = pd.read_csv(f"{d}/visits.csv", parse_dates=["arrival_time", "disposition_time", "bed_requested_time",
                                                       "bed_matched_time", "bed_ready_time", "left_ed_time"])
    events = pd.read_csv(f"{d}/events.csv", parse_dates=["event_time"])
    vitals = pd.read_csv(f"{d}/vitals.csv", parse_dates=["recorded_time"])
    occ = pd.read_csv(f"{d}/occupancy_hourly.csv", parse_dates=["hour_start"])
    return visits, events, vitals, occ


if __name__ == "__main__":
    visits, events, vitals, occ = load()
    snap = build_snapshots(visits, events, vitals, occ)
    problems = report(snap, visits, events, vitals, occ)
    snap.to_csv(os.path.join(C.DATA_DIR, "snapshots.csv"), index=False)
    print(f"\nWrote {C.DATA_DIR}/snapshots.csv")