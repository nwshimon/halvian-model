"""
build_demo.py -- demo day: the 40-bed board, 30 ED patients caught mid-flight at DEMO_NOW, and 3 scripted timelines.

How the demo world is made
  * 27 background patients are drawn from a SECOND run of the Step 1 generator (config.DEMO_POOL_SEED), picked by where they
    are at DEMO_NOW (waiting, in work-up, awaiting discharge, boarding) and shifted in time to land there. They are real
    generator output, so they obey every rule the history does.
  * 1 of the work-up patients has a payer twin: identical clinically, different payer. Their scores must match (fairness demo).
  * 3 scripted patients (A, B, C) are hand-built so the live demo has a story. Their numbers are then CHECKED against the models.
  * 30 occupied beds hold real admitted visits from the last days of history; beds and occupants agree on level of care.
Events after DEMO_NOW are the "what happened without the tool" replay; the live tables (bed_matches, tasks, audit_log) start empty.

Order:  generate_visits.py -> admit_model.py (needs snapshots) ... -> build_demo.py -> build_snapshots.py
(re-run build_snapshots.py afterwards so snapshots.csv includes the demo patients; they carry split='demo' and are never trained on)
Run:  python build_demo.py
"""
import json
import os

import numpy as np
import pandas as pd

import admit_model as AM
import bed_rules as BR
import build_snapshots as BS
import config as C
import demand_forecast as DF
import generate_visits as GV
import schema as S

DAY = pd.Timestamp(C.DEMO_DATE)
NOW = DAY + pd.Timedelta(hours=int(C.DEMO_NOW_TIME[:2]), minutes=int(C.DEMO_NOW_TIME[3:]))
TIME_COLS = ["arrival_time", "disposition_time", "bed_requested_time", "bed_matched_time", "bed_ready_time", "left_ed_time"]
ROLE = {"arrival": None, "triage": "ed_charge_nurse", "provider_seen": "ed_physician", "order": "ed_physician", "result": None,
        "reassessment": "ed_charge_nurse", "admit_decision": "ed_physician", "bed_requested": "ed_charge_nurse",
        "bed_matched": "bed_manager", "bed_ready": "bed_manager", "report_given": "inpatient_nurse",
        "transported": "ed_charge_nurse", "departed": "ed_charge_nurse"}
TYPE_ORDER = {t: i for i, t in enumerate(S.EVENT_TYPES)}


def T(hhmm):
    """'09:12' or '09:12:30' on the demo day."""
    parts = [int(x) for x in hhmm.split(":")] + [0]
    return DAY + pd.Timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])


# ------------------------------------------------------------------ occupancy for the days between history and demo
def extend_occupancy(occ, rng):
    last = occ.iloc[-1]
    start = last["hour_start"] + pd.Timedelta(hours=1)
    hours = pd.date_range(start, DAY + pd.Timedelta(days=2) - pd.Timedelta(hours=1), freq="h")
    mean = lambda h: C.INPATIENT_OCCUPANCY + C.OCCUPANCY_DAILY_AMPLITUDE * np.cos(2 * np.pi * ((h % 24) - 15) / 24)
    x = last["inpatient_occupancy"] - mean(last["hour_start"].hour)
    vals = []
    for h in hours:
        x = C.OCCUPANCY_AR1 * x + rng.normal(0, C.OCCUPANCY_NOISE_SD)
        vals.append(float(np.clip(mean(h.hour) + x, *C.OCCUPANCY_LIMITS)))
    return pd.concat([occ, pd.DataFrame({"hour_start": hours, "inpatient_occupancy": vals})], ignore_index=True)


# ------------------------------------------------------------------ pool: stage times for every pool visit
def pool_wide(pv, pe):
    first = pe.groupby(["visit_id", "event_type"])["event_time"].min().unstack()
    w = pv.set_index("visit_id").join(first.add_prefix("t_"))
    act = pe[pe["event_type"].isin(["provider_seen", "order", "result"])].groupby("visit_id")["event_time"].max()
    w["last_activity"] = act
    n_orders = pe[pe["event_type"] == "order"].groupby("visit_id").size()
    w["n_orders"] = n_orders.reindex(w.index).fillna(0).astype(int)
    return w


def anchors(w, rng, first_order):
    """For each archetype: Series of the pool time that should be moved onto DEMO_NOW (NaT = ineligible)."""
    mins = lambda a, b: (a - b).dt.total_seconds() / 60
    u = lambda lo, hi: pd.Series(rng.uniform(lo, hi, len(w)), index=w.index)
    out = {}
    ok = w["t_provider_seen"].notna() & ~w["lwbs"].astype(bool)
    gap = mins(w["t_provider_seen"], w["t_triage"])
    out["waiting_for_provider"] = (w["t_triage"] + pd.to_timedelta(u(0.3, 0.9) * gap, unit="m")).where(ok & (gap >= 15))
    end = w["disposition_time"].fillna(w["left_ed_time"])
    span = mins(end, w["t_provider_seen"])
    a = w["t_provider_seen"] + pd.to_timedelta(u(0.25, 0.85) * span, unit="m")
    a = pd.concat([a, first_order + pd.Timedelta(minutes=1)], axis=1).max(axis=1)
    out["in_workup"] = a.where(ok & (w["n_orders"] >= 1) & (span >= 40) & (a < end - pd.Timedelta(minutes=2)))
    gap = mins(w["left_ed_time"], w["last_activity"])
    off = 8 + (np.minimum(40, gap - 1) - 8).clip(lower=0) * pd.Series(rng.random(len(w)), index=w.index)   # minutes before departure
    out["awaiting_discharge"] = (w["left_ed_time"] - pd.to_timedelta(off, unit="m")).where(
        ok & (w["disposition"] == "discharge") & (gap >= 12))
    adm = w["admitted"].astype(bool)
    gap = (w["t_bed_requested"] - w["t_admit_decision"]).dt.total_seconds()
    out["boarding_decided"] = (w["t_admit_decision"] + pd.to_timedelta(u(0.2, 0.8) * gap, unit="s")).where(adm & (gap >= 100))
    gap = mins(w["t_bed_matched"], w["t_bed_requested"])
    out["boarding_requested"] = (w["t_bed_requested"] + pd.to_timedelta(u(0.15, 0.85) * gap, unit="m")).where(adm & (gap >= 10))
    return out



# ------------------------------------------------------------------ helpers for building frames
def shift_frames(ids, delta, pv, pe, pt):
    d = pd.Series(delta)
    v = pv[pv["visit_id"].isin(ids)].copy()
    e = pe[pe["visit_id"].isin(ids)].copy()
    t = pt[pt["visit_id"].isin(ids)].copy()
    for c in TIME_COLS:
        v[c] = v[c] + v["visit_id"].map(d)
    e["event_time"] = e["event_time"] + e["visit_id"].map(d)
    t["recorded_time"] = t["recorded_time"] + t["visit_id"].map(d)
    v["split"] = "demo"
    return v, e, t


def latest_scores(am, snaps):
    s = snaps[snaps["snapshot_time"] <= NOW].sort_values("snapshot_time", kind="stable").groupby("visit_id").tail(1).copy()
    s["p"] = am.predict_proba(s)
    s["category"] = AM.AdmitModel.category(s["p"].to_numpy())
    return s.set_index("visit_id")


# ------------------------------------------------------------------ scripted patients
SCRIPTS = {
    "A": dict(  # rising risk: looks mild at first, then abnormal results and worsening vitals
        title="Looks mild, then isn't", age_band="35-49", payer="commercial", complaint="abdominal_pain", esi=3,
        arrival="09:12", triage="09:24", provider="09:58",
        triage_vitals=dict(hr=92, spo2=96.0, sbp=128, rr=18.0, temp_c=37.4),
        reassess=[("10:39", dict(hr=122, spo2=92.5, sbp=96, rr=25.0, temp_c=38.9)),
                  ("11:42", dict(hr=126, spo2=92.0, sbp=92, rr=27.0, temp_c=39.1))],
        orders=[("lab_panel", "10:08", "10:46", True), ("IV_medication", "10:10", None, None), ("CT", "10:14", "11:20", True)],
        decision="11:34", requested="11:37", matched="11:47", ready="12:40", report="12:58", transported="13:15",
        level="telemetry", isolation=False, preferred=None),
    "B": dict(  # stalled stage: bed requested, nobody matched it in time; the only fit is an isolation telemetry bed
        title="Bed request stalls", age_band="65-74", payer="medicare", complaint="shortness_of_breath", esi=2,
        arrival="08:02", triage="08:09", provider="08:32",
        triage_vitals=dict(hr=104, spo2=91.0, sbp=142, rr=26.0, temp_c=38.2),
        reassess=[("09:22", dict(hr=98, spo2=93.0, sbp=138, rr=24.0, temp_c=37.9)),
                  ("10:35", dict(hr=96, spo2=94.0, sbp=136, rr=22.0, temp_c=37.8))],
        orders=[("ECG", "08:40", "08:50", False), ("chest_xray", "08:42", "09:20", True),
                ("lab_panel", "08:42", "09:35", True), ("IV_medication", "08:50", None, None)],
        decision="10:05", requested="10:10", matched="10:52", ready="11:40", report="11:58", transported="12:15",
        level="telemetry", isolation=True, preferred="pulmonary"),
    "C": dict(  # human override: the AI's top bed is rejected by the charge nurse, who owns the decision
        title="Charge nurse overrides", age_band="50-64", payer="uninsured", complaint="chest_pain", esi=2,
        arrival="08:40", triage="08:47", provider="09:12",
        triage_vitals=dict(hr=96, spo2=95.0, sbp=158, rr=20.0, temp_c=36.9),
        reassess=[("09:55", dict(hr=90, spo2=96.0, sbp=150, rr=18.0, temp_c=36.9))],
        orders=[("ECG", "09:20", "09:30", True), ("troponin", "09:20", "10:05", True),
                ("lab_panel", "09:22", "10:00", False), ("IV_medication", "09:25", None, None)],
        decision="10:20", requested="10:24", matched="10:41", ready="11:10", report="11:28", transported="11:45",
        level="telemetry", isolation=False, preferred="cardiac"),
}


def scripted_frames(key, spec):
    vid = f"SCRIPT-{key}"
    ev = [("arrival", T(spec["arrival"]), None, None), ("triage", T(spec["triage"]), None, None),
          ("provider_seen", T(spec["provider"]), None, None)]
    for name, t_o, t_r, abn in spec["orders"]:
        ev.append(("order", T(t_o), name, None))
        if t_r:
            ev.append(("result", T(t_r), name, bool(abn)))
    for t_r, _ in spec["reassess"]:
        ev.append(("reassessment", T(t_r), None, None))
    for et, k in [("admit_decision", "decision"), ("bed_requested", "requested"), ("bed_matched", "matched"),
                  ("bed_ready", "ready"), ("report_given", "report"), ("transported", "transported")]:
        ev.append((et, T(spec[k]), None, None))
    e = pd.DataFrame(ev, columns=["event_type", "event_time", "order_name", "result_abnormal"])
    e["visit_id"] = vid
    e["actor_role"] = e["event_type"].map(ROLE)
    e["result_abnormal"] = e["result_abnormal"].astype("boolean")
    vit = [(T(spec["triage"]), spec["triage_vitals"])] + [(T(t), v) for t, v in spec["reassess"]]
    t = pd.DataFrame([{"visit_id": vid, "recorded_time": ts, **{k: float(x) for k, x in v.items()}} for ts, v in vit])
    dec, left = T(spec["decision"]), T(spec["transported"])
    v = pd.DataFrame([{
        "visit_id": vid, "arrival_time": T(spec["arrival"]), "age_band": spec["age_band"], "payer": spec["payer"],
        "complaint_category": spec["complaint"], "esi": spec["esi"], "requested_level_of_care": spec["level"],
        "isolation_required": spec["isolation"], "preferred_unit": spec["preferred"], "disposition": "admit",
        "admitted": True, "lwbs": False,
        "door_to_provider_min": (T(spec["provider"]) - T(spec["arrival"])).total_seconds() / 60,
        "disposition_time": dec, "bed_requested_time": T(spec["requested"]), "bed_matched_time": T(spec["matched"]),
        "bed_ready_time": T(spec["ready"]), "left_ed_time": left, "boarding_hours": (left - dec).total_seconds() / 3600,
        "split": "demo"}])
    return v, e, t


# ------------------------------------------------------------------ beds
# (bed_id, unit, bed_type, isolation_capable, status, discharge_pending, minutes until ready)
BED_SPEC = (
    [("ICU-01", "icu", "icu", True, "occupied", False, None), ("ICU-02", "icu", "icu", True, "occupied", False, None),
     ("ICU-03", "icu", "icu", False, "ready", False, 0), ("ICU-04", "icu", "icu", False, "cleaning", False, 25),
     ("ICU-05", "icu", "icu", False, "occupied", True, 90), ("ICU-06", "icu", "icu", False, "occupied", False, None)]
    + [("CARD-01", "cardiac", "telemetry", False, "occupied", False, None), ("CARD-02", "cardiac", "telemetry", False, "occupied", False, None),
       ("CARD-03", "cardiac", "telemetry", True, "ready", False, 0), ("CARD-04", "cardiac", "telemetry", False, "occupied", True, 50),
       ("CARD-05", "cardiac", "telemetry", False, "cleaning", False, 15), ("CARD-06", "cardiac", "telemetry", False, "occupied", False, None),
       ("CARD-07", "cardiac", "telemetry", False, "occupied", False, None), ("CARD-08", "cardiac", "telemetry", False, "occupied", False, None),
       ("CARD-09", "cardiac", "telemetry", True, "occupied", True, 60), ("CARD-10", "cardiac", "telemetry", False, "occupied", False, None),
       ("CARD-11", "cardiac", "telemetry", False, "ready", False, 0), ("CARD-12", "cardiac", "telemetry", False, "blocked", False, None)]
    + [("GEN-01", "general", "med_surg", False, "occupied", True, 45), ("GEN-02", "general", "med_surg", True, "occupied", False, None),
       ("GEN-03", "general", "med_surg", False, "occupied", False, None), ("GEN-04", "general", "med_surg", False, "cleaning", False, 20),
       ("GEN-05", "general", "med_surg", False, "occupied", False, None), ("GEN-06", "general", "med_surg", False, "occupied", False, None),
       ("GEN-07", "general", "med_surg", False, "ready", False, 0), ("GEN-08", "general", "med_surg", False, "occupied", False, None),
       ("GEN-09", "general", "med_surg", False, "occupied", False, None), ("GEN-10", "general", "med_surg", False, "occupied", False, None)]
    + [("PULM-01", "pulmonary", "med_surg", True, "occupied", False, None), ("PULM-02", "pulmonary", "med_surg", False, "occupied", False, None),
       ("PULM-03", "pulmonary", "med_surg", False, "occupied", False, None), ("PULM-04", "pulmonary", "med_surg", False, "cleaning", False, 30),
       ("PULM-05", "pulmonary", "med_surg", False, "occupied", True, 75), ("PULM-06", "pulmonary", "med_surg", False, "occupied", False, None)]
    + [("NEURO-01", "neuro", "med_surg", False, "occupied", True, 110), ("NEURO-02", "neuro", "med_surg", True, "occupied", False, None),
       ("NEURO-03", "neuro", "med_surg", False, "occupied", False, None), ("NEURO-04", "neuro", "med_surg", False, "occupied", False, None),
       ("NEURO-05", "neuro", "med_surg", False, "occupied", False, None), ("NEURO-06", "neuro", "med_surg", False, "blocked", False, None)]
)
UNIT_COMPLAINTS = {"cardiac": ("chest_pain", "shortness_of_breath"), "neuro": ("neuro",), "pulmonary": ("shortness_of_breath",)}


def build_beds(hist_visits, rng):
    cutoff = pd.Timestamp(C.HISTORY_START_DATE) + pd.Timedelta(days=C.HISTORY_DAYS)
    pool = hist_visits[(hist_visits["disposition"] == "admit") & (hist_visits["left_ed_time"] <= cutoff)
                       & (hist_visits["left_ed_time"] >= cutoff - pd.Timedelta(days=C.DEMO_OCCUPANT_WINDOW_DAYS))]
    used, rows = set(), []
    for bed_id, unit, btype, iso, status, pending, ready_in in BED_SPEC:
        occupant, updated, expected = None, None, pd.NaT
        if status == "occupied":
            cand = pool[(pool["requested_level_of_care"] == btype) & ~pool["visit_id"].isin(used)]
            pref = cand[cand["complaint_category"].isin(UNIT_COMPLAINTS.get(unit, ()))]
            pick = (pref if len(pref) else cand).sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            occupant = pick["visit_id"]; used.add(occupant)
            updated = pick["left_ed_time"] + pd.Timedelta(minutes=15)
            if pending:
                expected = NOW + pd.Timedelta(minutes=ready_in)
        elif status == "ready":
            updated = NOW - pd.Timedelta(minutes=int(rng.integers(10, 120)))
            expected = updated
        elif status == "cleaning":
            updated = NOW - pd.Timedelta(minutes=int(rng.integers(3, 15)))
            expected = NOW + pd.Timedelta(minutes=ready_in)
        else:
            updated = NOW - pd.Timedelta(hours=int(rng.integers(20, 70)))
        rows.append({"bed_id": bed_id, "unit": unit, "bed_type": btype, "isolation_capable": iso, "status": status,
                     "discharge_pending": pending, "current_visit_id": occupant, "status_updated_time": updated.floor("s"),
                     "expected_ready_time": expected})
    return pd.DataFrame(rows)


def check_beds(beds):
    bad = S.validate(beds, "beds")
    cfg = C.BEDS_DEMO
    if len(beds) != sum(cfg["unit_counts"].values()):
        bad.append(f"beds: {len(beds)} rows, config says {sum(cfg['unit_counts'].values())}")
    for t, n in cfg["unit_counts"].items():
        if (beds["bed_type"] == t).sum() != n:
            bad.append(f"beds: {t} count {(beds['bed_type'] == t).sum()} != {n}")
        if (beds[beds["bed_type"] == t]["isolation_capable"]).sum() != cfg["isolation_capable"][t]:
            bad.append(f"beds: isolation count for {t} != {cfg['isolation_capable'][t]}")
    for st, n in cfg["status_counts"].items():
        if (beds["status"] == st).sum() != n:
            bad.append(f"beds: status {st} count != {n}")
    if (beds["discharge_pending"] & (beds["status"] == "occupied")).sum() != cfg["occupied_with_discharge_pending"]:
        bad.append("beds: discharge-pending count != config")
    if beds[beds["status"] != "occupied"]["current_visit_id"].notna().any():
        bad.append("beds: a non-occupied bed has an occupant")
    if (beds[beds["status"] == "occupied"]["current_visit_id"].duplicated()).any():
        bad.append("beds: an inpatient is in two beds")
    return bad


# ------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(C.RANDOM_SEED + 1)
    visits, events, vitals, occ = BS.load()
    # idempotent: drop any earlier demo rows
    old = visits[visits["split"] == "demo"]["visit_id"]
    visits, events, vitals = visits[~visits["visit_id"].isin(old)], events[~events["visit_id"].isin(old)], vitals[~vitals["visit_id"].isin(old)]
    occ = occ[occ["hour_start"] < pd.Timestamp(C.HISTORY_START_DATE) + pd.Timedelta(days=C.HISTORY_DAYS + 1)]
    occ_ext = extend_occupancy(occ, rng)
    am = AM.AdmitModel.load()

    print("Generating the demo pool (second generator run) ...")
    tables, _ = GV.generate(seed=C.DEMO_POOL_SEED)
    pv, pe, pt = tables["visits"], tables["events"], tables["vitals"]
    w = pool_wide(pv, pe)
    first_order = pe[pe["event_type"] == "order"].groupby("visit_id")["event_time"].min().reindex(w.index)
    anc = anchors(w, rng, first_order)

    # ---- candidates per archetype, shifted onto DEMO_NOW, then scored
    used = set()
    cand_frames = {}
    for arch, a in anc.items():
        elig = w[a.notna()].copy()
        if arch.startswith("boarding"):
            elig = elig[(elig["isolation_required"] == False)]            # noqa: E712 (nullable boolean)
            parts = [elig[elig["requested_level_of_care"] == lv].sample(min(25, (elig["requested_level_of_care"] == lv).sum()),
                                                                       random_state=int(rng.integers(1e9))) for lv in ("icu", "telemetry", "med_surg")]
            elig = pd.concat(parts)
        else:
            elig = elig.sample(min(160 if arch == "in_workup" else 40, len(elig)), random_state=int(rng.integers(1e9)))
        delta = (NOW - a[elig.index]).dt.round("s")
        cand_frames[arch] = (elig.index, delta)
    # one shifted copy per (visit, archetype): built separately so a visit eligible for two archetypes keeps both options
    arch_frames = {}
    snaps_by_arch = {}
    for arch, (ids, d) in cand_frames.items():
        v, e, t = shift_frames(set(ids), d.to_dict(), pv, pe, pt)
        s = BS.build_snapshots(v, e, t, occ_ext)
        arch_frames[arch] = (v, e, t)
        snaps_by_arch[arch] = latest_scores(am, s)

    def choose(arch, n, mask=None, seed_off=0):
        sc = snaps_by_arch[arch]
        pool_ids = [i for i in sc.index if i not in used and (mask is None or mask(sc.loc[i]))]
        r = np.random.default_rng(C.RANDOM_SEED + 100 + seed_off)
        got = list(r.choice(pool_ids, size=n, replace=False)) if len(pool_ids) >= n else None
        if got is None:
            raise RuntimeError(f"not enough candidates for {arch} (need {n}, have {len(pool_ids)})")
        used.update(got)
        return got

    level_of = lambda arch, i: arch_frames[arch][0].set_index("visit_id").loc[i, "requested_level_of_care"]
    sel = {"waiting_for_provider": choose("waiting_for_provider", C.DEMO_ARCHETYPES["waiting_for_provider"], seed_off=1)}
    cat = lambda band: (lambda row: row["category"] == band)
    twin_base = choose("in_workup", 1, mask=lambda r: r["category"] == "possible" and 0.25 <= r["p"] <= 0.50, seed_off=2)
    bands = C.DEMO_WORKUP_BANDS
    work = twin_base[:]
    for k, n in bands.items():
        need = n - 2 if k == "possible" else n               # the twin pair (base + clone) fills two 'possible' slots
        work += choose("in_workup", need, mask=cat(k), seed_off=3 + len(work))
    sel["in_workup"] = work
    sel["awaiting_discharge"] = choose("awaiting_discharge", C.DEMO_ARCHETYPES["awaiting_discharge"], seed_off=9)
    for arch in ("boarding_decided", "boarding_requested"):
        got = []
        for lv in C.DEMO_BOARDING_LEVELS[arch]:
            got += choose(arch, 1, mask=lambda r, lv=lv, arch=arch: level_of(arch, r.name) == lv, seed_off=20 + len(got))
        sel[arch] = got

    # ---- assemble demo frames
    V, E, TV, labels = [], [], [], {}
    for arch, ids in sel.items():
        v, e, t = arch_frames[arch]
        V.append(v[v["visit_id"].isin(ids)]); E.append(e[e["visit_id"].isin(ids)]); TV.append(t[t["visit_id"].isin(ids)])
        for i in ids:
            labels[i] = arch
    # payer twin: clone the first work-up patient, 5 minutes later, uninsured vs commercial
    base_id = twin_base[0]
    allV, allE, allT = pd.concat(V), pd.concat(E), pd.concat(TV)
    v0, e0, t0 = (x[x["visit_id"] == base_id].copy() for x in (allV, allE, allT))
    twin_id = "TWIN-" + base_id
    dt5 = pd.Timedelta(minutes=5)
    v1, e1, t1 = v0.copy(), e0.copy(), t0.copy()
    for fr, col in ((v1, None), (e1, "event_time"), (t1, "recorded_time")):
        fr["visit_id"] = twin_id
        if col:
            fr[col] = fr[col] + dt5
    for c in TIME_COLS:
        v1[c] = v1[c] + dt5
    v1["payer"] = "uninsured"
    for i, fr in enumerate(V):
        V[i] = fr.assign(payer=np.where(fr["visit_id"] == base_id, "commercial", fr["payer"]))
    V.append(v1); E.append(e1); TV.append(t1)
    labels[twin_id] = "in_workup_twin"
    for key, spec in SCRIPTS.items():
        v, e, t = scripted_frames(key, spec)
        V.append(v); E.append(e); TV.append(t)
        labels[f"SCRIPT-{key}"] = f"scripted_{key}"

    dv, de, dt = pd.concat(V, ignore_index=True), pd.concat(E, ignore_index=True), pd.concat(TV, ignore_index=True)
    dv["split"] = "demo"
    assert len(dv) == C.N_DEMO_PATIENTS, f"{len(dv)} demo patients, expected {C.N_DEMO_PATIENTS}"
    order = dv.sort_values("arrival_time", kind="stable")["visit_id"].tolist()
    new_id = {old_id: f"V-{900001 + i}" for i, old_id in enumerate(order)}
    dv["visit_id"] = dv["visit_id"].map(new_id)
    de["visit_id"] = de["visit_id"].map(new_id); dt["visit_id"] = dt["visit_id"].map(new_id)
    labels = {new_id[k]: v for k, v in labels.items()}
    de["_o"] = de["event_type"].map(TYPE_ORDER)
    de = de.sort_values(["visit_id", "event_time", "_o"], kind="stable").reset_index(drop=True)
    de["event_id"] = [f"E-{9000001 + i}" for i in range(len(de))]
    dt = dt.sort_values(["visit_id", "recorded_time"], kind="stable").reset_index(drop=True)
    dt["vitals_id"] = [f"VT-{9000001 + i}" for i in range(len(dt))]
    dv = dv.sort_values("visit_id").reset_index(drop=True)
    dv = dv[S.cols("visits")]; de = de[S.cols("events")]; dt = dt[S.cols("vitals")]
    for c in ("esi",):
        dv[c] = dv[c].astype("Int64")

    beds = build_beds(visits, rng)
    beds_problems = check_beds(beds)

    # ---- checks
    problems = list(beds_problems)
    allv = pd.concat([visits, dv], ignore_index=True)
    alle = pd.concat([events, de], ignore_index=True)
    allt = pd.concat([vitals, dt], ignore_index=True)
    tables_all = {"visits": allv, "events": alle, "vitals": allt, "occupancy_hourly": occ_ext, "beds": beds}
    for name, df in tables_all.items():
        problems += S.validate(df, name)
    problems += S.check_foreign_keys(tables_all)
    problems += S.check_visit_event_consistency(dv, de)
    occupants_ok = beds[beds["status"] == "occupied"]["current_visit_id"].isin(visits["visit_id"]).all()
    if not occupants_ok:
        problems.append("beds: an occupant is not a known visit")

    dsnap = BS.build_snapshots(dv, de, dt, occ_ext)
    problems += S.check_snapshots(dsnap, dv, de)
    sc = latest_scores(am, dsnap)
    dsnap["p"] = am.predict_proba(dsnap); dsnap["category"] = AM.AdmitModel.category(dsnap["p"].to_numpy())

    # stage of every patient at NOW, compared with what each archetype promised
    def stage(vid):
        e = de[(de["visit_id"] == vid) & (de["event_time"] <= NOW)]
        has = lambda t: (e["event_type"] == t).any()
        if has("departed") or has("transported"):
            return "gone"
        if has("bed_matched"):
            return "matched_or_later"
        if has("bed_requested"):
            return "boarding_requested"
        if has("admit_decision"):
            return "boarding_decided"
        if has("provider_seen") and (e["event_type"] == "order").any():
            return "in_workup"
        if has("provider_seen"):
            return "seen_no_orders"
        return "waiting_for_provider" if has("triage") else "untriaged"
    expect = {"waiting_for_provider": "waiting_for_provider", "in_workup": "in_workup", "in_workup_twin": "in_workup",
              "awaiting_discharge": "in_workup", "boarding_decided": "boarding_decided", "boarding_requested": "boarding_requested",
              "scripted_A": "in_workup", "scripted_B": "boarding_requested", "scripted_C": "boarding_requested"}
    stages = {vid: stage(vid) for vid in dv["visit_id"]}
    wrong = [(vid, labels[vid], stages[vid]) for vid in stages if stages[vid] != expect[labels[vid]]]
    if wrong:
        problems.append(f"patients not at their intended stage at DEMO_NOW: {wrong}")
    present = ((dv["arrival_time"] <= NOW) & (dv["left_ed_time"] > NOW)).sum()
    if present != C.N_DEMO_PATIENTS:
        problems.append(f"{present} of {C.N_DEMO_PATIENTS} demo patients are in the ED at DEMO_NOW")

    vid_of = {labels[v]: v for v in labels if labels[v].startswith(("scripted", "in_workup_twin"))}
    a_id, b_id, c_id = vid_of["scripted_A"], vid_of["scripted_B"], vid_of["scripted_C"]
    arcA = dsnap[dsnap["visit_id"] == a_id][["trigger_type", "snapshot_time", "p", "category"]]
    if not (arcA["category"].iloc[0] in ("unlikely", "possible") and arcA["category"].iloc[-1] == "likely"
            and arcA[arcA["snapshot_time"] <= NOW]["category"].iloc[-1] != "likely"):
        problems.append(f"script A: score arc does not go from not-likely at DEMO_NOW to likely: {arcA['category'].tolist()}")
    twin_new = new_id[twin_id]; base_new = new_id[base_id]
    pp = sc.loc[[base_new, twin_new], "p"]
    if abs(pp.iloc[0] - pp.iloc[1]) > 0.005:
        problems.append(f"payer twins score differently: {pp.tolist()}")

    boarders = dv[(dv["disposition_time"] <= NOW) & (dv["bed_matched_time"] > NOW)]
    recs, infeasible = {}, []
    for _, r in boarders.iterrows():
        pat = {"requested_level_of_care": r["requested_level_of_care"], "isolation_required": r["isolation_required"],
               "preferred_unit": r["preferred_unit"]}
        top = BR.rank_beds(pat, beds, NOW)
        recs[r["visit_id"]] = top
        if not top:
            infeasible.append(r["visit_id"])
    if infeasible:
        problems.append(f"boarders with no feasible bed: {infeasible}")
    expected_B, expected_C = ["CARD-03"], ["CARD-11", "CARD-03"]
    if [x["bed_id"] for x in recs[b_id]][:1] != expected_B or len(BR.feasible_beds(
            {"requested_level_of_care": "telemetry", "isolation_required": True}, beds)) < 2:
        problems.append(f"script B recommendation is not as designed: {recs[b_id]}")
    if [x["bed_id"] for x in recs[c_id]][:2] != expected_C:
        problems.append(f"script C recommendation is not as designed: {recs[c_id]}")

    # ---- forecast at DEMO_NOW over the demo census
    hist_snap = pd.read_csv(f"{C.DATA_DIR}/snapshots.csv", parse_dates=["snapshot_time"])
    hist_snap = hist_snap[hist_snap["split"] != "demo"]          # demo patients are never trained on
    fc = DF.DemandForecaster().fit(hist_snap, visits)
    fnow = DF.forecast_now(fc, am, NOW, dv, dsnap)
    actual = {h: int(((dv["disposition_time"] > NOW) & (dv["disposition_time"] <= NOW + pd.Timedelta(hours=h))).sum()) for h in C.FORECAST_HORIZONS_HOURS}

    # ---- report
    print(f"\n=== Demo day: {DAY.date()} ({DAY.day_name()}), DEMO_NOW = {NOW.strftime('%H:%M')} ===")
    arch_counts = pd.Series(labels).value_counts().to_dict()
    print("patients by archetype:", arch_counts)
    print("admit-score category of the 30 at DEMO_NOW (latest snapshot):", sc["category"].value_counts().to_dict())
    print(f"payers: {dv['payer'].value_counts().to_dict()} | ESI: {dv['esi'].value_counts().sort_index().to_dict()}")
    print(f"boarding at DEMO_NOW (decision made, no bed matched yet): {len(boarders)} patients")
    print(f"beds: {beds['status'].value_counts().to_dict()} | discharge pending {int((beds['discharge_pending'] & (beds['status'] == 'occupied')).sum())} | "
          f"occupied by {beds['current_visit_id'].nunique()} distinct real inpatient visits")
    print("\nScript A, score arc (triage -> orders -> results):")
    print(arcA.assign(time=arcA["snapshot_time"].dt.strftime("%H:%M"), p=arcA["p"].round(2))[["time", "trigger_type", "p", "category"]].to_string(index=False))
    print(f"\nPayer twins {base_new} (commercial) vs {twin_new} (uninsured): scores {pp.round(4).tolist()}")
    print("\nProvisional bed recommendations (hard filters + config weights) for patients waiting on a bed at DEMO_NOW:")
    for vid, top in recs.items():
        r = dv.set_index("visit_id").loc[vid]
        tag = labels[vid]
        print(f"  {vid} [{tag}] needs {r['requested_level_of_care']}{' +isolation' if r['isolation_required'] else ''}"
              f"{' pref ' + r['preferred_unit'] if pd.notna(r['preferred_unit']) else ''}: "
              + ", ".join(f"{x['bed_id']} {x['score']:.2f}" for x in top))
    print(f"\nForecast at DEMO_NOW from the demo census: " + " | ".join(
        f"{h}h {fnow[f'{h}h']['expected_new_admits']:.1f} (90% {fnow[f'{h}h']['low90']:.1f}-{fnow[f'{h}h']['high90']:.1f}; replay shows {actual[h]} decisions from these 30)"
        for h in C.FORECAST_HORIZONS_HOURS))
    print("(the forecast includes expected future arrivals; the replay counts only these 30 patients, so actual is lower by design)")
    print("\nPROBLEMS:", problems if problems else "none")

    # ---- write
    os.makedirs(C.DATA_DIR, exist_ok=True)
    pd.concat([visits, dv], ignore_index=True)[S.cols("visits")].to_csv(f"{C.DATA_DIR}/visits.csv", index=False)
    pd.concat([events, de], ignore_index=True)[S.cols("events")].to_csv(f"{C.DATA_DIR}/events.csv", index=False)
    pd.concat([vitals, dt], ignore_index=True)[S.cols("vitals")].to_csv(f"{C.DATA_DIR}/vitals.csv", index=False)
    occ_ext.to_csv(f"{C.DATA_DIR}/occupancy_hourly.csv", index=False)
    beds.to_csv(f"{C.DATA_DIR}/beds.csv", index=False)
    manifest = {
        "demo_date": str(DAY.date()), "demo_now": str(NOW),
        "patients": [{"visit_id": v, "archetype": labels[v], "stage_at_now": stages[v]} for v in sorted(labels)],
        "scripts": {k: {"visit_id": vid_of[f"scripted_{k}"], "title": SCRIPTS[k]["title"]} for k in SCRIPTS},
        "script_A_score_arc": [{"time": str(r.snapshot_time), "trigger": r.trigger_type, "p": round(float(r.p), 3), "category": r.category}
                               for r in arcA.itertuples()],
        "payer_twins": {"commercial": base_new, "uninsured": twin_new, "scores": [round(float(x), 4) for x in pp]},
        "provisional_recommendations": recs,
        "suggested_override": {"patient": c_id, "ai_top": "CARD-11", "override_to": "CARD-05",
                               "reason_text": "Holding CARD-11 (beside the nurses' station) for an unstable patient arriving from the ward."},
        "timeline_after_now": {v: [{"time": str(r.event_time), "event": r.event_type, "order": None if pd.isna(r.order_name) else r.order_name,
                                    "abnormal": None if pd.isna(r.result_abnormal) else bool(r.result_abnormal)}
                                   for r in de[(de["visit_id"] == v) & (de["event_time"] > NOW)].itertuples()]
                               for v in (a_id, b_id, c_id)},
        "checks_passed": not problems,
    }
    with open(f"{C.DATA_DIR}/demo_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str, allow_nan=False)   # strict JSON: fail loudly instead of writing NaN
    print(f"\nWrote visits/events/vitals/occupancy_hourly (with demo rows), beds.csv and demo_manifest.json to ./{C.DATA_DIR}/")
    return problems


if __name__ == "__main__":
    import build_demo
    build_demo.main()