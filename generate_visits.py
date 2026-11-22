"""
generate_visits.py -- Step 1: build visits, events, vitals and occupancy_hourly.

Causal order per visit (matches config.py section 2):
    age -> complaint -> latent severity -> triage vitals -> ESI -> LWBS -> orders -> results -> admit
then the timeline (events) and, for admits, the five boarding stages.

Every number comes from config.py; every column name comes from schema.py.
Run:  python generate_visits.py          (writes CSVs to config.DATA_DIR and prints a check report)
"""
import os
from datetime import date

import numpy as np
import pandas as pd

import config as C
import schema as S

MIN = 60                                  # seconds per minute
EPOCH = np.datetime64("1970-01-01T00:00:00")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1.0 - p))


def to_dt(seconds):
    """Integer epoch seconds -> datetime64[ns] (NaT stays NaT)."""
    s = pd.Series(seconds)
    return pd.to_datetime(s, unit="s")


def uniform_sec(rng, lo_hi_min, size):
    lo, hi = lo_hi_min
    return np.round(rng.uniform(lo, hi, size) * MIN).astype(np.int64)


# ------------------------------------------------------------------ occupancy
def make_occupancy(rng, start_sec):
    """Hourly inpatient occupancy: mean INPATIENT_OCCUPANCY, afternoon peak, AR(1) drift.
    Covers the window plus one extra day so late-night arrivals still find an hour to join."""
    n_hours = (C.HISTORY_DAYS + 1) * 24
    hour = np.arange(n_hours)
    mean = C.INPATIENT_OCCUPANCY + C.OCCUPANCY_DAILY_AMPLITUDE * np.cos(2 * np.pi * ((hour % 24) - 15) / 24)
    x = np.zeros(n_hours)
    eps = rng.normal(0, C.OCCUPANCY_NOISE_SD, n_hours)
    for t in range(1, n_hours):
        x[t] = C.OCCUPANCY_AR1 * x[t - 1] + eps[t]
    occ = np.clip(mean + x, *C.OCCUPANCY_LIMITS)
    return pd.DataFrame({"hour_start": to_dt(start_sec + hour * 3600), "inpatient_occupancy": occ}), occ


# ------------------------------------------------------------------ vitals
class VitalModel:
    """A patient's vitals = baseline + complaint shift + severity loading + a fixed personal offset
    (so repeat readings of one patient stay correlated) + per-reading noise."""

    def __init__(self, rng, complaint, severity):
        self.rng = rng
        n = len(severity)
        self.mean = {}
        for v, (mu, sd) in C.VITAL_BASELINE.items():
            shift = np.array([C.VITAL_SHIFT_BY_COMPLAINT.get(c, {}).get(v, 0.0) for c in complaint])
            offset = rng.normal(0, sd * C.VITAL_REPEAT_SD_SHARE, n)
            self.mean[v] = mu + shift + C.SEVERITY_LOADING_ON_VITALS[v] * severity + offset

    def read(self, idx):
        """One reading per entry of idx (visit positions). Returns {vital: array}."""
        out = {}
        for v, (_, sd) in C.VITAL_BASELINE.items():
            val = self.mean[v][idx] + self.rng.normal(0, sd * C.VITAL_NOISE_SD_SHARE, len(idx))
            lo, hi = C.VITAL_LIMITS[v]
            out[v] = np.clip(val, lo, hi)
        return out


def count_abnormal(vit):
    n = np.zeros(len(next(iter(vit.values()))), dtype=int)
    for v, (lo, hi) in C.VITAL_ABNORMAL.items():
        if lo is not None:
            n += (vit[v] < lo)
        if hi is not None:
            n += (vit[v] > hi)
    return n


# ------------------------------------------------------------------ main generator
def generate(seed=C.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    start = date.fromisoformat(C.HISTORY_START_DATE)
    start_sec = int((np.datetime64(start.isoformat()) - EPOCH) / np.timedelta64(1, "s"))

    # ---- 1. arrivals: daily counts drawn around config.expected_daily_visits() (total fixed), spread by hour
    exp = C.expected_daily_visits()
    days = sorted(exp)
    # exactly N_HISTORY_VISITS_EXPECTED visits, spread over days in proportion to the expected daily volume
    counts = rng.multinomial(C.N_HISTORY_VISITS_EXPECTED, np.array([exp[d] for d in days]) / sum(exp.values()))
    day_idx = np.repeat(np.arange(len(days)), counts)
    hour = rng.choice(24, size=len(day_idx), p=C.ARRIVAL_HOURLY_WEIGHTS)
    arrival = start_sec + day_idx * 86400 + hour * 3600 + rng.integers(0, 3600, len(day_idx))
    arrival = np.sort(arrival)
    N = len(arrival)
    arr_dt = to_dt(arrival)
    arr_month = arr_dt.dt.month.to_numpy()
    winter = np.isin(arr_month, C.WINTER_MONTHS)

    # ---- 2. occupancy (hospital context)
    occ_df, occ = make_occupancy(rng, start_sec)
    occ_at_arrival = occ[(arrival - start_sec) // 3600]

    # ---- 3. who: age -> payer -> complaint (age tilt, winter tilt) -> latent severity
    age_mix = {a: p for a, p in C.AGE_MIX.items() if C.INCLUDE_PEDIATRIC or a != "0-17"}
    ages = np.array(list(age_mix))
    age = rng.choice(ages, size=N, p=np.array(list(age_mix.values())) / sum(age_mix.values()))
    payers = list(C.PAYER_MIX)
    payer = rng.choice(payers, size=N, p=list(C.PAYER_MIX.values()))
    complaints = list(C.COMPLAINT_MIX)
    complaint = np.empty(N, dtype=object)
    for a in ages:
        for w in (False, True):
            m = (age == a) & (winter == w)
            if not m.any():
                continue
            wts = np.array([C.COMPLAINT_MIX[c] * C.COMPLAINT_AGE_TILT.get(a, {}).get(c, 1.0)
                            * (C.WINTER_COMPLAINT_TILT.get(c, 1.0) if w else 1.0) for c in complaints])
            complaint[m] = rng.choice(complaints, size=int(m.sum()), p=wts / wts.sum())
    severity = rng.normal(0, C.LATENT_SEVERITY_SD, N)

    # ---- 4. triage vitals and ESI
    vm = VitalModel(rng, complaint, severity)
    all_idx = np.arange(N)
    triage_vit = vm.read(all_idx)
    abn_vitals = count_abnormal(triage_vit)
    esi_score = (C.ESI_SCORE_WEIGHTS["severity"] * severity + C.ESI_SCORE_WEIGHTS["abnormal_vitals"] * abn_vitals
                 + C.ESI_SCORE_WEIGHTS["age_75plus"] * (age == "75+")
                 + rng.normal(0, C.ESI_SCORE_WEIGHTS["noise_sd"], N))
    rank = np.empty(N)
    rank[np.argsort(-esi_score)] = (np.arange(N) + 0.5) / N          # 0 = most acute
    esi = 1 + np.searchsorted(np.cumsum(list(C.ESI_MIX.values())), rank, side="right")
    esi = np.clip(esi, 1, 5)

    # ---- 5. waiting: LWBS and door-to-provider
    eligible = np.isin(esi, C.LWBS_ELIGIBLE_ESI)
    p_lwbs = np.array([C.LWBS_BY_PAYER[p] for p in payer]) / eligible.mean()
    lwbs = eligible & (rng.random(N) < p_lwbs)
    med_p = np.array([C.DOOR_TO_PROVIDER_MIN_BY_PAYER[p] for p in payer], dtype=float)
    d2p = med_p * rng.lognormal(0, C.DOOR_TO_PROVIDER_SIGMA, N) * np.array([C.D2P_ESI_FACTOR[e] for e in esi])
    for p in payers:                          # re-centre so each payer's median (patients seen) hits the brief
        m = payer == p
        seen = m & ~lwbs
        if seen.any():
            d2p[m] *= C.DOOR_TO_PROVIDER_MIN_BY_PAYER[p] / np.median(d2p[seen])
    triage_sec = arrival + uniform_sec(rng, C.TRIAGE_DELAY_MIN, N)
    provider_sec = arrival + np.maximum(np.round(d2p * MIN).astype(np.int64), (triage_sec - arrival) + MIN)

    # ---- 6. orders and results (seen patients only)
    names = list(C.ORDER_BASE_PROB)
    K = len(names)
    seen = ~lwbs
    p_order = np.minimum(0.95, np.outer([C.ORDER_ESI_MULTIPLIER[e] for e in esi], list(C.ORDER_BASE_PROB.values())))
    ordered = (rng.random((N, K)) < p_order) & seen[:, None]
    order_sec = provider_sec[:, None] + np.round(rng.uniform(*C.ORDER_DELAY_AFTER_PROVIDER_MIN, (N, K)) * MIN).astype(np.int64)
    lat = np.array([C.RESULT_LATENCY_MIN[n] for n in names], dtype=float)
    result_sec = order_sec + np.round(rng.uniform(lat[:, 0], lat[:, 1], (N, K)) * MIN).astype(np.int64)
    has_result = ordered & ~np.isin(names, C.ORDERS_WITHOUT_RESULT)[None, :]
    p_abn = sigmoid(logit(C.RESULT_ABNORMAL_BASE_PROB) + C.RESULT_ABNORMAL_SEVERITY_SLOPE * severity)
    abnormal = has_result & (rng.random((N, K)) < p_abn[:, None])
    n_orders = ordered.sum(1)
    n_abn_results = abnormal.sum(1)
    last_activity = np.maximum(provider_sec,
                               np.where(ordered, np.where(has_result, result_sec, order_sec), 0).max(1))

    # ---- 7. admit: ESI shape + true effects, intercept solved so the overall rate hits the brief
    lo = C.ADMIT_LOGODDS
    z = (logit([C.ESI_ADMIT_RATE_SHAPE[e] for e in esi])
         + np.array([lo["age_band"][a] for a in age]) + np.array([lo["complaint"][c] for c in complaint])
         + np.array([C.PAYER_ADMIT_LOGODDS[p] for p in payer])
         + lo["per_abnormal_vital"] * abn_vitals + lo["per_order"] * n_orders
         + lo["per_abnormal_result"] * n_abn_results + lo["severity"] * severity
         + lo["inpatient_occupancy"] * (occ_at_arrival - C.INPATIENT_OCCUPANCY) / 0.10     # per +10 occupancy points
         + rng.normal(0, lo["noise_sd"], N))
    lo_b, hi_b = -10.0, 10.0
    for _ in range(80):
        b = (lo_b + hi_b) / 2
        if (sigmoid(z + b) * seen).sum() / N > C.ADMIT_RATE_TARGET:
            hi_b = b
        else:
            lo_b = b
    admit_prob = sigmoid(z + b) * seen
    admitted = rng.random(N) < admit_prob
    disposition = np.where(lwbs, "lwbs", np.where(admitted, "admit", "discharge"))

    # ---- 8. bed-matching bridge fields (admits only)
    adm = np.flatnonzero(admitted)
    level = np.full(N, None, dtype=object)
    iso = pd.array([pd.NA] * N, dtype="boolean")
    pref = np.full(N, None, dtype=object)
    icu_score = severity[adm] + 0.6 * n_abn_results[adm] + 0.4 * (esi[adm] <= 2) + rng.normal(0, 0.5, len(adm))
    r_icu = np.empty(len(adm)); r_icu[np.argsort(-icu_score)] = (np.arange(len(adm)) + 0.5) / len(adm)
    is_icu = r_icu < C.LEVEL_OF_CARE_SHARE["icu"]
    cardiac = np.isin(complaint[adm], ["chest_pain", "shortness_of_breath", "neuro"])
    tele_score = 1.0 * cardiac + 0.5 * n_abn_results[adm] + rng.normal(0, 0.5, len(adm))
    tele_score = np.where(is_icu, -np.inf, tele_score)
    n_tele = int(round(C.LEVEL_OF_CARE_SHARE["telemetry"] * len(adm)))
    is_tele = np.zeros(len(adm), dtype=bool)
    is_tele[np.argsort(-tele_score)[:n_tele]] = True
    level[adm] = np.where(is_icu, "icu", np.where(is_tele, "telemetry", "med_surg"))
    p_iso = np.array([C.ISOLATION_PROB_ADMIT.get(c, C.ISOLATION_PROB_ADMIT["default"]) for c in complaint[adm]])
    iso[adm] = rng.random(len(adm)) < p_iso
    pref[adm] = [C.PREFERRED_UNIT_BY_COMPLAINT.get(c) for c in complaint[adm]]

    # ---- 9. timeline: discharge / LWBS departure, admit decision and the five boarding stages
    decision = np.full(N, -1, dtype=np.int64)
    stage_sec = {s: np.full(N, -1, dtype=np.int64) for s in C.BOARDING_STAGE_ORDER}
    left = np.zeros(N, dtype=np.int64)
    dis = np.flatnonzero(disposition == "discharge")
    left[dis] = last_activity[dis] + uniform_sec(rng, C.DISCHARGE_DELAY_MIN, len(dis))
    lw = np.flatnonzero(lwbs)
    gave_up = np.round(d2p[lw] * rng.uniform(*C.LWBS_LEAVE_FACTOR, len(lw)) * MIN).astype(np.int64)
    left[lw] = arrival[lw] + np.maximum(gave_up, (triage_sec[lw] - arrival[lw]) + MIN)
    decision[adm] = last_activity[adm] + uniform_sec(rng, C.ADMIT_DECISION_DELAY_MIN, len(adm))
    mult = np.array([C.BOARDING_PAYER_MULTIPLIER[p] if C.PLANT_PAYER_BOARDING_GAP else 1.0 for p in payer[adm]])
    total = C.MEDIAN_BOARDING_HOURS * 3600 * rng.lognormal(0, C.BOARDING_SIGMA, len(adm)) * mult
    frac = rng.dirichlet(C.BOARDING_STAGE_ALPHA, len(adm))
    seg = np.maximum(np.round(frac * total[:, None]).astype(np.int64), int(C.BOARDING_MIN_STAGE_MIN * MIN))
    cum = decision[adm][:, None] + np.cumsum(seg, axis=1)
    for j, s in enumerate(C.BOARDING_STAGE_ORDER):
        stage_sec[s][adm] = cum[:, j]
    left[adm] = cum[:, -1]                                           # the 'transported' event

    # ---- 10. reassessments (nurse re-checks vitals every 60-120 min until the patient leaves)
    re_idx, re_sec = [], []
    cur = triage_sec.copy()
    active = np.ones(N, dtype=bool)
    while active.any():
        cur = cur + np.round(rng.uniform(*C.REASSESSMENT_INTERVAL_MIN, N) * MIN).astype(np.int64)
        active &= cur < left
        re_idx.append(np.flatnonzero(active))
        re_sec.append(cur[active])
    re_idx, re_sec = np.concatenate(re_idx), np.concatenate(re_sec)

    # ---- 11. assemble events
    blocks = []

    def add(idx, sec, etype, role, order_name=None, abn=None):
        n = len(idx)
        blocks.append(pd.DataFrame({
            "pos": idx, "t": sec, "event_type": etype, "actor_role": role,
            "order_name": order_name if order_name is not None else None,
            "result_abnormal": abn if abn is not None else pd.array([pd.NA] * n, dtype="boolean")}))

    add(all_idx, arrival, "arrival", None)
    add(all_idx, triage_sec, "triage", "ed_charge_nurse")
    add(all_idx[seen], provider_sec[seen], "provider_seen", "ed_physician")
    oi, ok = np.nonzero(ordered)
    add(oi, order_sec[oi, ok], "order", "ed_physician", order_name=np.array(names)[ok])
    ri, rk = np.nonzero(has_result)
    add(ri, result_sec[ri, rk], "result", None, order_name=np.array(names)[rk],
        abn=pd.array(abnormal[ri, rk], dtype="boolean"))
    add(re_idx, re_sec, "reassessment", "ed_charge_nurse")
    add(adm, decision[adm], "admit_decision", "ed_physician")
    roles = {"bed_requested": "ed_charge_nurse", "bed_matched": "bed_manager", "bed_ready": "bed_manager",
             "report_given": "inpatient_nurse", "transported": "ed_charge_nurse"}
    for s in C.BOARDING_STAGE_ORDER:
        add(adm, stage_sec[s][adm], s, roles[s])
    non_admit = np.flatnonzero(~admitted)
    add(non_admit, left[non_admit], "departed", "ed_charge_nurse")

    ev = pd.concat(blocks, ignore_index=True)
    order_of_type = {t: i for i, t in enumerate(S.EVENT_TYPES)}
    ev["_o"] = ev["event_type"].map(order_of_type)
    ev = ev.sort_values(["pos", "t", "_o"], kind="stable").reset_index(drop=True)

    visit_id = np.array([f"V-{i + 1:06d}" for i in range(N)])
    events = pd.DataFrame({
        "event_id": [f"E-{i + 1:07d}" for i in range(len(ev))],
        "visit_id": visit_id[ev["pos"].to_numpy()],
        "event_time": to_dt(ev["t"].to_numpy()),
        "event_type": ev["event_type"],
        "order_name": ev["order_name"],
        "result_abnormal": ev["result_abnormal"].astype("boolean"),
        "actor_role": ev["actor_role"]})

    # ---- 12. vitals rows: one at triage, one per reassessment
    v_pos = np.concatenate([all_idx, re_idx])
    v_sec = np.concatenate([triage_sec, re_sec])
    later = vm.read(re_idx)
    vit = {v: np.concatenate([triage_vit[v], later[v]]) for v in C.VITAL_BASELINE}
    o = np.lexsort((v_sec, v_pos))
    vitals = pd.DataFrame({"vitals_id": [f"VT-{i + 1:07d}" for i in range(len(o))],
                           "visit_id": visit_id[v_pos[o]], "recorded_time": to_dt(v_sec[o])})
    for v in C.VITAL_BASELINE:
        vitals[v] = np.round(vit[v][o], 1)

    # ---- 13. visits (timestamps taken from the same arrays as events, so the two always agree)
    def tcol(a, mask=None):
        a = np.where(mask, a, np.iinfo(np.int64).min) if mask is not None else a
        out = to_dt(a)
        return out.where(mask, pd.NaT) if mask is not None else out

    am = admitted
    split = rng.choice(list(C.SPLIT), size=N, p=list(C.SPLIT.values()))
    visits = pd.DataFrame({
        "visit_id": visit_id, "arrival_time": arr_dt, "age_band": age, "payer": payer,
        "complaint_category": complaint, "esi": pd.array(esi, dtype="Int64"),
        "requested_level_of_care": level, "isolation_required": iso, "preferred_unit": pref,
        "disposition": disposition, "admitted": pd.array(admitted, dtype="boolean"),
        "lwbs": pd.array(lwbs, dtype="boolean"),
        "door_to_provider_min": np.where(seen, (provider_sec - arrival) / 60.0, np.nan),
        "disposition_time": tcol(decision, am),
        "bed_requested_time": tcol(stage_sec["bed_requested"], am),
        "bed_matched_time": tcol(stage_sec["bed_matched"], am),
        "bed_ready_time": tcol(stage_sec["bed_ready"], am),
        "left_ed_time": to_dt(left),
        "boarding_hours": np.where(am, (left - decision) / 3600.0, np.nan),
        "split": split})
    tables = {"visits": visits, "events": events, "vitals": vitals, "occupancy_hourly": occ_df}
    extras = {"severity": severity, "admit_prob": admit_prob, "intercept": b, "abn_vitals": abn_vitals,
              "n_orders": n_orders, "n_abn_results": n_abn_results, "occ_at_arrival": occ_at_arrival}
    return tables, extras


# ------------------------------------------------------------------ checks
def report(tables, extras):
    v, ev, vt, oc = (tables[k] for k in ("visits", "events", "vitals", "occupancy_hourly"))
    problems = []
    for t, df in tables.items():
        problems += S.validate(df, t)
    problems += S.check_foreign_keys(tables)
    problems += S.check_visit_event_consistency(v, ev)
    # visit-level rules the schema does not cover
    if (v["arrival_time"] > v["left_ed_time"]).any():
        problems.append("a visit leaves before it arrives")
    adm = v[v["admitted"].eq(True)]
    stage_cols = ["disposition_time", "bed_requested_time", "bed_matched_time", "bed_ready_time", "left_ed_time"]
    if (adm[stage_cols].diff(axis=1).iloc[:, 1:] < pd.Timedelta(0)).any().any():
        problems.append("boarding stages out of order")
    first_vit = vt.merge(ev[ev.event_type == "triage"][["visit_id", "event_time"]], on="visit_id")
    if (vt.groupby("visit_id")["recorded_time"].min().reindex(v["visit_id"]).to_numpy()
            != ev[ev.event_type == "triage"].set_index("visit_id")["event_time"].reindex(v["visit_id"]).to_numpy()).any():
        problems.append("first vitals reading is not at the triage time")

    N = len(v)
    print(f"\n=== Step 1 report ({N:,} visits) ===")
    print(f"rows: events {len(ev):,} | vitals {len(vt):,} | occupancy_hourly {len(oc):,}")
    print(f"admit rate {v['admitted'].mean():.3%}  (target {C.ADMIT_RATE_TARGET:.1%}) | LWBS {v['lwbs'].mean():.3%} (target {C.LWBS_RATE_OVERALL:.1%})")
    seen = v[~v["lwbs"].astype(bool)]
    print(f"door-to-provider median, all seen: {seen['door_to_provider_min'].median():.1f} min (brief ~{C.DOOR_TO_PROVIDER_MIN_OVERALL})")
    print(f"boarding median: {adm['boarding_hours'].median():.2f} h (brief {C.MEDIAN_BOARDING_HOURS})")
    print("\npayer     share  LWBS(brief)   d2p median(brief)")
    for p in C.PAYER_MIX:
        g = v[v.payer == p]
        print(f"{p:14s}{len(g) / N:6.3f}  {g['lwbs'].mean():.3f} ({C.LWBS_BY_PAYER[p]:.3f})   "
              f"{g[~g['lwbs'].astype(bool)]['door_to_provider_min'].median():5.1f} ({C.DOOR_TO_PROVIDER_MIN_BY_PAYER[p]})")
    print("\nESI mix  :", {int(k): round(x, 3) for k, x in v["esi"].value_counts(normalize=True).sort_index().items()})
    print("admit/ESI:", {int(k): round(x, 3) for k, x in v.groupby("esi")["admitted"].mean().items()})
    print("level of care:", adm["requested_level_of_care"].value_counts(normalize=True).round(3).to_dict())
    h = v["arrival_time"].dt.hour.value_counts(normalize=True).sort_index()
    print(f"arrival peaks: 10:00 {h[10]:.3f}, 18:00 {h[18]:.3f}, dip 14:00 {h[14]:.3f}, 04:00 {h[4]:.3f}")
    month = v["arrival_time"].dt.month
    wv = v[month.isin(C.WINTER_MONTHS)]
    per_day = lambda d: len(d) / d["arrival_time"].dt.date.nunique()
    print(f"winter vs other visits/day: {per_day(wv):.0f} vs {per_day(v[~month.isin(C.WINTER_MONTHS)]):.0f}")
    dow = v["arrival_time"].dt.dayofweek.value_counts().sort_index()
    print("visits by weekday Mon..Sun:", dow.tolist())
    print(f"occupancy mean {oc['inpatient_occupancy'].mean():.3f} (target {C.INPATIENT_OCCUPANCY}), "
          f"min {oc['inpatient_occupancy'].min():.2f}, max {oc['inpatient_occupancy'].max():.2f}")
    print("share of admits with a stalled stage (limits from config, measured on events):")
    st = ev[ev.event_type.isin(["admit_decision"] + C.BOARDING_STAGE_ORDER)].pivot(index="visit_id", columns="event_type", values="event_time")
    seq = ["admit_decision"] + C.BOARDING_STAGE_ORDER
    for j, nme in enumerate(C.STALL_LIMIT_MIN):
        mins = (st[seq[j + 1]] - st[seq[j]]).dt.total_seconds() / 60
        print(f"   {nme:36s} {(mins > C.STALL_LIMIT_MIN[nme]).mean():6.2%} over {C.STALL_LIMIT_MIN[nme]} min (median {mins.median():.0f})")
    print(f"uninsured: {int((v.payer == 'uninsured').sum()):,} visits, {int(((v.payer == 'uninsured') & v.admitted.astype(bool)).sum()):,} admits "
          f"(test split: {int(((v.payer == 'uninsured') & (v.split == 'test')).sum()):,} visits)")

    # AUC sanity check on end-of-visit features (the real model is Step 3 and uses 'so far' snapshots)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        X = pd.get_dummies(v[["age_band", "complaint_category", "esi"]].astype({"esi": str}), drop_first=True).astype(float)
        X["abn_vitals"], X["n_orders"], X["n_abn_results"] = extras["abn_vitals"], extras["n_orders"], extras["n_abn_results"]
        X["occ"] = extras["occ_at_arrival"]
        y = v["admitted"].astype(int)
        tr, te = (v.split == "train").to_numpy(), (v.split == "test").to_numpy()
        m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        auc = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
        print(f"\nAUC sanity (logistic on full-visit features, test split): {auc:.3f}  expected {C.EXPECTED_AUC_RANGE}")
        coef = dict(zip(X.columns, m.coef_[0]))
        print(f"learned occupancy coef {coef['occ']:+.2f} (true effect 0)  |  per_abnormal_result {coef['n_abn_results']:+.2f} (true {C.ADMIT_LOGODDS['per_abnormal_result']})")
    except Exception as e:                                          # sklearn missing is not fatal
        print("AUC check skipped:", e)

    print("\nSCHEMA / CONSISTENCY PROBLEMS:", problems if problems else "none")
    return problems


def save(tables):
    os.makedirs(C.DATA_DIR, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(os.path.join(C.DATA_DIR, f"{name}.csv"), index=False)


if __name__ == "__main__":
    tables, extras = generate()
    problems = report(tables, extras)
    save(tables)
    print(f"\nWrote {len(tables)} CSVs to ./{C.DATA_DIR}/")