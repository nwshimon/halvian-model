"""
early_prep.py -- the early-prep loop: act on the admit forecast BEFORE the doctor decides, so the bed is closer to ready when the decision arrives.

What it does at a moment `t` (all numbers from config.py section 12; priors from data/priors.json):
  1. Lever 1, tiered thresholds.  p >= 0.40 soft (reversible: soft hold on a ready or cleaning bed, draft bed request, EVS standby),
                                  p >= 0.60 expedite (send the bed request, start task chains on a discharge-pending bed),
                                  p >= 0.85 urgent (same, urgent priority).  A hold is released only when p falls below 0.28 (hysteresis).
  2. Lever 2, automatic prompts.  Tasks are created already assigned to a role with a due time; a human confirms in one tap. The system never
                                  discharges anyone: sign-off stays with the inpatient physician.
  3. Lever 3, forecaster feeds the loop.  The 4-hour demand forecast is compared with the beds that can be ready in 4 hours; a shortfall starts a
                                  POOLED hold (a hold on a bed for a level of care, not for a named patient). The next patient who crosses a
                                  threshold claims it.
  Claim order: decided patients first (by decision time), then likely patients by p. A soft hold yields to anyone ahead.
  Savings come from STARTING existing work earlier. Nobody is assumed to work faster (durations are the published ones in config).
Payer and every other prohibited field are never read here (asserted in tests).
"""
import json

import numpy as np
import pandas as pd

import bed_rules as BR
import config as C

LEVELS = ("med_surg", "telemetry", "icu")
STEPUP_TOLERANCE_MIN = 30          # a higher-level bed is used for a decided patient only if the exact-level bed is this much later
POOL_MAX_CHAINS_PER_LEVEL = 2


# ------------------------------------------------------------------ priors
class Priors:
    def __init__(self, path=None):
        d = json.load(open(path or f"{C.DATA_DIR}/priors.json"))
        self.level = {(r["complaint_category"], r["esi"]): r for r in d["level_priors"]}
        self.iso = {(r["complaint_category"], r["esi"]): r["share"] for r in d["isolation_priors"]}
        self.dt = {r["pending_results"]: r for r in d["decision_time_priors"]}
        self.overall = d["overall_level_share"]

    def level_guess(self, complaint, esi):
        """(most likely level, its share, full distribution) from complaint and ESI only."""
        r = self.level.get((complaint, int(esi)))
        dist = {lv: (r[lv] if r else self.overall.get(lv, 0.0)) for lv in LEVELS}
        best = max(dist, key=dist.get)
        return best, dist[best], dist

    def isolation_guess(self, complaint, esi):
        return self.iso.get((complaint, int(esi)), 0.0) >= 0.5

    def minutes_to_decision(self, pending, which=None):
        r = self.dt[min(int(pending), 3)]
        return r[which or C.PREP_NEEDED_BY_PERCENTILE]


def tier_name(p):
    if p >= C.PREP_URGENT_THRESHOLD:
        return "urgent"
    if p >= C.PREP_EXPEDITE_THRESHOLD:
        return "expedite"
    if p >= C.PREP_SOFT_THRESHOLD:
        return "soft"
    return None


# ------------------------------------------------------------------ the board at time t
def board_at(beds, t):
    """The demo board as of t: a cleaning bed whose expected ready time has passed is ready. Nothing else is assumed to change."""
    b = beds.copy()
    done = (b["status"] == "cleaning") & b["expected_ready_time"].notna() & (b["expected_ready_time"] <= t)
    b.loc[done, "status"] = "ready"
    return b


# ------------------------------------------------------------------ task chains
def schedule_chain(blocker_types, start, expedited_after=None):
    """Forward-schedule the chain for one discharge-pending bed.
    blockers run in parallel from `start`; then sign-off -> patient leaves -> bed cleaning.
    A task uses the expedited duration only if it starts at or after `expedited_after` (a human prioritised it).
    Returns (tasks, ready_time). Each task: type, owner_role, start, due, minutes, depends_on (index or None)."""
    tasks = []

    def dur(tt, s):
        norm, exp = C.PREP_TASK_MINUTES[tt]
        return exp if (expedited_after is not None and s >= expedited_after) else norm

    fin = []
    for bt in blocker_types:
        m = dur(bt, start)
        tasks.append({"task_type": bt, "owner_role": C.BLOCKER_TYPES[bt], "start": start, "minutes": m, "due": start + pd.Timedelta(minutes=m),
                      "depends_on": None})
        fin.append(start + pd.Timedelta(minutes=m))
    cur_end = max(fin) if fin else start
    blockers_idx = list(range(len(tasks)))
    prev = None
    for k, tt in enumerate(C.PREP_TAIL_TASKS):
        m = dur(tt, cur_end)
        dep = blockers_idx if k == 0 else [prev]
        tasks.append({"task_type": tt, "owner_role": C.PREP_TASK_OWNERS[tt], "start": cur_end, "minutes": m,
                      "due": cur_end + pd.Timedelta(minutes=m), "depends_on": dep})
        cur_end = cur_end + pd.Timedelta(minutes=m)
        prev = len(tasks) - 1
    return tasks, cur_end


def chain_feasible(blocker_types, needed_by, now):
    """Placement (days) is excluded unless the bed is needed far enough ahead."""
    if "placement" in blocker_types:
        return (needed_by - now) >= pd.Timedelta(hours=C.PREP_PLACEMENT_MAX_HOURS)
    return True


# ------------------------------------------------------------------ claimants
def claimants_at(t, J, pri, keep_matched=False):
    """Demo patients that want a bed at time t, from the journey data (scores are the real model's).
    decided: the admit decision is made and no bed is matched yet. likely: no decision yet, score >= soft threshold.
    keep_matched: the day replay assigns beds itself, so a patient keeps claiming until the replay assigns them (the data's own match time is ignored)."""
    v, ev, sn = J["visits"], J["events"], J["snap"]
    out = []
    for vid, r in v.iterrows():
        if not (r["arrival_time"] <= t < r["left_ed_time"]):
            continue
        adm = pd.notna(r["disposition_time"]) and r["disposition"] == "admit"
        if adm and r["disposition_time"] <= t:
            if not keep_matched and pd.notna(r["bed_matched_time"]) and pd.Timestamp(r["bed_matched_time"]) <= t:
                continue                                              # already matched by the normal flow
            out.append({"visit_id": vid, "kind": "decided", "p": 1.0, "level": r["requested_level_of_care"],
                        "level_confidence": 1.0, "isolation": bool(r["isolation_required"]),
                        "preferred_unit": r["preferred_unit"] if pd.notna(r["preferred_unit"]) else None,
                        "decision_time": r["disposition_time"], "needed_by": t, "pending": 0, "complaint": r["complaint_category"]})
            continue
        s = sn[(sn["visit_id"] == vid) & (sn["snapshot_time"] <= t)]
        if s.empty:
            continue
        last = s.iloc[-1]
        lv, conf, _ = pri.level_guess(r["complaint_category"], last["esi"])
        out.append({"visit_id": vid, "kind": "ed", "p": float(last["p"]), "level": lv, "level_confidence": float(conf),
                    "isolation": pri.isolation_guess(r["complaint_category"], last["esi"]),
                    "preferred_unit": C.PREFERRED_UNIT_BY_COMPLAINT.get(r["complaint_category"]),
                    "decision_time": None, "needed_by": t + pd.Timedelta(minutes=pri.minutes_to_decision(last["pending_results_so_far"])),
                    "pending": int(last["pending_results_so_far"]), "complaint": r["complaint_category"], "score_time": last["snapshot_time"]})
    return out


# ------------------------------------------------------------------ candidate beds for one claimant
def _candidates(cl, beds, taken, blockers, t, pooled_chain, allow_tier3, expedited):
    rows = []
    pat = {"requested_level_of_care": cl["level"], "isolation_required": cl["isolation"], "preferred_unit": cl["preferred_unit"]}
    for _, bed in beds.iterrows():
        bid = bed["bed_id"]
        if bid in taken or not BR.passes_hard_filters(pat, bed):
            continue
        tier, tasks, ready = None, [], None
        if bed["status"] == "ready":
            tier, ready = 1, t
        elif bed["status"] == "cleaning":
            tier, ready = 2, max(t, bed["expected_ready_time"])
        else:                                                           # occupied, discharge pending
            if not allow_tier3:
                continue
            if bid in pooled_chain:                                     # a chain a pooled hold already started
                h = pooled_chain[bid]
                tasks, ready = schedule_chain(h["blocker_types"], h["chain_start"], expedited_after=t if expedited else None)
                tier = 3
            else:
                bt = [f["blocker_type"] for f in blockers.get(bid, {}).get("findings", [])]
                if not chain_feasible(bt, cl["needed_by"], t):
                    continue
                tasks, ready = schedule_chain(bt, t, expedited_after=t if expedited else None)
                tier = 3
        mins = (ready - t).total_seconds() / 60
        late = max(0.0, (ready - cl["needed_by"]).total_seconds() / 60)
        score = (C.MATCH_WEIGHTS["exact_level_of_care"] * (bed["bed_type"] == cl["level"])
                 + C.MATCH_WEIGHTS["preferred_unit"] * bool(cl["preferred_unit"] and bed["unit"] == cl["preferred_unit"])
                 + C.MATCH_WEIGHTS["isolation_bed_wasted"] * (bool(bed["isolation_capable"]) and not cl["isolation"])
                 + C.MATCH_WEIGHTS["minutes_until_ready"] * mins + C.PREP_LATENESS_PENALTY * late)
        rows.append({"bed_id": bid, "tier": tier, "ready": ready, "tasks": tasks, "score": score, "exact": bed["bed_type"] == cl["level"],
                     "unit": bed["unit"], "bed_type": bed["bed_type"], "late_min": late})
    return rows


def _choose(cl, cands):
    """Decided patients: earliest ready, exact level preferred unless a step-up bed is much sooner.
       Likely patients: the lowest tier that is ready by needed-by (no pointless expediting); if none is, the earliest ready (flagged critical)."""
    if not cands:
        return None
    exact = [c for c in cands if c["exact"]]
    pool = cands
    if cl["kind"] == "ed":                                  # a patient who might be admitted never takes a higher-level bed than they would need
        pool = exact
    elif exact:
        best_exact = min(c["ready"] for c in exact)
        best_any = min(c["ready"] for c in cands)
        pool = exact if (best_exact - best_any).total_seconds() / 60 <= STEPUP_TOLERANCE_MIN else cands
    if not pool:
        return None
    if cl["kind"] == "decided":
        return sorted(pool, key=lambda c: (c["ready"], -c["score"], c["bed_id"]))[0]
    on_time = [c for c in pool if c["late_min"] == 0]
    if on_time:
        low = min(c["tier"] for c in on_time)
        return sorted([c for c in on_time if c["tier"] == low], key=lambda c: (-c["score"], c["bed_id"]))[0]
    return sorted(pool, key=lambda c: (c["ready"], -c["score"], c["bed_id"]))[0]


# ------------------------------------------------------------------ the plan at time t
def plan(t, beds_now, claimants, blockers, pri, prior_holds=None, forecast=None, log=None):
    """One planning pass. prior_holds (from the previous pass) make soft holds sticky and give hysteresis.
    Returns {holds, tasks, bed_requests, pool, banner, events}. Nothing is written anywhere."""
    t = pd.Timestamp(t)
    prior = {h["visit_id"]: h for h in (prior_holds or []) if h["visit_id"] and h["status"] == "active"}
    pooled_prior = [h for h in (prior_holds or []) if h["visit_id"] is None and h["status"] == "active"]
    pooled_chain = {h["bed_id"]: h for h in pooled_prior}
    events, holds, tasks, requests = [], [], [], []
    taken = set()

    # claim order: decided first (decision time), then likely patients by p. A patient already holding with p >= release stays in the queue.
    live = []
    for cl in claimants:
        had = cl["visit_id"] in prior
        if cl["kind"] == "decided" or cl["p"] >= C.PREP_SOFT_THRESHOLD or (had and cl["p"] >= C.PREP_RELEASE_BELOW):
            live.append(cl)
    live_ids = {c["visit_id"] for c in live}
    for vid, h in prior.items():
        if vid not in live_ids:                                      # score fell below the release line, or the patient left
            events.append({"time": t, "action": "hold_released", "visit_id": vid, "bed_id": h["bed_id"],
                           "detail": "p_dropped" if any(c["visit_id"] == vid for c in claimants) else "left_ed"})
    live.sort(key=lambda c: (0, c["decision_time"]) if c["kind"] == "decided" else (1, -c["p"]))

    for cl in live:
        tn = "decided" if cl["kind"] == "decided" else tier_name(cl["p"])
        allow3 = tn in ("decided", "expedite", "urgent") or (cl["visit_id"] in prior and prior[cl["visit_id"]]["tier"] == 3)
        # tier name for a patient in the hysteresis band (0.28-0.40) stays soft
        tn = tn or "soft"
        expedited = tn in ("decided", "expedite", "urgent")
        cands = _candidates(cl, beds_now, taken, blockers, t, pooled_chain, allow3, expedited)
        old = prior.get(cl["visit_id"])
        if old and old["level_assumed"] != cl["level"]:
            events.append({"time": t, "action": "hold_released", "visit_id": cl["visit_id"], "bed_id": old["bed_id"], "detail": "wrong_level"})
            old = None
        pick = None
        if old:                                                     # sticky: keep the same bed while it is still free and still fits
            pick = next((c for c in cands if c["bed_id"] == old["bed_id"]), None)
        if pick is None:
            pick = _choose(cl, cands)
        if pick is None:
            events.append({"time": t, "action": "no_feasible_bed", "visit_id": cl["visit_id"], "bed_id": None,
                           "detail": f"{cl['level']}{' +isolation' if cl['isolation'] else ''}, needed by {cl['needed_by']:%H:%M}"})
            continue
        taken.add(pick["bed_id"])
        critical = bool(pick["late_min"] > 0 and cl["kind"] != "decided")
        h = {"hold_id": old["hold_id"] if old else f"H-{cl['visit_id']}-{pick['bed_id']}", "visit_id": cl["visit_id"], "bed_id": pick["bed_id"],
             "tier": pick["tier"], "kind": ("decided" if cl["kind"] == "decided" else ("soft" if tn == "soft" else "hard")),
             "action_tier": tn, "level_assumed": cl["level"], "level_confidence": cl["level_confidence"],
             "p_at_hold": old["p_at_hold"] if old else cl["p"], "p_latest": cl["p"], "needed_by_time": cl["needed_by"],
             "realistic_ready_time": pick["ready"], "critical": critical, "status": "active", "release_reason": None,
             "created_time": old["created_time"] if old else t, "updated_time": t, "isolation_assumed": cl["isolation"],
             "pooled_from": pooled_chain[pick["bed_id"]]["hold_id"] if pick["bed_id"] in pooled_chain else None}
        holds.append(h)
        if not old:
            events.append({"time": t, "action": "hold_created", "visit_id": cl["visit_id"], "bed_id": pick["bed_id"],
                           "detail": f"{tn}, tier {pick['tier']}, ready {pick['ready']:%H:%M}, needed by {cl['needed_by']:%H:%M}"})
        elif old["kind"] == "soft" and h["kind"] != "soft":
            events.append({"time": t, "action": "hold_upgraded", "visit_id": cl["visit_id"], "bed_id": pick["bed_id"], "detail": "soft -> hard"})
        if pick["tasks"]:
            blockers_n = len([x for x in pick["tasks"] if x["task_type"] not in C.PREP_TAIL_TASKS])
            for i, k in enumerate(pick["tasks"]):
                tasks.append({"task_key": f"{h['hold_id']}-{i}", "hold_id": h["hold_id"], "bed_id": pick["bed_id"], "claimant": cl["visit_id"],
                              "occupant_visit_id": (blockers.get(pick["bed_id"], {}).get("visit_id")), "task_type": k["task_type"],
                              "owner_role": k["owner_role"], "start_by_time": k["start"], "due_time": k["due"], "duration_min": k["minutes"],
                              "depends_on_key": None if k["depends_on"] is None else [f"{h['hold_id']}-{j}" for j in k["depends_on"]],
                              "priority": "urgent" if tn == "urgent" else "standard", "critical": critical, "created_time": h["created_time"]})
        if tn in ("expedite", "urgent", "decided") and cl["kind"] == "ed":
            requests.append({"visit_id": cl["visit_id"], "level": cl["level"], "isolation": cl["isolation"], "preferred_unit": cl["preferred_unit"],
                             "bed_id": pick["bed_id"], "sent_early": True})
        elif tn == "soft":
            requests.append({"visit_id": cl["visit_id"], "level": cl["level"], "isolation": cl["isolation"], "preferred_unit": cl["preferred_unit"],
                             "bed_id": pick["bed_id"], "sent_early": False})

    # ---- lever 3: demand forecast against supply, pooled holds for the shortfall
    banner, pool_new = _pool(t, beds_now, holds, taken, blockers, pri, forecast, pooled_chain, events)
    keep_pool = [h for h in pooled_prior if h["bed_id"] not in taken]          # pooled holds nobody has claimed yet
    for h in keep_pool:
        h = dict(h); h["updated_time"] = t; holds.append(h)
        taken.add(h["bed_id"])
        tasks.extend(_pool_tasks(h, t, blockers))
    for h in pool_new:
        holds.append(h); tasks.extend(_pool_tasks(h, t, blockers))
    return {"time": t, "holds": holds, "tasks": tasks, "bed_requests": requests, "banner": banner, "events": events,
            "pool": [h for h in holds if h["visit_id"] is None]}


def _pool_tasks(h, t, blockers):
    chain, _ = schedule_chain(h["blocker_types"], h["chain_start"])
    return [{"task_key": f"{h['hold_id']}-{i}", "hold_id": h["hold_id"], "bed_id": h["bed_id"], "claimant": None,
             "occupant_visit_id": blockers.get(h["bed_id"], {}).get("visit_id"), "task_type": k["task_type"], "owner_role": k["owner_role"],
             "start_by_time": k["start"], "due_time": k["due"], "duration_min": k["minutes"],
             "depends_on_key": None if k["depends_on"] is None else [f"{h['hold_id']}-{j}" for j in k["depends_on"]],
             "priority": "standard", "critical": False, "created_time": h["chain_start"]} for i, k in enumerate(chain)]


def _pool(t, beds_now, holds, taken, blockers, pri, forecast, pooled_chain, events):
    """Expected admits in the next H hours (the demand forecaster) against beds that can be ready within H, by level. Planning shares by level are the fixed config shares."""
    H = pd.Timedelta(hours=C.PREP_POOL_HORIZON_HOURS)
    fh = (forecast or {}).get(f"{C.PREP_POOL_HORIZON_HOURS}h")
    banner = {"horizon_hours": C.PREP_POOL_HORIZON_HOURS, "forecast_used": fh is not None, "forecast_admits": fh, "levels": {}, "shortfall_beds": 0.0}
    if fh is None:
        return banner, []
    pool_new = []
    for lv in LEVELS:
        share = C.LEVEL_OF_CARE_SHARE[lv]
        named = sum(h["p_latest"] for h in holds if h["visit_id"] and h["kind"] != "decided" and h["level_assumed"] == lv)
        demand = max(0.0, fh * share - named)
        supply, cand_dp = 0.0, []
        for _, bed in beds_now.iterrows():
            if bed["bed_id"] in taken or bed["bed_type"] != lv:
                continue
            if bed["status"] == "ready":
                supply += 1
            elif bed["status"] == "cleaning" and bed["expected_ready_time"] <= t + H:
                supply += 1
            elif bed["status"] == "occupied" and bool(bed["discharge_pending"]):
                bt = [f["blocker_type"] for f in blockers.get(bed["bed_id"], {}).get("findings", [])]
                if "placement" in bt:
                    continue
                ch, ready = schedule_chain(bt, t)
                cand_dp.append((ready, bed["bed_id"], bt))
        shortfall = demand - supply
        banner["levels"][lv] = {"expected_demand": round(demand, 2), "beds_ready_within_horizon": int(supply), "shortfall": round(shortfall, 2)}
        if shortfall >= C.PREP_POOL_MIN_SHORTFALL:
            banner["shortfall_beds"] += shortfall
            have = sum(1 for h in pooled_chain.values() if h["level_assumed"] == lv)
            for ready, bid, bt in sorted(cand_dp)[:max(0, min(POOL_MAX_CHAINS_PER_LEVEL - have, int(np.ceil(shortfall))))]:
                if ready > t + H:
                    continue
                hid = f"POOL-{lv}-{bid}"
                pool_new.append({"hold_id": hid, "visit_id": None, "bed_id": bid, "tier": 3, "kind": "pooled", "action_tier": "pooled",
                                 "level_assumed": lv, "level_confidence": share, "p_at_hold": None, "p_latest": None,
                                 "needed_by_time": t + H, "realistic_ready_time": ready, "critical": False, "status": "active",
                                 "release_reason": None, "created_time": t, "updated_time": t, "isolation_assumed": False, "pooled_from": None,
                                 "blocker_types": bt, "chain_start": t})
                taken.add(bid)
                events.append({"time": t, "action": "pooled_hold_created", "visit_id": None, "bed_id": bid,
                               "detail": f"{lv}: expected demand {demand:.1f} vs {int(supply)} beds ready in {C.PREP_POOL_HORIZON_HOURS} h; chain ready {ready:%H:%M}"})
    return banner, pool_new


# ------------------------------------------------------------------ task state, replay and the one-tap "mark done" cascade
def task_state(tasks, t, done_keys=None):
    """waiting (a dependency is not done) / open (can start) / in_progress (started) / done. In a replay a task is done when its scheduled
    due time has passed, unless a person marked it done earlier (done_keys)."""
    done_keys = set(done_keys or [])
    by = {k["task_key"]: k for k in tasks}
    out = {}
    for k in tasks:
        done = k["task_key"] in done_keys or k["due_time"] <= t
        deps = k["depends_on_key"] or []
        deps_done = all((d in done_keys) or by[d]["due_time"] <= t for d in deps)
        if done:
            st = "done"
        elif not deps_done:
            st = "waiting"
        elif k["start_by_time"] <= t:
            st = "in_progress"
        else:
            st = "open"
        out[k["task_key"]] = st
    return out


def mark_done(tasks, task_key, now, done_keys=None):
    """A person marks a task done at `now`. The tasks after it are re-timed from `now` (earlier or later than planned), and the new bed ready
    time is returned. This is the cascade: sign-off, patient leaves and cleaning all move."""
    by = {k["task_key"]: dict(k) for k in tasks}
    now = pd.Timestamp(now)
    by[task_key]["due_time"] = now
    changed = True
    while changed:
        changed = False
        for k in by.values():
            deps = k["depends_on_key"] or []
            if not deps:
                continue
            start = max(by[d]["due_time"] for d in deps)
            new_due = start + pd.Timedelta(minutes=k["duration_min"])
            if start != k["start_by_time"] or new_due != k["due_time"]:
                k["start_by_time"], k["due_time"] = start, new_due
                changed = True
    out = list(by.values())
    ready = max(k["due_time"] for k in out if k["task_type"] == "bed_cleaning") if any(k["task_type"] == "bed_cleaning" for k in out) else None
    return out, ready


# ------------------------------------------------------------------ replay over several moments
def replay(times, J, beds, blockers, pri, forecast_at_now=None, now=None):
    """Runs plan() at each time in order, carrying holds forward (so soft holds stick and hold ages are real). The demand forecast is only
    available at the live 'now', so pooled holds are created there and carried (never newly created) at later replay times."""
    holds, out = None, []
    for t in times:
        t = pd.Timestamp(t)
        cl = claimants_at(t, J, pri)
        fc = forecast_at_now if (now is not None and t == pd.Timestamp(now)) else None
        res = plan(t, board_at(beds, t), cl, blockers, pri, prior_holds=holds, forecast=fc)
        res["claimants"] = cl
        holds = res["holds"]
        out.append(res)
    return out


# ------------------------------------------------------------------ replay estimate (a scenario for ONE patient, never a measurement)
def replay_estimate(v, hold):
    """For a demo patient whose whole timeline is recorded: boarding with the loop vs the recorded boarding.
    The loop sends the bed request when the hard hold is created (the recorded request-to-match gap still applies) and the bed is ready no earlier
    than the hold's realistic ready time. The patient cannot move before the doctor's decision; the recorded report and transport gaps then apply.
    Assumes the held bed really is free. Also returns the bed-minutes it sat held before the decision, which is the cost of acting early."""
    ts = pd.Timestamp
    dec, rec_ready, left = ts(v["disposition_time"]), ts(v["bed_ready_time"]), ts(v["left_ed_time"])
    match_gap = ts(v["bed_matched_time"]) - ts(v["bed_requested_time"])
    ready_loop = max(hold["realistic_ready_time"], hold["created_time"] + match_gap)
    moves = max(ready_loop, dec)
    left_loop = moves + (left - rec_ready)
    return {"recorded_ready": rec_ready, "bed_ready_with_loop": ready_loop, "patient_moves": moves,
            "recorded_boarding_h": (left - dec).total_seconds() / 3600, "boarding_with_loop_h": (left_loop - dec).total_seconds() / 3600,
            "saved_min": round(((left - dec) - (left_loop - dec)).total_seconds() / 60),
            "bed_idle_min_before_decision": round(max(0.0, (dec - ready_loop).total_seconds() / 60))}


# ------------------------------------------------------------------ the whole demo day (added 2026-11-23)
def snapshot_time():
    return pd.Timestamp(C.DEMO_DATE) + pd.Timedelta(hours=int(C.DEMO_NOW_TIME[:2]), minutes=int(C.DEMO_NOW_TIME[3:]))


def day_board(beds, t):
    """The bed table at any time t of the demo day. The data holds ONE snapshot (the demo clock). From it:
      after the snapshot: a bed whose expected ready time has passed is ready (cleaning beds and discharge-pending beds alike). No new discharges are invented.
      before it: run the same facts backwards. A bed that became ready after t was being cleaned (or, if that was more than an hour later, was occupied with
      a discharge pending). A bed whose cleaning started after t was still occupied with a discharge pending. Occupied-no-discharge and blocked beds stay as they are."""
    t, snap = pd.Timestamp(t), snapshot_time()
    b = beds.copy()
    pend = (b["status"] == "occupied") & b["discharge_pending"].astype(bool)
    if t >= snap:
        done = (b["status"].isin(["cleaning"]) | pend) & b["expected_ready_time"].notna() & (b["expected_ready_time"] <= t)
        b.loc[done, "status"] = "ready"
        b.loc[done, "discharge_pending"] = False
        b.loc[done, "current_visit_id"] = np.nan
        return b
    su = b["status_updated_time"]
    late_ready = (b["status"] == "ready") & su.notna() & (su > t)
    far = late_ready & ((su - t) > pd.Timedelta(minutes=60))
    near = late_ready & ~far
    still_dirty = (b["status"] == "cleaning") & su.notna() & (su > t)
    for m, new in ((far, "pending"), (near, "cleaning"), (still_dirty, "pending")):
        if not m.any():
            continue
        if new == "cleaning":
            b.loc[m, "status"] = "cleaning"
            b.loc[m, "discharge_pending"] = False
        else:
            b.loc[m, "status"] = "occupied"
            b.loc[m, "discharge_pending"] = True
        b.loc[m & b["expected_ready_time"].isna(), "expected_ready_time"] = su[m & b["expected_ready_time"].isna()]
    return b


def sim_claimants(t, sims, pri, assigned_ids=frozenset()):
    """Patients a user added in the simulation, as claimants at time t. Each has arrival-based score steps and a decision time."""
    t = pd.Timestamp(t)
    out = []
    for s in sims:
        if s["patient_id"] in assigned_ids:
            continue
        steps = [x for x in s["steps"] if pd.Timestamp(x["time"]) <= t]
        if not steps:
            continue
        lv_pref = C.PREFERRED_UNIT_BY_COMPLAINT.get(s["complaint"])
        if s.get("decision_time") is not None and t >= pd.Timestamp(s["decision_time"]):
            out.append({"visit_id": s["patient_id"], "kind": "decided", "p": 1.0, "level": s["level"], "level_confidence": 1.0, "isolation": bool(s["isolation"]),
                        "preferred_unit": lv_pref, "decision_time": pd.Timestamp(s["decision_time"]), "needed_by": t, "pending": 0, "complaint": s["complaint"]})
            continue
        last = steps[-1]
        lv, conf, _ = pri.level_guess(s["complaint"], s["esi"])
        out.append({"visit_id": s["patient_id"], "kind": "ed", "p": float(last["p"]), "level": lv, "level_confidence": float(conf),
                    "isolation": pri.isolation_guess(s["complaint"], s["esi"]), "preferred_unit": lv_pref, "decision_time": None,
                    "needed_by": t + pd.Timedelta(minutes=pri.minutes_to_decision(last["pending"])), "pending": int(last["pending"]),
                    "complaint": s["complaint"], "score_time": pd.Timestamp(last["time"])})
    return out


def board_with_assigns(beds, t, assigns):
    """day_board plus every bed the user assigned at or before t, shown as occupied by that patient."""
    b = day_board(beds, t)
    gone = set()
    for bed, a in (assigns or {}).items():
        if pd.Timestamp(a["assign_time"]) <= pd.Timestamp(t):
            m = b["bed_id"] == bed
            b.loc[m, "status"] = "occupied"
            b.loc[m, "discharge_pending"] = False
            b.loc[m, "current_visit_id"] = a["patient_id"]
            gone.add(a["patient_id"])
    return b, gone


def replay_day(times, J, beds, blockers, pri, forecast_at_now=None, now=None, sims=(), assigns=None, exclude=()):
    """plan() at every time of the day, carrying holds forward. Demo patients plus any patients the user added. A bed that is assigned is occupied from
    the assignment time on, and its patient stops claiming.
    Assignments come from two places and are treated the same once made:
      the user, in the simulation (assigns), and
      early prep, for any patient whose admit decision is made (demo or simulated: a patient in the bed queue counts as approved): the moment the bed
      they hold is ready (it just finished cleaning, or was already ready) it is assigned to them and is occupied again.
      A bed nobody holds just stays ready. The patient must have held that same bed on the previous tick, so a bed never goes occupied straight from a non-outlined state.
    exclude: visit ids that never claim a bed here (the scripted patients that live only on the Demo Scenarios tab).
    Each state carries res["assigns"], every assignment made up to that time."""
    holds, out = None, []
    user = dict(assigns or {})
    auto = {}
    for t in times:
        t = pd.Timestamp(t)
        everything = {**auto, **user}
        b, gone = board_with_assigns(beds, t, everything)
        cl = [c for c in claimants_at(t, J, pri, keep_matched=True) if c["visit_id"] not in gone and c["visit_id"] not in exclude] + sim_claimants(t, sims, pri, gone)
        fc = forecast_at_now if (now is not None and t == pd.Timestamp(now)) else None
        res = plan(t, b, cl, blockers, pri, prior_holds=holds, forecast=fc)
        # A bed only turns occupied after the patient already held it on the previous tick (soft, expedited or admitted), so the hold is always visible
        # first. Only a human assignment (user, above) can skip that.
        seen = {(h["visit_id"], h["bed_id"]) for h in (holds or [])}
        landed = [h for h in res["holds"] if h["kind"] == "decided" and h["visit_id"] and (h["visit_id"], h["bed_id"]) in seen
                  and h["bed_id"] not in everything and (b.loc[b["bed_id"] == h["bed_id"], "status"] == "ready").any()]
        if landed:
            dec_of = {c["visit_id"]: c.get("decision_time") for c in cl}
            for h in landed:
                row = b[b["bed_id"] == h["bed_id"]].iloc[0]
                at = max([x for x in (dec_of.get(h["visit_id"]), row["expected_ready_time"]) if x is not None and pd.notna(x)] or [t])
                at = min(t, max(pd.Timestamp(at), t - pd.Timedelta(minutes=14)))      # the moment it became possible, not the next 15-minute tick
                auto[h["bed_id"]] = {"patient_id": h["visit_id"], "role": "Early prep (approved patient)", "time": f"{at:%H:%M}", "level": h["level_assumed"],
                                     "overridden": False, "assign_time": at, "auto": True}
            done = {h["hold_id"] for h in landed}
            res["holds"] = [h for h in res["holds"] if h["hold_id"] not in done]
            res["tasks"] = [k for k in res["tasks"] if k["hold_id"] not in done]
            b, _ = board_with_assigns(beds, t, {**auto, **user})
        res["claimants"], res["board"], res["assigns"] = cl, b, {**auto, **user}
        holds = res["holds"]
        out.append(res)
    return out
