"""
test_early_prep.py -- checks for the early-prep loop. Run:  python test_early_prep.py     (or: pytest test_early_prep.py)

  1  no bed is ever held twice, at any replay moment
  2  tiering: soft holds only on ready/cleaning beds; discharge-pending beds only from 60% (or once decided)
  3  hysteresis and release: a hold stays between 28% and 40%, and is released below 28%
  4  a decision at a different level of care releases the hold with reason wrong_level
  5  payer (and every other prohibited field) never enters the loop
  6  task chains reach the right roles, in the right order, with the config durations (tail 200 min normal / 149 min expedited)
  7  placement (days) is excluded unless the bed is needed 24 h or more away
  8  the mark-done cascade moves the tasks after it
  9  a pooled hold is claimed by the next patient who crosses a threshold
 10  the rows written for tasks and bed_holds pass the schema checks; a replay estimate never reports negative savings
"""
import inspect
import json
import re
import sys

import pandas as pd

import admit_model as AM
import case_engine as E
import config as C
import early_prep as EP
import journey as JR
import schema as S

D = C.DATA_DIR
T0 = pd.Timestamp(f"{C.DEMO_DATE} 10:30")
BEDS = pd.read_csv(f"{D}/beds.csv", parse_dates=["status_updated_time", "expected_ready_time"])
NOTES = pd.read_csv(f"{D}/inpatient_notes.csv", parse_dates=["note_time"])
LABELS = pd.read_csv(f"{D}/note_labels.csv")
PRI = EP.Priors()


def blockers_from_key():
    """Open blockers from the answer key for the latest note on each discharge-pending bed (so the tests do not call the language model)."""
    out = {}
    for _, b in BEDS[BEDS["discharge_pending"]].iterrows():
        n = NOTES[NOTES["bed_id"] == b["bed_id"]].sort_values("note_time").iloc[-1]
        f = LABELS[LABELS["note_id"] == n["note_id"]]
        out[b["bed_id"]] = {"visit_id": b["current_visit_id"],
                            "findings": [{"blocker_type": r.blocker_type, "owner_role": r.owner_role} for r in f.itertuples()]}
    return out


BLK = blockers_from_key()


def cl(vid, p, level="med_surg", kind="ed", needed_min=60, iso=False, pref=None, t=T0):
    return {"visit_id": vid, "kind": kind, "p": p, "level": level, "level_confidence": 0.8, "isolation": iso, "preferred_unit": pref,
            "decision_time": t if kind == "decided" else None, "needed_by": t + pd.Timedelta(minutes=needed_min), "pending": 1, "complaint": "abdominal_pain"}


def run(claimants, t=T0, beds=None, prior=None, fc=None):
    return EP.plan(t, EP.board_at(BEDS if beds is None else beds, t), claimants, BLK, PRI, prior_holds=prior, forecast=fc)


def test_no_bed_held_twice():
    J = JR.load(AM.AdmitModel.load())
    man = json.load(open(f"{D}/demo_manifest.json"))
    times = [pd.Timestamp(f"{C.DEMO_DATE} {x}") for x in ("10:30", "10:40", "10:46", "11:05", "11:30", "12:05", "12:28", "13:00")]
    states = EP.replay(times, J, BEDS, BLK, PRI, man["forecast_at_now"], now=T0)
    for s in states:
        beds = [h["bed_id"] for h in s["holds"]]
        assert len(beds) == len(set(beds)), f"a bed is held twice at {s['time']}: {beds}"
        ids = [h["visit_id"] for h in s["holds"] if h["visit_id"]]
        assert len(ids) == len(set(ids)), "a patient holds two beds"


def test_tiering():
    r = run([cl("S1", 0.45), cl("S2", 0.52), cl("H1", 0.70), cl("U1", 0.90), cl("S3", 0.41, needed_min=600)])
    for h in r["holds"]:
        if h["kind"] == "soft":
            assert h["tier"] in (1, 2), "a soft hold must never start discharge work on an occupied bed"
        if h["tier"] == 3 and h["visit_id"]:
            assert h["kind"] in ("hard", "decided")
    kinds = {h["visit_id"]: h["kind"] for h in r["holds"] if h["visit_id"]}
    assert kinds.get("S1") in (None, "soft") and kinds.get("H1") in (None, "hard") and kinds.get("U1") in (None, "hard")
    below = run([cl("N1", 0.39)])
    assert not [h for h in below["holds"] if h["visit_id"] == "N1"], "below 40% nothing is held"


def test_hysteresis_and_release():
    r1 = run([cl("P1", 0.55)])
    assert [h for h in r1["holds"] if h["visit_id"] == "P1"], "a 55% patient gets a hold"
    r2 = run([cl("P1", 0.33)], t=T0 + pd.Timedelta(minutes=10), prior=r1["holds"])
    assert [h for h in r2["holds"] if h["visit_id"] == "P1"], "between 28% and 40% a hold stays"
    r3 = run([cl("P1", 0.20)], t=T0 + pd.Timedelta(minutes=20), prior=r2["holds"])
    assert not [h for h in r3["holds"] if h["visit_id"] == "P1"], "below 28% the hold is released"
    assert any(e["action"] == "hold_released" and e["detail"] == "p_dropped" for e in r3["events"])


def test_wrong_level_release():
    r1 = run([cl("W1", 0.70, level="med_surg")])
    old = next(h for h in r1["holds"] if h["visit_id"] == "W1")
    r2 = run([cl("W1", 1.0, level="telemetry", kind="decided")], t=T0 + pd.Timedelta(minutes=30), prior=r1["holds"])
    assert any(e["action"] == "hold_released" and e["detail"] == "wrong_level" and e["bed_id"] == old["bed_id"] for e in r2["events"])
    new = next(h for h in r2["holds"] if h["visit_id"] == "W1")
    assert new["level_assumed"] == "telemetry"


def test_payer_never_used():
    """Identity fields never enter the loop. Decision fields (disposition, requested level) are read only once the doctor has decided."""
    import ast, textwrap

    def code_only(obj):                                   # drop comments and docstrings: only executable code counts
        tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
        for n in ast.walk(tree):
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr) \
                    and isinstance(getattr(n.body[0], "value", None), ast.Constant) and isinstance(n.body[0].value.value, str):
                n.body = n.body[1:] or [ast.Pass()]
        return ast.unparse(tree)
    src = code_only(EP) + code_only(E.prep_task_rows) + code_only(E.prep_hold_rows)
    for bad in ("payer", "language", "race", "ethnicity"):
        assert not re.search(rf"\b{bad}\b", src, re.I), f"{bad} appears in the loop code"
    J = JR.load(AM.AdmitModel.load())
    undecided = [c for c in EP.claimants_at(T0, J, PRI) if c["kind"] == "ed"]
    assert undecided, "there should be undecided likely patients at the demo 'now'"
    for c in undecided:
        assert not (set(c) & {"payer", "language", "race", "ethnicity", "requested_level_of_care", "disposition", "disposition_time"}), \
            "an undecided patient carries a field only known after the decision"
        snap = J["snap"][(J["snap"]["visit_id"] == c["visit_id"]) & (J["snap"]["snapshot_time"] <= T0)].iloc[-1]
        lv, conf, _ = PRI.level_guess(c["complaint"], snap["esi"])
        assert c["level"] == lv, "an undecided patient's bed type must come from the priors only"
    twin = ("V-900019", "V-900020")                      # same patient, different payer: identical plan inputs
    cc = {c["visit_id"]: c for c in EP.claimants_at(T0, J, PRI)}
    if all(t in cc for t in twin):
        a, b = cc[twin[0]], cc[twin[1]]
        assert abs(a["p"] - b["p"]) < 0.005 and a["level"] == b["level"] and a["needed_by"] - b["needed_by"] in (pd.Timedelta(minutes=0), pd.Timedelta(minutes=-5), pd.Timedelta(minutes=5))


def test_task_chain_roles_order_durations():
    chain, ready = EP.schedule_chain(["paperwork"], T0, expedited_after=None)
    assert [k["task_type"] for k in chain] == ["paperwork", "discharge_signoff", "patient_leaves", "bed_cleaning"]
    owners = {k["task_type"]: k["owner_role"] for k in chain}
    assert owners == {"paperwork": "case_management", "discharge_signoff": "inpatient_physician", "patient_leaves": "inpatient_nurse",
                      "bed_cleaning": "environmental_services"}
    assert (ready - T0) == pd.Timedelta(minutes=60 + 20 + 90 + 90)
    chain2, ready2 = EP.schedule_chain(["paperwork"], T0, expedited_after=T0)
    assert (ready2 - T0) == pd.Timedelta(minutes=30 + 15 + 66 + 68)
    tail_n = sum(C.PREP_TASK_MINUTES[t][0] for t in C.PREP_TAIL_TASKS); tail_e = sum(C.PREP_TASK_MINUTES[t][1] for t in C.PREP_TAIL_TASKS)
    assert (tail_n, tail_e) == (200, 149), "the tail after the last blocker is 200 min normal / 149 min expedited"
    two, _ = EP.schedule_chain(["paperwork", "pt_evaluation"], T0)
    assert two[0]["start"] == two[1]["start"], "blockers run in parallel"
    sign = next(k for k in two if k["task_type"] == "discharge_signoff")
    assert sign["start"] == max(k["due"] for k in two[:2]), "sign-off waits for the slowest blocker"
    r = run([cl("T1", 0.9), cl("T2", 0.88), cl("T3", 0.86)])
    for k in r["tasks"]:
        assert k["owner_role"] in S.ROLES and k["task_type"] in S.TASK_TYPES


def test_placement_excluded_unless_far_ahead():
    assert not EP.chain_feasible(["placement"], T0 + pd.Timedelta(hours=8), T0)
    assert EP.chain_feasible(["placement"], T0 + pd.Timedelta(hours=30), T0)
    assert EP.chain_feasible(["paperwork"], T0 + pd.Timedelta(hours=1), T0)


def test_mark_done_cascade():
    chain, ready = EP.schedule_chain(["paperwork"], T0, expedited_after=T0)
    tasks = [{"task_key": f"K{i}", "hold_id": "H", "bed_id": "B", "task_type": k["task_type"], "owner_role": k["owner_role"], "start_by_time": k["start"],
              "due_time": k["due"], "duration_min": k["minutes"], "depends_on_key": None if k["depends_on"] is None else [f"K{j}" for j in k["depends_on"]]}
             for i, k in enumerate(chain)]
    moved, new_ready = EP.mark_done(tasks, "K0", T0 + pd.Timedelta(minutes=10))
    assert new_ready == ready - pd.Timedelta(minutes=20), "finishing the first task 20 min early moves the bed-ready time 20 min earlier"
    moved2, late_ready = EP.mark_done(tasks, "K0", T0 + pd.Timedelta(minutes=50))
    assert late_ready == ready + pd.Timedelta(minutes=20)
    st = EP.task_state(tasks, T0 + pd.Timedelta(minutes=5))
    assert st["K0"] == "in_progress" and st["K1"] == "waiting"


def test_pooled_hold_claimed_by_next_patient():
    fc = {"4h": 9.0, "8h": 16.0}
    # only a discharge-pending bed is left for med-surg, and demand exceeds supply
    r1 = run([cl("D%d" % i, 1.0, level=lv, kind="decided") for i, lv in enumerate(["med_surg"] * 3 + ["telemetry"] * 3 + ["icu"] * 2)], fc=fc)
    assert r1["pool"], "a shortfall must start a pooled hold"
    pb = r1["pool"][0]
    lv = pb["level_assumed"]
    r2 = run([cl("N1", 0.9, level=lv)], t=T0 + pd.Timedelta(minutes=20), prior=r1["holds"], fc=None)
    got = next((h for h in r2["holds"] if h["visit_id"] == "N1"), None)
    assert got is not None
    claimed = [h for h in r2["holds"] if h["visit_id"] == "N1" and h["pooled_from"]]
    assert claimed or got["bed_id"] != pb["bed_id"], "the patient claims the pooled bed when it is the best fit"


def test_rows_pass_schema_and_estimate_never_negative():
    J = JR.load(AM.AdmitModel.load())
    man = json.load(open(f"{D}/demo_manifest.json"))
    res = EP.replay([T0], J, BEDS, BLK, PRI, man["forecast_at_now"], now=T0)[0]
    th, tt = E.prep_hold_rows(res), E.prep_task_rows(res, T0)
    assert not S.validate(th, "bed_holds"), S.validate(th, "bed_holds")
    assert not S.validate(tt, "tasks"), S.validate(tt, "tasks")
    a = man["scripts"]["A"]["visit_id"]
    res2 = EP.replay([pd.Timestamp(f"{C.DEMO_DATE} 10:46")], J, BEDS, BLK, PRI, man["forecast_at_now"], now=T0)[0]
    h = next(x for x in res2["holds"] if x["visit_id"] == a)
    est = EP.replay_estimate(J["visits"].loc[a], h)
    assert est["saved_min"] >= 0 and est["boarding_with_loop_h"] >= 0


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try:
            fn(); print(f"PASS  {name}")
        except Exception as e:
            fails += 1; print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print("\nALL PASSED" if not fails else f"\n{fails} FAILED")
    sys.exit(1 if fails else 0)
