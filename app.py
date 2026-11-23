"""
app.py -- the Boarding Command Center demo.  Run:  streamlit run app.py

Type a case -> the AI fills the form (you can correct it) -> admit score with the arithmetic shown ->
bed matches with reason codes -> blockers read from the notes of beds that are waiting on a discharge ->
a named person confirms or overrides. All data is synthetic. Do not type real patient information.
"""
import html

import pandas as pd
import streamlit as st

import admit_model as AM
import case_engine as E
import config as C
import extract_blockers as EB
import fairness as FR
import journey as JR
import intake as I
import llm_client as L

st.set_page_config(page_title="Boarding Command Center", layout="wide")

HEAD_CSS = """<style>
/* one heading scale for every page: H1 page title, H2 demo time, H3 section/side, H4 step/panel/unit, H5 small label */
[data-testid="stMain"] h1{font-size:2rem !important;font-weight:700;padding:.25rem 0 .5rem}
[data-testid="stMain"] h2{font-size:1.6rem !important;font-weight:700;padding:.75rem 0 .25rem}
[data-testid="stMain"] h3{font-size:1.3rem !important;font-weight:700;padding:.6rem 0 .25rem}
[data-testid="stMain"] h4{font-size:1.1rem !important;font-weight:650;padding:.5rem 0 .2rem}
[data-testid="stMain"] h5{font-size:1rem !important;font-weight:650;padding:.4rem 0 .15rem}
[data-testid="stSidebar"] h2{font-size:1.25rem !important;font-weight:700}
</style>"""
st.markdown(HEAD_CSS, unsafe_allow_html=True)

# --- hosted deployment: copy keys from Streamlit secrets into the environment (nothing is read if there are none) ---
import os
for _k in ("GROQ_API_KEY", "GEMINI_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_CACHE_ONLY", "APP_PASSWORD"):
    try:
        if _k not in os.environ and _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        pass

# --- optional shared password so only teammates can use the key. Set APP_PASSWORD in secrets to turn it on. ---
if os.environ.get("APP_PASSWORD"):
    if st.session_state.get("authed") != True:
        st.title("Boarding Command Center")
        pw = st.text_input("Team password", type="password")
        if pw:
            if pw == os.environ["APP_PASSWORD"]:
                st.session_state["authed"] = True
                st.rerun()
            else:
                st.error("Wrong password.")
        st.stop()

EXAMPLES = {
    "A: looks mild, then isn't": "35-49 year old with abdominal pain, ESI 3, arrived 94 minutes ago. Reassessed 7 minutes ago: HR 122, SpO2 92.5%, "
                                 "BP 96/60, RR 25, temp 38.9 C. Lab panel resulted abnormal. CT ordered, result pending. IV medication given. "
                                 "Doctor has decided to admit to a med-surg bed.",
    "B: needs an isolation bed": "65-74 year old with shortness of breath, ESI 2, here 150 minutes. HR 104, SpO2 91%, BP 142/88, RR 26, temp 38.2 C. "
                                 "ECG normal, chest x-ray abnormal, lab panel abnormal. Admit decision made: telemetry bed, isolation required for suspected influenza.",
    "C: chest pain, admit": "50-64 year old with chest pain, ESI 2, 110 minutes since arrival. HR 96, SpO2 95%, BP 158/92, RR 20, temp 36.9 C. "
                            "ECG abnormal, troponin resulted normal, lab panel resulted normal. Admit to telemetry, no isolation needed.",
    "D: not decided yet": "18-34 year old with ankle injury after a fall, ESI 4, 40 minutes since arrival. HR 78, SpO2 99%, BP 124/76, RR 16, temp 36.7 C. "
                          "X-ray ordered and pending.",
}
BAND_COLOR = {"unlikely": "#6B7280", "possible": "#B45309", "likely": "#2563EB"}
ROLES = {"bed_manager": "Bed manager", "ed_charge_nurse": "ED charge nurse"}
REASON = C.MATCH_REASON_CODES


@st.cache_resource
def world():
    return E.load_world()


W = world()


# ------------------------------------------------------------------ state
RESULT_STATES = ["pending", "reassuring", "concerning"]


def _orders_frame(orders):
    """Orders table: order, result received (pending / reassuring / concerning) and free-text notes. Fixed column types so an empty table cannot crash st.data_editor."""
    rows = [{"name": o["name"], "result": ("pending" if not o.get("resulted") else ("concerning" if o.get("abnormal") else "reassuring")), "notes": o.get("notes", "")}
            for o in (orders or [])]
    df = pd.DataFrame(rows, columns=["name", "result", "notes"])
    return df.astype({"name": str, "result": str, "notes": str})


def _o(name, resulted=False, abnormal=False):
    return {"name": name, "resulted": resulted, "abnormal": abnormal}


# the example cases as structured forms, so picking one fills every step without needing the AI key
EXAMPLE_FORMS = {
    "A: looks mild, then isn't": dict(f_age="35-49", f_complaint="abdominal_pain", f_esi=3, f_hr=122.0, f_spo2=92.5, f_sbp=96.0, f_rr=25.0, f_temp=38.9,
                                      f_arr=94.0, f_last=7.0, f_admit=True, f_level="med_surg", f_iso=False,
                                      orders=[_o("lab_panel", True, True), _o("CT"), _o("IV_medication")]),
    "B: needs an isolation bed": dict(f_age="65-74", f_complaint="shortness_of_breath", f_esi=2, f_hr=104.0, f_spo2=91.0, f_sbp=142.0, f_rr=26.0, f_temp=38.2,
                                      f_arr=150.0, f_last=142.0, f_admit=True, f_level="telemetry", f_iso=True,
                                      orders=[_o("ECG", True, False), _o("chest_xray", True, True), _o("lab_panel", True, True)]),
    "C: chest pain, admit": dict(f_age="50-64", f_complaint="chest_pain", f_esi=2, f_hr=96.0, f_spo2=95.0, f_sbp=158.0, f_rr=20.0, f_temp=36.9,
                                 f_arr=110.0, f_last=102.0, f_admit=True, f_level="telemetry", f_iso=False,
                                 orders=[_o("ECG", True, True), _o("troponin", True, False), _o("lab_panel", True, False)]),
    "D: not decided yet": dict(f_age="18-34", f_complaint="injury_trauma", f_esi=4, f_hr=78.0, f_spo2=99.0, f_sbp=124.0, f_rr=16.0, f_temp=36.7,
                               f_arr=40.0, f_last=32.0, f_admit=False, f_level="med_surg", f_iso=False, orders=[_o("chest_xray")]),
}


def _apply_example(name):
    f = dict(EXAMPLE_FORMS[name])
    orders = f.pop("orders")
    f.pop("f_admit", None)
    f["f_iso_choice"] = "Yes" if f.pop("f_iso", False) else "No"
    s = st.session_state
    s.update(f)
    s["f_approve"] = False
    s["f_text"] = EXAMPLES[name]
    s["f_orders"] = _orders_frame(orders)
    s["orders_ver"] = s.get("orders_ver", 0) + 1
    s["parse_info"] = None
    s["decision"] = None


def _defaults():
    return {"f_text": EXAMPLES["A: looks mild, then isn't"], "orders_ver": 0, "f_orders": _orders_frame([]), "parse_info": None, "decision": None,
            "example_pick": "A: looks mild, then isn't"}


for k, v in _defaults().items():
    st.session_state.setdefault(k, v)
if "f_age" not in st.session_state:
    _apply_example("A: looks mild, then isn't")


def load_example():
    _apply_example(st.session_state["example_pick"])


def do_queue():
    """'Submit to AI Queue': read the note with the AI, then put the patient on the bed board as a provisional (not yet admitted) patient."""
    s = st.session_state
    do_parse()
    s["provisional"] = {"patient_id": f"SIM-{s['submitted_n'] + 1:03d}", "sim": None}     # the sim is filled in on this same run, once the case is built


def do_parse():
    s = st.session_state
    try:
        res = I.parse_case(s["f_text"])
    except L.LLMUnavailable as e:
        s["parse_info"] = {"error": str(e)}
        return
    c = res["case"]
    s["parse_info"] = {"warnings": res["warnings"], "source": res["source"], "model": res["model"], "evidence": c["evidence"]}
    if c["age_band"]:
        s["f_age"] = c["age_band"]
    if c["complaint_category"]:
        s["f_complaint"] = c["complaint_category"]
    if c["esi"]:
        s["f_esi"] = c["esi"]
    for k, key in (("hr", "f_hr"), ("spo2", "f_spo2"), ("sbp", "f_sbp"), ("rr", "f_rr"), ("temp_c", "f_temp")):
        if c["vitals"].get(k) is not None:
            s[key] = float(c["vitals"][k])
    if c["minutes_since_arrival"] is not None:
        s["f_arr"] = float(c["minutes_since_arrival"])
    s["f_last"] = float(c["minutes_since_last_reassessment"]) if c["minutes_since_last_reassessment"] is not None else max(0.0, s["f_arr"] - 8)
    if c["requested_level_of_care"]:
        s["f_level"] = c["requested_level_of_care"]
    s["f_iso_choice"] = "Yes" if c["isolation_required"] else "No"
    s["f_orders"] = _orders_frame(c["orders"])
    s["orders_ver"] += 1
    s["decision"] = None


def current_case():
    s = st.session_state
    orders = [{"name": r["name"], "resulted": r["result"] in ("reassuring", "concerning"), "abnormal": r["result"] == "concerning", "notes": r.get("notes", "")}
              for _, r in s["orders_table"].iterrows() if r["name"] in I.ORDERS]
    return {"age_band": s["f_age"], "complaint_category": s["f_complaint"], "esi": s["f_esi"],
            "vitals": {"hr": s["f_hr"], "spo2": s["f_spo2"], "sbp": s["f_sbp"], "rr": s["f_rr"], "temp_c": s["f_temp"]},
            "orders": orders, "minutes_since_arrival": s["f_arr"], "minutes_since_last_reassessment": s["f_last"],
            "admit_decided": False, "requested_level_of_care": None, "isolation_required": s["f_iso_choice"] == "Yes"}


def highlight(text, phrases):
    out = html.escape(text)
    for p in phrases:
        out = out.replace(html.escape(p), f"<mark style='background:#FDE68A;padding:1px 3px;border-radius:3px'>{html.escape(p)}</mark>", 1)
    return out


def badge(text, color):
    return f"<span style='background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:0.85em;font-weight:600'>{html.escape(text)}</span>"



# ------------------------------------------------------------------ shared pieces
import json as _json
import importlib
import early_prep as EP
_ep_mtime = os.path.getmtime(EP.__file__)
if getattr(EP, "_loaded_mtime", None) != _ep_mtime:     # Streamlit keeps imported modules alive across redeploys; reload early_prep whenever its file changed
    EP = importlib.reload(EP)
    EP._loaded_mtime = _ep_mtime

st_ = L.status()
USE_LLM = st_["key_present"] or st_["cache_only"]
STATUS_NAME = {"ready": "Ready now", "cleaning": "Being cleaned", "occupied": "Occupied", "blocked": "Blocked"}


@st.cache_resource(show_spinner="Scoring every step of the demo patients...")
def journey_data():
    return JR.load(W["model"])


@st.cache_resource
def priors():
    return EP.Priors()


TITLES = {}
try:
    for _k, _v in _json.load(open(f"{C.DATA_DIR}/demo_manifest.json"))["scripts"].items():
        TITLES[_v["visit_id"]] = f"{_k}: {_v['title']}"
except Exception:
    pass


def who_label(v):
    return TITLES.get(v, v) if v else "pooled (no named patient yet)"


def step_title(n, title, hint=None):
    st.markdown(f"#### Step {n} · {title}")
    if hint:
        st.caption(hint)


def show_vitals(vit):
    flags = E.abnormal_flags(vit)
    cols = st.columns(5)
    for col, (k, label, fmt) in zip(cols, (("hr", "HR (worst)", "{:.0f}"), ("spo2", "SpO2 % (low)", "{:.1f}"), ("sbp", "Sys. BP (low)", "{:.0f}"),
                                           ("rr", "RR (worst)", "{:.0f}"), ("temp_c", "Temp °C (high)", "{:.1f}"))):
        col.metric(label, fmt.format(vit[k]))
        if flags.get(k):
            col.caption(":red[⚠ abnormal]")


def sc_from_snapshot(JD, row, idx):
    cw = JD["contrib"].loc[idx]
    rows = [{"feature": JR.LABEL[f], "value": row[f], "push": float(cw[f])} for f in JR.FIELDS]
    rows.sort(key=lambda r: -abs(r["push"]))
    base = float(JD["baseline"])
    return {"p": float(row["p"]), "band": str(AM.AdmitModel.category([float(row["p"])])[0]), "baseline_logodds": base,
            "total_logodds": base + sum(r["push"] for r in rows), "rows": rows, "top3": rows[:C.TOP_FACTORS_SHOWN]}


def show_score(sc, assumed=None, prev_p=None):
    k1, k2 = st.columns([1, 2])
    with k1:
        st.markdown(f"<div style='font-size:3em;font-weight:700;color:{BAND_COLOR[sc['band']]}'>{sc['p']:.0%}</div>"
                    + badge(f"{sc['band']} admit", BAND_COLOR[sc["band"]]), unsafe_allow_html=True)
        if prev_p is not None:
            st.caption(f"{(sc['p'] - prev_p) * 100:+.0f} points since the last score")
        st.caption("Chance this patient is admitted, from what is known right now.")
        for r in sc["top3"]:
            st.write(f"{'▲' if r['push'] >= 0 else '▼'} {r['feature']} ({r['push']:+.2f})")
    with k2:
        st.caption(f"Typical patient starts at {sc['baseline_logodds']:+.2f} (log-odds). Each bar pushes the score up or down. "
                   f"Total {sc['total_logodds']:+.2f} → {sc['p']:.1%}.")
        tab = pd.DataFrame(sc["rows"])
        tab["value"] = tab["value"].map(lambda x: f"{x:.2f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x))
        st.bar_chart(tab.set_index("feature")["push"], horizontal=True, height=280)
        with st.expander("Show the numbers"):
            st.dataframe(tab.rename(columns={"feature": "Factor", "value": "Value used", "push": "Push (log-odds)"}), hide_index=True, width="stretch")
    if assumed:
        st.info("Not stated, so typical values were used: " + ", ".join(assumed))
    st.caption("Payer, race and language are never used by the score.")


def show_request(case):
    pat = E.patient_for_matching(case)
    a, b, c = st.columns(3)
    a.metric("Bed type requested", str(case["requested_level_of_care"]).replace("_", "-"))
    b.metric("Isolation", "required" if case["isolation_required"] else "not needed")
    c.metric("Preferred unit", pat["preferred_unit"] or "none")


def bed_cards(mt, blockers):
    cols = st.columns(len(mt["ranked"]))
    for col, r in zip(cols, mt["ranked"]):
        with col.container(border=True):
            st.markdown(f"##### #{r['rank']} {r['bed_id']}" + ("  ·  best fit" if r["rank"] == 1 else ""))
            when = "ready now" if r["status"] == "ready" else f"ready in ~{r['minutes_until_ready']} min"
            st.caption(f"{r['unit']} unit · {r['bed_type']}{' · isolation-capable' if r['isolation_capable'] else ''}")
            st.write(f"{r['status'].capitalize()} · {when}" + (" (waiting on a discharge)" if r["discharge_pending"] else ""))
            st.metric("Score", f"{r['score']:.2f}")
            st.caption(" · ".join(REASON[x] for x in r["reasons"]) or "No bonus reasons")
            bl = blockers.get(r["bed_id"])
            if bl:
                if bl["findings"]:
                    st.markdown(badge("Discharge may slip", "#B45309"), unsafe_allow_html=True)
                    for f in bl["findings"]:
                        st.write(f"• **{f['blocker_type'].replace('_', ' ')}** → {f['owner_role'].replace('_', ' ')} ({f['confidence']:.0%})")
                else:
                    st.markdown(badge("No blockers in latest note", "#15803D"), unsafe_allow_html=True)
                with st.expander("Note and evidence"):
                    st.markdown(highlight(bl["note_text"], [f["evidence_phrase"] for f in bl["findings"]]), unsafe_allow_html=True)
                    st.caption(f"{bl['note_id']} · read by {bl['model']} ({bl['source']})"
                               + (f" · AI unavailable, used keyword rules: {bl['unavailable_reason']}" if bl.get("unavailable_reason") else ""))


TIER_TEXT = {"soft": "soft tier: hold a ready or cleaning bed (reversible) and draft the bed request. Nothing is sent.",
             "expedite": "expedite tier: send the bed request now and, if the best bed is still occupied, start its discharge and cleaning tasks.",
             "urgent": "urgent tier: as expedite, with the tasks marked urgent and escalated."}


def early_prep_preview(case, p, pending, board, now, hold=None):
    """Step 5 before a doctor decides: what the early-prep loop would do for this one patient (no competing patients; the Bed Board tab has those)."""
    tier = EP.tier_name(p)
    if tier is None:
        st.info(f"No admit decision yet, and the score ({p:.0%}) is below the {C.PREP_SOFT_THRESHOLD:.0%} trigger, so the system only counts this patient "
                "toward the forecast of expected admits. No bed is held.")
        return
    pri = priors()
    lv, share, _ = pri.level_guess(case["complaint_category"], case["esi"])
    iso = pri.isolation_guess(case["complaint_category"], case["esi"])
    needed = pri.minutes_to_decision(pending)
    guess = {**case, "requested_level_of_care": lv, "isolation_required": iso}
    mt = E.match_beds(guess, board, now=now, top_n=1) if not hold else {"ranked": []}
    st.success(f"No decision yet, but the score ({p:.0%}) is in the {TIER_TEXT[tier]}")
    st.write(f"Assumed bed type: **{lv.replace('_', '-')}** ({share:.0%} of similar admits)" + (", **isolation**" if iso else "")
             + f". Earliest likely decision: about **{needed:.0f} min** from now (25th percentile, with {int(pending)} result(s) pending).")
    if hold:
        ready = pd.Timestamp(hold["realistic_ready_time"])
        st.write(f"Bed it holds: **{hold['bed_id']}**, held since {pd.Timestamp(hold['created_time']):%H:%M} ({hold['kind']} hold), realistic ready time {ready:%H:%M}. A person confirms before anything is sent.")
    elif mt["ranked"]:
        r = mt["ranked"][0]
        when = "ready now" if r["status"] == "ready" else f"ready in ~{r['minutes_until_ready']} min"
        st.write(f"Bed it would hold: **{r['bed_id']}** ({r['unit']}, {r['status']}, {when}). A person confirms before anything is sent.")
    else:
        st.warning("No suitable bed can be prepared in time.")


def matching_step(case, p, pending, board, now, record_key=None, case_id=None, hold=None):
    """Step 5. Decided -> ranked beds (and, for the typed case, the confirm/override form). Not decided -> early-prep preview."""
    if not case["admit_decided"]:
        early_prep_preview(case, p, pending, board, now, hold)
        return
    mt = E.match_beds(case, board, now=now)
    pat = mt["patient"]
    st.write(f"Needs **{pat['requested_level_of_care'].replace('_', '-')}**" + (" with **isolation**" if pat["isolation_required"] else "")
             + (f", preferred unit **{pat['preferred_unit']}**" if pat["preferred_unit"] else "")
             + f". {mt['n_feasible']} of {mt['n_beds']} beds can take this patient now or soon.")
    if not mt["ranked"]:
        st.error("No bed fits. Escalate to the bed manager.")
        return
    blockers = E.annotate_blockers(mt["ranked"], board, W["notes"], use_llm=USE_LLM)
    bed_cards(mt, blockers)
    if record_key is None:
        return
    st.markdown("##### A named person confirms or overrides")
    dcs = st.session_state.get("decision")
    if dcs:
        st.success(f"{'Override' if dcs['overridden'] else 'Confirmed'} by {dcs['role']} at {dcs['time']}: {dcs['bed']}. "
                   f"Wrote {dcs['matches']} match rows, {dcs['tasks']} tasks, {dcs['audit']} audit entries (see the Audit trail page).")
    ids = [r["bed_id"] for r in mt["ranked"]]
    others = [b for b in board["bed_id"] if b not in ids]
    who = st.radio("Decided by", list(ROLES), format_func=ROLES.get, horizontal=True, key=f"{record_key}_who")
    pick = st.radio("Bed", ids + ["(another bed)"], horizontal=True, key=f"{record_key}_pick")
    bed_id = st.selectbox("Which bed?", others, key=f"{record_key}_other") if pick == "(another bed)" else pick
    override = bed_id != ids[0]
    reason = st.text_input("Reason (required when you do not take the AI's top pick)", key=f"{record_key}_reason") if override else ""
    if st.button(f"{'Override and assign' if override else 'Confirm'} {bed_id}", type="primary", disabled=override and not reason.strip(), key=f"{record_key}_go"):
        res = E.record_decision(case_id, mt["ranked"], bed_id, who, reason, blockers)
        st.session_state["decision"] = {**res, "role": ROLES[who], "bed": bed_id, "time": pd.Timestamp.now().strftime("%H:%M:%S")}
        st.rerun()



LIVE_CSS = """<style>
div[data-testid="stColumn"]:has(.st-key-live_sim), div[data-testid="column"]:has(.st-key-live_sim),
div[data-testid="stColumn"]:has(.st-key-live_sim_frozen), div[data-testid="column"]:has(.st-key-live_sim_frozen)
{position: sticky; top: 3.6rem; align-self: flex-start; max-height: calc(100vh - 4.5rem); overflow-y: auto;}
.st-key-live_sim_frozen {opacity: 0.45; filter: grayscale(1);}
</style>"""

TEMPLATE_TEXT = ("[age band] year old with [complaint], ESI [1-5], [N] minutes since arrival. Reassessed [N] minutes ago: HR [..], SpO2 [..]%, BP [sys]/[dia], "
                 "RR [..], temp [..] C. [Order] ordered, result [pending / reassuring / concerning]. [Next order] ... Doctor has decided to admit to a "
                 "[med-surg / telemetry / ICU] bed[, isolation required].")


# ------------------------------------------------------------------ simulation state shared by the two Simulate tabs
def _sim_init():
    s = st.session_state
    s.setdefault("queue", [])
    s.setdefault("provisional", None)
    s.setdefault("assignments", {})
    s.setdefault("frozen", False)
    s.setdefault("score_hist", [])
    s.setdefault("submitted_n", 0)
    s.setdefault("f_approve", False)
    s.setdefault("f_arrive", "09:00")


SIM_TRIAGE_MIN, SIM_ORDERS_MIN, SIM_RESULTS_MIN, SIM_DECISION_MIN = 10, 30, 75, 120    # typical minutes from arrival [ASSUMPTION], same lead as the model's median


def sim_timeline(case, arrival):
    """Score steps for a patient the user adds. The model rescores at triage, when orders go in (results pending) and when results return,
    using the case the user typed. The doctor decides SIM_DECISION_MIN after arrival."""
    orders = list(case.get("orders", []))
    variants = [(SIM_TRIAGE_MIN, {**case, "orders": [], "minutes_since_arrival": float(SIM_TRIAGE_MIN), "minutes_since_last_reassessment": 2.0}),
                (SIM_ORDERS_MIN, {**case, "orders": [{**o, "resulted": False, "abnormal": False} for o in orders],
                                  "minutes_since_arrival": float(SIM_ORDERS_MIN), "minutes_since_last_reassessment": 8.0}),
                (SIM_RESULTS_MIN, {**case, "minutes_since_arrival": float(SIM_RESULTS_MIN), "minutes_since_last_reassessment": 20.0})]
    steps = []
    for mins, c in variants:
        sc_ = E.score_case(c, W)
        steps.append({"time": arrival + pd.Timedelta(minutes=mins), "p": sc_["p"], "pending": int(sc_["features"]["pending_results_so_far"])})
    return steps, arrival + pd.Timedelta(minutes=SIM_DECISION_MIN)


def sim_restart():
    s = st.session_state
    s["frozen"] = False
    s["provisional"] = None          # restarting before an admit means the patient left (discharged or LWBS): their holds are released
    s["score_hist"] = []
    s["f_approve"] = False
    _apply_example(s["example_pick"])


# ------------------------------------------------------------------ tab 1: Simulate Patient Scenario (doctor / ED side)
def simulate_patient():
    _sim_init()
    s = st.session_state
    frozen = s["frozen"]
    pid = f"SIM-{s['submitted_n']:03d}" if frozen else f"SIM-{s['submitted_n'] + 1:03d}"
    st.markdown(LIVE_CSS, unsafe_allow_html=True)
    left, right = st.columns([3, 1.25], gap="large")
    live = right.container(key="live_sim_frozen" if frozen else "live_sim")
    with left:
        st.markdown(f"### Doctor / ED side · patient {pid}")
        st.caption("You are the ED doctor with a new patient. Write your note, record each result as it comes back, then decide whether to admit and submit the bed request. "
                   "The bed manager picks it up under Simulate Bed Assignments.")
        st.select_slider("Patient arrives at", options=ARRIVE_OPTIONS, key="f_arrive", disabled=frozen)
        st.caption("The day starts at 8:00 AM with the demo patients already in the ED. Your patient joins at this time: triage about 10 minutes after arrival, orders at 30, results at 75, "
                   "and the doctor decides at 120 minutes. The Bed Board shows the effect.")
        # ---- step 1
        step_title(1, "Type a patient ED case note", "Write a note in plain, natural language as a doctor would, and see how the AI translates the note into data.")
        c1, c2 = st.columns([1, 1.1], gap="large")
        with c1:
            st.selectbox("Example cases (the demo patients are here too)", list(EXAMPLES), key="example_pick", on_change=load_example, disabled=frozen)
            st.text_area("Case note (free text, include the last assessed vitals)", key="f_text", height=170, disabled=frozen)
            with st.expander("How do I write this? Template with placeholders"):
                st.code(TEMPLATE_TEXT, language=None)
                st.caption("Everything in square brackets is a value to fill in. Assume the note contains everything the model needs; this tab is a baseline of the data model, not a test of the AI's reading.")
            st.button("Submit to AI Queue", type="primary", on_click=do_queue, disabled=frozen or bool(s.get("provisional")), key="sim_queue")
            if s.get("provisional") and not frozen:
                st.caption(f"{pid} is on the Bed Board as a likely patient (no admit decision yet). Edit the case to see the score and holds move, "
                           "then submit the bed request, or start a new patient scenario if they leave.")
            info = s["parse_info"]
            if info and "error" in info:
                st.error(f"The AI reader is not available ({info['error']}). Fill in the form on the right by hand; every step after this still works.")
            elif info:
                st.caption(f"Read by {info['model']} ({info['source']}).")
                for w_ in info["warnings"]:
                    st.warning(w_)
        with c2:
            with st.container(border=True):
                st.markdown("##### AI autofill (edit anything)")
                a, b = st.columns(2)
                a.selectbox("Age band", C.AGE_BANDS, key="f_age", disabled=frozen)
                b.selectbox("ESI level", [1, 2, 3, 4, 5], key="f_esi", disabled=frozen)
                st.selectbox("Chief complaint", I.COMPLAINTS, key="f_complaint", disabled=frozen)
                v = st.columns(3)
                v[0].number_input("Heart rate (worst)", 20.0, 250.0, key="f_hr", disabled=frozen)
                v[1].number_input("SpO2 % (lowest)", 50.0, 100.0, key="f_spo2", disabled=frozen)
                v[2].number_input("Systolic BP (lowest)", 40.0, 260.0, key="f_sbp", disabled=frozen)
                v2 = st.columns(3)
                v2[0].number_input("Resp. rate (worst)", 4.0, 60.0, key="f_rr", disabled=frozen)
                v2[1].number_input("Temp °C (highest)", 30.0, 43.0, key="f_temp", disabled=frozen)
                m = st.columns(2)
                m[0].number_input("Minutes since arrival", 0.0, 4320.0, key="f_arr", disabled=frozen)
                m[1].number_input("Minutes since last vitals check", 0.0, 4320.0, key="f_last", disabled=frozen)
                if info and "evidence" in info and info["evidence"]:
                    st.caption("Where the AI found each value: " + "; ".join(f"{k}: “{p}”" for k, p in info["evidence"].items()))

        # ---- step 2
        step_title(2, "Fill in order results", "Add each order for a test and its result, and see how each new piece of information changes the admit score on the right.")
        edited = st.data_editor(s["f_orders"], key=f"orders_editor_{s['orders_ver']}", num_rows="dynamic", hide_index=True, width="stretch", disabled=frozen,
                                column_config={"name": st.column_config.SelectboxColumn("Order", options=I.ORDERS, required=True),
                                               "result": st.column_config.SelectboxColumn("Result received", options=RESULT_STATES, required=True, default="pending"),
                                               "notes": st.column_config.TextColumn("Notes")})
        s["orders_table"] = edited
        case, assumed = E.complete_case(current_case())
        sc = E.score_case(case, W)
        if not frozen and (not s["score_hist"] or abs(s["score_hist"][-1] - sc["p"]) > 0.0005):
            s["score_hist"].append(sc["p"])
        if s.get("provisional") and not frozen:     # keep the board in step with the case as it is edited
            arr_ = pd.Timestamp(f"{C.DEMO_DATE} {s['f_arrive']}")
            steps_, _dec = sim_timeline(case, arr_)
            s["provisional"]["sim"] = {"patient_id": pid, "arrival": arr_, "steps": steps_, "decision_time": None, "level": s["f_level"],
                                       "isolation": s["f_iso_choice"] == "Yes", "complaint": case["complaint_category"], "esi": int(case["esi"])}
        if assumed:
            st.caption("Not stated, so typical values were used for the score: " + ", ".join(assumed))

        # ---- step 3
        step_title(3, "Admit decision", "Decide whether the patient should be admitted, with the bed type and isolation request.")
        with st.container(border=True):
            st.markdown("##### Confirm patient admission")
            d1, d2 = st.columns(2)
            d1.selectbox("Level of care", list(C.LEVEL_OF_CARE_SHARE), key="f_level", format_func=lambda x: x.replace("_", "-"), disabled=frozen)
            d2.radio("Isolation?", ["No", "Yes"], key="f_iso_choice", horizontal=True, disabled=frozen)
            st.checkbox("Yes. I approve this patient's admit.", key="f_approve", disabled=frozen)
            go = st.button("Submit Bed Request", type="primary", disabled=frozen or not s["f_approve"], key="sim_submit")
        if go:
            c_ = {**case, "admit_decided": True, "requested_level_of_care": s["f_level"], "isolation_required": s["f_iso_choice"] == "Yes"}
            arr = pd.Timestamp(f"{C.DEMO_DATE} {s['f_arrive']}")
            steps, dec_t = sim_timeline(case, arr)
            sim = {"patient_id": pid, "arrival": arr, "steps": steps, "decision_time": dec_t, "level": s["f_level"], "isolation": s["f_iso_choice"] == "Yes",
                   "complaint": case["complaint_category"], "esi": int(case["esi"])}
            s["queue"].append({"patient_id": pid, "time": pd.Timestamp.now().strftime("%H:%M:%S"), "case": c_, "orders": current_case()["orders"],
                               "p": sc["p"], "band": sc["band"], "top3": sc["top3"], "pending": int(sc["features"]["pending_results_so_far"]),
                               "status": "queued", "bed": None, "sim": sim})
            s["submitted_n"] += 1
            s["provisional"] = None
            s["frozen"] = True
            st.rerun()
        if frozen:
            st.success(f"{pid} arrives at {s['f_arrive']} and is in the bed queue. The bed manager can match it under Simulate Bed Assignments, and the Bed Board shows it from {s['f_arrive']}. The score on the right is frozen until you start a new patient scenario.")
        st.button("New patient scenario", on_click=sim_restart, key="sim_restart")

    with live:
        st.markdown("#### Admit score so far")
        st.markdown(f"<div style='font-size:2.8em;font-weight:700;line-height:1.1;color:{BAND_COLOR[sc['band']]}'>{sc['p']:.0%}</div>"
                    + badge(f"{sc['band']} admit", BAND_COLOR[sc["band"]]), unsafe_allow_html=True)
        for r in sc["top3"]:
            st.caption(f"{'▲' if r['push'] >= 0 else '▼'} {r['feature']} ({r['push']:+.2f})")
        h = s["score_hist"]
        if len(h) > 1:
            st.line_chart(pd.DataFrame({"Admit score": h}, index=range(1, len(h) + 1)), height=160)
            st.caption("Each point is a change you made to the case.")
        with st.expander("How the score is calculated"):
            show_score(sc)
        if frozen:
            st.caption("Frozen at submission.")


# ------------------------------------------------------------------ tab 2: Simulate Bed Assignments (nurse / bed management side)
def simulate_beds():
    _sim_init()
    s = st.session_state
    st.markdown("### Nurse / bed management side")
    st.caption("You are the bed manager. Patients the doctor has submitted wait in the queue. Pick one, review the best-matching beds, and confirm. "
               "The assignment then shows on the Bed Board.")
    ents = queue_entries()
    step_title(1, "Beds in queue")
    st.caption("Patients with a bed show where it came from. Click a patient with no bed yet to review the ranked beds and assign one yourself.")
    rows = [{"Patient": e["patient_id"], "Admit score": f"{e['p']:.0%}", "Complaint": str(e["case"]["complaint_category"]).replace("_", " "),
             "Bed type": str(e["case"]["requested_level_of_care"]).replace("_", "-"), "Isolation": "yes" if e["case"]["isolation_required"] else "no",
             "Arrives": f"{e['arrival']:%H:%M}", "Doctor decides": f"{e['decision']:%H:%M}",
             "Status": (lambda b: f"assigned to {b[0]} at {b[1]['time']} ({_how(b[1])})" if b else "queued, no bed yet")(bed_of(e["patient_id"]))} for e in ents]
    sel = st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key=f"bq_table_{s.get('bq_ver', 0)}")
    if not any(s_["Status"].startswith("queued") for s_ in rows):
        st.success("Every queued patient has a bed. See the Bed Board for the assignments.")
        return
    tws = twin_story()
    if tws:
        st.caption(f"Fairness twins {tws[0]['Patient']} (commercial) and {tws[1]['Patient']} (uninsured) are the same clinical case with the same {tws[0]['Admit score']} score and the same matching rule. "
                   "Any difference in their wait is bed supply, not payer: the first twin took the one ICU bed that was ready, so the second had to wait for the next one. "
                   "<a href='?page=fairness' target='_blank'>Open the Fairness check ↗</a>", unsafe_allow_html=True)
    picked = sel.selection.rows[0] if sel.selection.rows else None
    if picked is None:
        st.info("Click a patient marked \"queued, no bed yet\" to match them.")
        return
    item = ents[picked]
    pick = item["patient_id"]
    if bed_of(pick):
        st.info(f"{pick} already has a bed ({bed_of(pick)[0]}). Click a queued patient to match.")
        return
    tdec = item["decision"]
    board, mine, n_other = free_board(tdec, item["patient_id"])
    st.caption(f"Bed board as of {tdec:%H:%M}, when the doctor decides. {n_other} beds are held for other patients at that time and are left out. "
               + (f"Early prep has already held {mine['bed_id']} for this patient since {pd.Timestamp(mine['created_time']):%H:%M}." if mine else "Early prep had not held a bed for this patient."))
    mt = E.match_beds(item["case"], board, now=tdec)
    pat = mt["patient"]
    st.write(f"Needs **{pat['requested_level_of_care'].replace('_', '-')}**" + (" with **isolation**" if pat["isolation_required"] else "")
             + (f", preferred unit **{pat['preferred_unit']}**" if pat["preferred_unit"] else "")
             + f". {mt['n_feasible']} of {mt['n_beds']} beds can take this patient now or soon (beds already assigned in this simulation are excluded).")
    if not mt["ranked"]:
        st.error("No bed passes the rules" + (" (every isolation-capable bed that could serve this patient is occupied or held)" if pat["isolation_required"] else "") + ". Escalate, or override below.")
        alt = BR_feasible_ignoring_isolation(pat, board)
        if alt:
            st.markdown("##### Override: assign a bed outside the rules")
            st.caption("This is a human override and needs a written reason. It is logged with your role." + (" The patient needs isolation and these beds are not isolation-capable." if pat["isolation_required"] else ""))
            oc = st.selectbox("Bed", alt, key=f"bq_alt_{pick}")
            who = st.radio("Decided by", list(ROLES), format_func=ROLES.get, horizontal=True, key="bq_who")
            reason = st.text_input("Reason (required)", key=f"bq_altreason_{pick}")
            if st.button("Override and assign", type="primary", disabled=not reason.strip(), key="bq_altgo"):
                E.record_decision(item["patient_id"], [], oc, who, reason, {})
                s["assignments"][oc] = {"patient_id": item["patient_id"], "role": ROLES[who], "time": f"{tdec + pd.Timedelta(minutes=5):%H:%M}", "level": pat["requested_level_of_care"],
                                        "overridden": True, "assign_time": tdec + pd.Timedelta(minutes=5)}
                if item["item"]:
                    item["item"]["status"], item["item"]["bed"] = "assigned", oc
                s["bq_ver"] = s.get("bq_ver", 0) + 1
                st.rerun()
        return
    bed_cards(mt, {})
    ids = [r["bed_id"] for r in mt["ranked"]]
    step_title(2, "Next steps for the selected bed")
    chosen = st.radio("Selected bed", ids, horizontal=True, key=f"bq_bed_{pick}", format_func=lambda b: f"#{ids.index(b) + 1} {b}")
    r = next(x for x in mt["ranked"] if x["bed_id"] == chosen)
    blockers = E.annotate_blockers([r], board, W["notes"], use_llm=USE_LLM)
    with st.container(border=True):
        st.markdown(f"##### #{ids.index(chosen) + 1} {chosen}")
        bl = blockers.get(chosen)
        if bl and bl["findings"]:
            st.write("Discharge blockers to clear before this bed is ready:")
            for f in bl["findings"]:
                st.write(f"• **{f['blocker_type'].replace('_', ' ')}** → {f['owner_role'].replace('_', ' ')} ({f['confidence']:.0%})")
            with st.expander("Note and evidence"):
                st.markdown(highlight(bl["note_text"], [f["evidence_phrase"] for f in bl["findings"]]), unsafe_allow_html=True)
        elif bl:
            st.write("No blockers in the latest note. The patient in this bed is waiting on a discharge with nothing flagged.")
        else:
            st.write("No discharge blockers: the bed is " + ("ready now." if r["status"] == "ready" else f"being cleaned, ready in about {r['minutes_until_ready']} minutes."))
    who = st.radio("Decided by", list(ROLES), format_func=ROLES.get, horizontal=True, key="bq_who")
    override = chosen != ids[0]
    reason = st.text_input("Reason (required when you do not take the AI's top pick)", key=f"bq_reason_{pick}") if override else ""
    if st.button("Confirm Bed Match", type="primary", disabled=override and not reason.strip(), key="bq_confirm"):
        res = E.record_decision(item["patient_id"], mt["ranked"], chosen, who, reason, blockers)
        s["assignments"][chosen] = {"patient_id": item["patient_id"], "role": ROLES[who], "time": f"{tdec + pd.Timedelta(minutes=5):%H:%M}", "level": pat["requested_level_of_care"],
                                    "overridden": res["overridden"], "assign_time": tdec + pd.Timedelta(minutes=5)}
        if item["item"]:
            item["item"]["status"], item["item"]["bed"] = "assigned", chosen
        s["bq_ver"] = s.get("bq_ver", 0) + 1          # clears the row selection
        st.rerun()


# ------------------------------------------------------------------ tab 2: Demo Scenarios
def _jreset():
    st.session_state["jstep"] = 1


def _jmove(d, n):
    st.session_state["jstep"] = int(min(max(st.session_state.get("jstep", 1) + d, 1), n))


def _case_text(row):
    return (f"{row['age_band']} year old with {str(row['complaint_category']).replace('_', ' ')}, ESI {int(row['esi'])}, "
            f"{row['mins_since_arrival']:.0f} minutes since arrival. Last vitals check {row['mins_since_last_reassessment']:.0f} minutes ago "
            f"(worst so far): HR {row['max_hr_so_far']:.0f}, SpO2 {row['min_spo2_so_far']:.1f}%, BP {row['min_sbp_so_far']:.0f} systolic, "
            f"RR {row['max_rr_so_far']:.0f}, temp {row['max_temp_so_far']:.1f} C.")


def _orders_until(JD, vid, t):
    ev = JD["events"][(JD["events"]["visit_id"] == vid) & (JD["events"]["event_time"] <= t)]
    rows = []
    for _, e in ev[ev["event_type"] == "order"].iterrows():
        name = e["order_name"]
        res = ev[(ev["event_type"] == "result") & (ev["order_name"] == name)]
        no_res = name in C.ORDERS_WITHOUT_RESULT
        state = "n/a" if no_res else ("pending" if not len(res) else ("concerning" if str(res.iloc[0]["result_abnormal"]) == "True" else "reassuring"))
        rows.append({"Order": str(name).replace("_", " "), "Ordered at": f"{e['event_time']:%H:%M}", "Result": state})
    return rows


def demo():
    st.caption("Our seeded patients, replayed one event at a time. At each time you see what the doctor sees (left) and what the bed manager sees (right), using only what was known at that moment.")
    JD = journey_data()
    ids = [v for v in JR.patients(JD) if v in TITLES]
    ids = sorted(ids, key=lambda v: TITLES[v])
    pick = st.selectbox("Patient", ids, format_func=lambda v: TITLES[v], key="jpatient", on_change=_jreset)
    S_ = JR.steps(JD, pick)
    n = len(S_)
    st.session_state.setdefault("jstep", 1)
    st.session_state["jstep"] = min(st.session_state["jstep"], n)
    b1, b2, b3, _sp = st.columns([1, 1, 1, 5])
    b1.button("Back", on_click=_jmove, args=(-1, n), disabled=st.session_state["jstep"] <= 1, key="jback")
    b2.button("Next", on_click=_jmove, args=(1, n), disabled=st.session_state["jstep"] >= n, type="primary", key="jnext")
    b3.button("Restart", on_click=_jreset, key="jrestart")
    st.select_slider("Time", options=list(range(1, n + 1)), key="jstep",
                     format_func=lambda k: f"Time {k} · {S_[k - 1]['event']['event_time']:%H:%M}")
    i = st.session_state["jstep"] - 1
    cur = S_[i]
    t = cur["event"]["event_time"]
    row, idx = cur["snap"], cur["snap_idx"]
    prev_row = S_[i - 1]["snap"] if i > 0 else None
    visit = JD["visits"].loc[pick]
    st.markdown(f"## Time {i + 1} · {t:%H:%M}")
    st.caption(cur["text"])
    left, right = st.columns(2, gap="large")
    with right:
        st.markdown("### Nurse / bed management side")
    with left:
        st.markdown("### Doctor / ED side")
        step_title(1, "The case so far")
        if row is None:
            st.info("The patient has arrived but nothing the model uses is charted yet. The first score comes at triage.")
            with right:
                st.info("Nothing for the bed manager yet.")
            return
        st.write(_case_text(row))
        show_vitals({"hr": row["max_hr_so_far"], "spo2": row["min_spo2_so_far"], "sbp": row["min_sbp_so_far"], "rr": row["max_rr_so_far"], "temp_c": row["max_temp_so_far"]})

        step_title(2, "Orders so far and their results")
        od = _orders_until(JD, pick, t)
        if od:
            st.dataframe(pd.DataFrame(od), hide_index=True, width="stretch")
        else:
            st.caption("No orders yet.")

        step_title(3, "Admit score and how it was calculated")
        sc = sc_from_snapshot(JD, row, idx)
        show_score(sc, prev_p=None if prev_row is None else float(prev_row["p"]))
        if not cur["new_score"]:
            st.caption("No new score at this time: the model rescores at triage, each order and each result.")
        hist = [(s["event"]["event_time"], float(s["snap"]["p"])) for s in S_[: i + 1] if s["snap"] is not None]
        if len(hist) > 1:
            st.line_chart(pd.DataFrame(hist, columns=["time", "Admit score"]).set_index("time"), height=160)
        with st.expander("Effect on the ED as a whole"):
            tot, npat = JR.ed_total(JD, t)
            prev_tot = JR.ed_total(JD, S_[i - 1]["event"]["event_time"])[0] if i > 0 else None
            m1, m2 = st.columns(2)
            m1.metric("Expected admits among patients in the ED now", f"{tot:.1f}", None if prev_tot is None else f"{tot - prev_tot:+.1f} since last time", delta_color="off")
            m2.metric("Patients still awaiting a decision", f"{npat}")
            sp = JR.split(tot)
            st.write("Beds those expected admits would need: " + ", ".join(f"{k.replace('_', '-')} {v:.1f}" for k, v in sp.items()))
            st.caption("This total also moves when other patients arrive or are decided, so not every change comes from this patient.")

        step_title(4, "The doctor's admit decision")
        decided = cur["event"]["event_type"] == "admit_decision" or (pd.notna(visit["disposition_time"]) and visit["disposition_time"] <= t and bool(visit["admitted"]))
        case = {"complaint_category": row["complaint_category"], "esi": int(row["esi"]), "admit_decided": bool(decided),
                "requested_level_of_care": visit["requested_level_of_care"] if decided else None, "isolation_required": bool(visit["isolation_required"]) if decided else False}
        if decided:
            st.success(f"Doctor decided to admit at {visit['disposition_time']:%H:%M}.")
            show_request(case)
        else:
            st.info("Not decided yet. Until then the system only forecasts.")

    with right:
        step_title(5, "Bed matching")
        fb, fh, n_oth = free_board(t, pick)
        st.caption(f"Bed board as of {t:%H:%M} (the same board as the Bed Board tab). {n_oth} beds are held for other patients and are left out.")
        matching_step(case, float(row["p"]), int(row["pending_results_so_far"]), fb, t, hold=fh)


# ------------------------------------------------------------------ tab 3: Bed Board (with prepare-ahead merged in)
DAY_TIMES = [pd.Timestamp(C.DEMO_DATE) + pd.Timedelta(hours=8) + pd.Timedelta(minutes=15 * i) for i in range(33)]      # 08:00 to 16:00
ARRIVE_OPTIONS = [f"{t:%H:%M}" for t in DAY_TIMES if t <= pd.Timestamp(C.DEMO_DATE) + pd.Timedelta(hours=14)]


@st.cache_resource(show_spinner="Reading the discharge notes for every bed that is waiting on a discharge...")
def prep_world():
    pri = priors()
    man = _json.load(open(f"{C.DATA_DIR}/demo_manifest.json"))
    try:
        blk = E.prep_blockers(W["beds"], W["notes"], use_llm=True)
    except Exception:
        blk = E.prep_blockers(W["beds"], W["notes"], use_llm=False)
    return {"pri": pri, "man": man, "blk": blk}


def demo_only_ids():
    """The scripted Demo Scenarios patients (A, B, C). They are replayed on that tab only and never claim beds on the board or in the queue."""
    return frozenset(v["visit_id"] for v in prep_world()["man"].get("scripts", {}).values())


@st.cache_resource(show_spinner="Replaying the day...")
def _day_states(sims_json, assigns_json):
    PW = prep_world()
    sims = _json.loads(sims_json)
    assigns = _json.loads(assigns_json)
    return EP.replay_day(DAY_TIMES, journey_data(), W["beds"], PW["blk"], PW["pri"], PW["man"].get("forecast_at_now"), now=E.NOW, sims=sims, assigns=assigns, exclude=demo_only_ids())


def sims_in_session():
    out = [x["sim"] for x in st.session_state.get("queue", []) if x.get("sim")]
    pv = st.session_state.get("provisional")
    if pv and pv.get("sim"):
        out.append(pv["sim"])        # a patient read by the AI but not yet admitted: shows on the board as a likely patient
    return out


def day_states():
    """Every 15 minutes from 08:00: the bed board and early-prep plan, including the patients the user added and the beds they assigned."""
    return _day_states(_json.dumps(sims_in_session(), default=str, sort_keys=True),
                       _json.dumps(st.session_state.get("assignments", {}), default=str, sort_keys=True))


def bed_of(patient_id):
    """(bed, assignment) once a patient has a bed anywhere in the day: assigned by the bed manager, or taken automatically when the held bed freed up."""
    return next(((b, a) for b, a in day_states()[-1]["assigns"].items() if a["patient_id"] == patient_id), None)


def BR_feasible_ignoring_isolation(pat, board):
    """Beds that could serve the patient's level of care and are free or freeing soon, whether or not they meet the isolation need (for a human override)."""
    import bed_rules as BR
    return BR.feasible_beds({**pat, "isolation_required": False}, board)


def twin_story():
    """The two clinically identical demo patients (one commercial, one uninsured): same score, same rule, and what bed supply gave each of them."""
    tw = prep_world()["man"].get("payer_twins") or {}
    ents = {e["patient_id"]: e for e in queue_entries()}
    rows = []
    for payer in ("commercial", "uninsured"):
        e = ents.get(tw.get(payer))
        if not e:
            return None
        got = bed_of(e["patient_id"])
        rdy = W["beds"].set_index("bed_id")["expected_ready_time"].get(got[0]) if got else None
        rows.append({"Patient": e["patient_id"], "Payer": payer, "Admit score": f"{e['p']:.0%}", "Doctor decides": f"{e['decision']:%H:%M}",
                     "Bed taken": (f"{got[0]} at {got[1]['time']}" if got else "not assigned yet"),
                     "That bed was ready at": (f"{pd.Timestamp(rdy):%H:%M}" if rdy is not None and pd.notna(rdy) else ""),
                     "Minutes from decision to bed": (round((pd.Timestamp(got[1]["assign_time"]) - e["decision"]).total_seconds() / 60) if got else None)})
    return rows


def queue_entries():
    """Everyone in the bed queue as one list: the admitted demo patients (V-...) and the patients the user submitted (SIM-...). Each can be matched by the bed manager."""
    J = journey_data()
    v, sn = J["visits"], J["snap"]
    out = []
    for vid, r in v[(v["disposition"] == "admit") & v["disposition_time"].notna() & ~v.index.isin(list(demo_only_ids()))].iterrows():
        sv = sn[(sn["visit_id"] == vid) & (sn["snapshot_time"] <= r["disposition_time"])]
        out.append({"patient_id": vid, "p": float(sv.iloc[-1]["p"]) if len(sv) else 0.0, "arrival": pd.Timestamp(r["arrival_time"]), "decision": pd.Timestamp(r["disposition_time"]),
                    "case": {"complaint_category": r["complaint_category"], "esi": int(r["esi"]), "requested_level_of_care": r["requested_level_of_care"],
                             "isolation_required": bool(r["isolation_required"]), "admit_decided": True}, "item": None})
    for x in st.session_state.get("queue", []):
        out.append({"patient_id": x["patient_id"], "p": x["p"], "arrival": pd.Timestamp(x["sim"]["arrival"]), "decision": pd.Timestamp(x["sim"]["decision_time"]),
                    "case": x["case"], "item": x})
    out.sort(key=lambda e: e["decision"])
    return out


def _how(a):
    return "early prep" if "Early prep" in str(a.get("role", "")) else str(a.get("role", "")).lower()


def state_at(t):
    i = int(min(max((pd.Timestamp(t) - DAY_TIMES[0]) // pd.Timedelta(minutes=15), 0), len(DAY_TIMES) - 1))
    return day_states()[i]


def free_board(t, visit_id):
    """The board at t for matching one patient: bed state at t, beds already assigned in the simulation taken, and beds early prep holds for OTHER patients left out.
    Also returns the hold early prep placed for this patient, if any."""
    res = state_at(t)
    b, _ = EP.board_with_assigns(W["beds"], t, res["assigns"])          # simulation assignments and demo-patient assignments alike
    others = {h["bed_id"] for h in res["holds"] if h["visit_id"] != visit_id}
    mine = next((h for h in res["holds"] if h["visit_id"] == visit_id), None)
    return b[~b["bed_id"].isin(others)], mine, len(others)


UNIT_ORDER = [("icu", "ICU"), ("cardiac", "Cardiac (telemetry)"), ("pulmonary", "Pulmonary"), ("neuro", "Neuro"), ("general", "General (med-surg)")]
STATUS_RGB = {"ready": (134, 205, 160), "cleaning": (147, 197, 232), "discharge": (246, 190, 110), "occupied": (236, 140, 140), "blocked": (150, 156, 168)}
HOLD_STYLE = {"soft": ("#7C3AED", "dotted"), "hard": ("#7C3AED", "dashed"), "decided": ("#7C3AED", "solid")}      # pooled holds (no patient yet) are not drawn
HOLD_NAME = {"soft": "Soft hold (40-60%)", "hard": "Expedited hold (60%+)", "decided": "Admitted: held for an approved patient", "pooled": "Pooled hold (no patient yet)"}


def _bed_state(b):
    if b["status"] == "occupied" and bool(b["discharge_pending"]):
        return "discharge"
    return b["status"]


BED_CSS = """<style>
[class*="st-key-unit_"]{flex-direction:row !important;flex-wrap:wrap !important;gap:8px !important;margin-bottom:10px}
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-bed_"]){width:92px !important;flex:0 0 92px !important}
[class*="st-key-bed_"]{position:relative;width:92px !important;flex:0 0 92px !important;gap:0 !important}
[class*="st-key-bed_"] [data-testid="stElementContainer"]{width:92px !important}
[class*="st-key-bed_"] button{width:92px;height:62px;min-height:62px;padding:5px 7px;border-radius:10px;color:#1F2937;box-shadow:0 1px 2px rgba(0,0,0,.06);
 justify-content:flex-start;align-items:flex-start;text-align:left;overflow:hidden}
[class*="st-key-bed_"] button > div, [class*="st-key-bed_"] button [data-testid="stMarkdownContainer"]{width:100%;justify-content:flex-start;text-align:left}
[class*="st-key-bed_"]{height:62px}
[class*="st-key-bed_"] button p{font-size:12px;line-height:1.3;text-align:left;margin:0}
[class*="st-key-bed_"] button:hover{filter:brightness(.97);color:#1F2937}
[class*="st-key-bed_"] [data-testid="stMarkdown"]{height:0;overflow:visible}
.bc{display:none;position:absolute;top:68px;left:0;z-index:9999;width:290px;background:#fff;color:#1F2937;border:1px solid rgba(0,0,0,.14);
 border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,.16);padding:10px 12px;font-size:12.5px;line-height:1.45;text-align:left}
.bc.rt{left:auto;right:0}
.bc.mid{left:50%;transform:translateX(-50%)}
[class*="st-key-bed_"]:hover .bc{display:block}
.bc .h{font-weight:700;font-size:13.5px;margin-bottom:1px}
.bc .sub{color:#6B7280;font-size:11.5px;margin-bottom:6px}
.bc .row{margin-top:5px;padding-top:5px;border-top:1px solid rgba(0,0,0,.07)}
.bc .k{display:block;color:#6B7280;font-size:10.5px;letter-spacing:.04em;text-transform:uppercase}
.bc ul{margin:2px 0 0 16px;padding:0}
</style>"""


def _bed_box(b, tnow, hold, blk, tasks, assign=None, col=0):
    esc = html.escape
    key = _bed_state(b)          # a bed assigned in the simulation is already "occupied" on this board, same as one from the data
    mins = None
    if key in ("cleaning", "discharge") and pd.notna(b["expected_ready_time"]):
        mins = max(0, round((b["expected_ready_time"] - tnow).total_seconds() / 60))
    assign = None
    alpha = 1.0
    named = hold if hold and hold["visit_id"] else None      # who the bed is held for (also an approved patient waiting for the bed to finish cleaning)
    r, g, bl = STATUS_RGB[key]
    border = "1px solid rgba(0,0,0,.10)"
    tag = ""
    if assign:
        border = "2px solid #6D5BD0"
        tag = f"<div style='font-size:10px;color:#4C3FA8;font-weight:600'>{esc(assign['patient_id'])}</div>"
    if hold and hold["kind"] in HOLD_STYLE:      # soft, expedited and admitted holds get an outline; a pooled hold has no patient yet
        c_, sty = HOLD_STYLE[hold["kind"]]
        border = f"{3 if hold['kind'] == 'decided' else 2}px {sty} {c_}"

    ready_t = f"{b['expected_ready_time']:%H:%M}" if pd.notna(b["expected_ready_time"]) else "unknown"
    sub = dict(UNIT_ORDER).get(b["unit"], b["unit"]) + (f" · {b['bed_type'].replace('_', ' ')}" if b["bed_type"].replace("_", " ") not in dict(UNIT_ORDER).get(b["unit"], "").lower() and b["bed_type"] not in (b["unit"], "med_surg") else "") + (" · isolation-capable" if bool(b["isolation_capable"]) else "")
    rows = []   # (label, html)
    vid = b["current_visit_id"] if isinstance(b["current_visit_id"], str) else None
    # 1) what the bed is doing now
    if key == "occupied":
        rows.append(("Right now", f"Occupied by patient <b>{esc(vid or 'unknown')}</b>. No discharge planned."))
    elif key == "discharge":
        rows.append(("Right now", f"Occupied by patient <b>{esc(vid or 'unknown')}</b>, <b>pending discharge</b>. Expected free around <b>{ready_t}</b> (~{mins} min)."))
        fnd = (blk or {}).get("findings") or []
        if fnd:
            li = "".join(f"<li>{esc(f['blocker_type'].replace('_', ' '))} <span style='color:#6B7280'>(owner: {esc(f['owner_role'].replace('_', ' '))})</span></li>" for f in fnd)
            rows.append(("Discharge held up by", f"<ul>{li}</ul>"))
        else:
            rows.append(("Discharge held up by", "Nothing flagged in the latest notes."))
    elif key == "cleaning":
        steps = [k["task_type"].replace("_", " ") for k in (tasks or []) if k["task_type"] not in ("none",)]
        rows.append(("Right now", f"Being cleaned. Expected ready around <b>{ready_t}</b> (~{mins} min)."))
        rows.append(("Waiting on", esc(" → ".join(steps)) if steps else "Housekeeping turnover (no other steps tracked)."))
    elif key == "ready":
        rows.append(("Right now", "Clean and <b>ready now</b>. Nobody is in it."))
    elif key == "blocked":
        rows.append(("Right now", "Out of service (blocked). Not available for matching."))
    elif key == "assigned":
        rows.append(("Right now", f"Free in the data, but <b>assigned in the simulation</b>."))
    # 2) who is coming next
    if assign:
        rows.append(("Next", f"Assigned to incoming patient <b>{esc(assign['patient_id'])}</b> ({esc(assign['level'].replace('_', '-'))}) by {esc(assign['role'])} at {esc(assign['time'])}" + (" (override)" if assign.get("overridden") else "") + "."))
    elif hold:
        who = hold["visit_id"]
        if hold["kind"] == "pooled":
            nxt = "Held in the <b>pool</b> for the next likely admit. No patient named yet."
        elif hold["kind"] == "decided":
            nxt = f"Held for incoming patient <b>{esc(str(who))}</b> (admit decided, score {hold['p_latest']:.0%})." if hold["p_latest"] is not None else f"Held for incoming patient <b>{esc(str(who))}</b> (admit decided)."
        else:
            nxt = f"Prepared ahead for likely patient <b>{esc(str(who))}</b> (admit score {hold['p_latest']:.0%}, not decided yet)."
        extra = f"<br>Assumed {esc(hold['level_assumed'].replace('_', '-'))} bed ({hold['level_confidence']:.0%} sure)."
        if hold["kind"] != "pooled":
            extra += f"<br>Needed by <b>{hold['needed_by_time']:%H:%M}</b>; realistic ready {hold['realistic_ready_time']:%H:%M}" + (" (<b>misses the earliest plan</b>)" if hold["critical"] else "") + "."
        rows.append(("Next", nxt + extra))
        if key in ("occupied", "discharge", "cleaning") and tasks:
            rows.append(("Early-prep steps", esc(" → ".join(f"{k['task_type'].replace('_', ' ')} {k['start_by_time']:%H:%M}-{k['due_time']:%H:%M}" for k in tasks))))
    else:
        rows.append(("Next", "No incoming patient yet." if key != "blocked" else "None."))

    status_txt = {"ready": "Ready", "cleaning": "Being cleaned", "discharge": "Discharge pending", "occupied": "Occupied", "blocked": "Blocked", "assigned": "Assigned"}[key]
    card = (f"<div class='bc'><div class='h'>{esc(b['bed_id'])} · {status_txt}</div><div class='sub'>{esc(sub)}</div>"
            + "".join(f"<div class='row'><span class='k'>{lab}</span>{body}</div>" for lab, body in rows) + "</div>")
    label = {"ready": "ready", "cleaning": f"clean {mins}m", "discharge": f"leaves ~{mins}m", "occupied": "occupied", "blocked": "blocked", "assigned": "assigned"}[key]
    pos = " rt" if col % 10 >= 7 else (" mid" if col % 10 >= 4 else "")
    card = card.replace("<div class='bc'>", f"<div class='bc{pos}'>", 1)
    bid = b["bed_id"]
    label_md = f"**{bid}**  \n{label}" + (f"  \n→ {named['visit_id']}" if named else "")
    css = (f".st-key-bed_{bid} button, .st-key-bed_{bid} button:hover, .st-key-bed_{bid} button:focus{{background:rgba({r},{g},{bl},{alpha:.2f}) !important;border:{border} !important}}")
    return label_md, css, card


def _legend():
    chips = []
    for name, key in (("Ready now", "ready"), ("Being cleaned", "cleaning"), ("Occupied, discharge pending", "discharge"), ("Occupied", "occupied"), ("Blocked", "blocked")):
        r, g, b = STATUS_RGB[key]
        chips.append(f"<span style='display:inline-flex;align-items:center;margin-right:14px;font-size:13px'><span style='width:16px;height:16px;border-radius:4px;"
                     f"background:rgba({r},{g},{b},1);border:1px solid rgba(0,0,0,.10);margin-right:5px'></span>{name}</span>")
    for k, (col, sty) in HOLD_STYLE.items():
        chips.append(f"<span style='display:inline-flex;align-items:center;margin-right:14px;font-size:13px'><span style='width:16px;height:16px;border-radius:4px;"
                     f"border:{3 if k == 'decided' else 2}px {sty} {col};margin-right:5px;box-sizing:border-box'></span>{HOLD_NAME[k]}</span>")
    st.markdown("<div style='line-height:2'>" + "".join(chips) + "</div>", unsafe_allow_html=True)


def _pick_bed(bed_id):
    st.session_state["board_inspect"] = bed_id
    st.toast(f"Showing {bed_id} under Inspect a bed, below the board.")


def board():
    PW = prep_world()
    st.caption("Every inpatient bed at a glance, from 8:00 AM. Move the time to watch beds free up and early-prep holds appear. Hover a bed to see who is in it, what its discharge or cleaning is waiting on, and who is coming next. "
               "Click a bed to open it under Inspect a bed, below the board. A purple outline means the bed is being prepared ahead for the patient whose code is printed on it (dotted = soft hold, dashed = expedited hold, thick solid = the patient is admitted and takes the bed the moment it is ready). "
               "Patients you add under Simulate Patient Scenario join this day at the time you choose, and beds you assign are shown as occupied, like any other bed.")
    states = day_states()
    idx = st.select_slider("Time of day", options=list(range(len(DAY_TIMES))), value=0, format_func=lambda i: f"{DAY_TIMES[i]:%H:%M}", key="prep_t")
    res, tnow = states[idx], DAY_TIMES[idx]
    beds = res["board"]
    hold_by_bed = {h["bed_id"]: h for h in res["holds"]}
    tasks_by_hold = {}
    for k in res["tasks"]:
        tasks_by_hold.setdefault(k["hold_id"], []).append(k)
    assigned = {bed: a for bed, a in res["assigns"].items() if pd.Timestamp(a["assign_time"]) <= tnow}
    for x in st.session_state.get("queue", []):
        sm = x.get("sim")
        if not sm:
            continue
        dec, arr = pd.Timestamp(sm["decision_time"]), pd.Timestamp(sm["arrival"])
        line = f"**{x['patient_id']}** arrives {arr:%H:%M}, doctor decides {dec:%H:%M}"
        asg = next((bd for bd, a in res["assigns"].items() if a["patient_id"] == x["patient_id"]), None)
        hd = next((h for h in res["holds"] if h["visit_id"] == x["patient_id"]), None)
        if tnow < arr:
            line += ". Not in the ED yet."
        elif asg and pd.Timestamp(res["assigns"][asg]["assign_time"]) <= tnow:
            line += f". Assigned to {asg}."
        elif hd:
            line += f". Bed {hd['bed_id']} held since {pd.Timestamp(hd['created_time']):%H:%M} ({hd['kind']} hold)."
        else:
            line += ". No bed held yet."
        st.caption(line)
    named = [h for h in res["holds"] if h["visit_id"] and h["kind"] in ("soft", "hard")]
    c = st.columns(3)
    c[0].metric("Ready beds", int((beds["status"] == "ready").sum()))
    c[1].metric("Prepared ahead (likely patients)", len(named))
    c[2].metric("Tasks started early", len(res["tasks"]))
    bn = res["banner"]
    if bn.get("forecast_used"):
        msg = (f"Next {bn['horizon_hours']} hours: about {bn['forecast_admits']:.1f} new admits expected. After beds already claimed, "
               + ", ".join(f"{k.replace('_', '-')} short by {max(0, v['shortfall']):.1f}" for k, v in bn["levels"].items()) + ".")
        if bn["shortfall_beds"] < C.PREP_POOL_MIN_SHORTFALL:      # the yellow shortfall warning is intentionally not shown
            st.success(msg)
    _legend()
    st.markdown(BED_CSS, unsafe_allow_html=True)
    for unit, title in UNIT_ORDER:
        ub = beds[beds["unit"] == unit]
        if ub.empty:
            continue
        st.markdown(f"#### {title}")
        parts = [_bed_box(r, tnow, hold_by_bed.get(r["bed_id"]), PW["blk"].get(r["bed_id"]),
                          tasks_by_hold.get(hold_by_bed[r["bed_id"]]["hold_id"]) if r["bed_id"] in hold_by_bed else None,
                          assigned.get(r["bed_id"]), col=i) for i, (_, r) in enumerate(ub.iterrows())]
        sel_bed = st.session_state.get("board_inspect")
        css = "".join(p[1] for p in parts) + (f".st-key-bed_{sel_bed} button{{box-shadow:0 0 0 3px #111827 !important}}" if sel_bed else "")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        with st.container(key=f"unit_{unit}"):
            for (_, r), (lab, _c, card) in zip(ub.iterrows(), parts):
                with st.container(key=f"bed_{r['bed_id']}"):
                    st.button(lab, key=f"bedbtn_{r['bed_id']}", on_click=_pick_bed, args=(r["bed_id"],))
                    st.markdown(card, unsafe_allow_html=True)

    # ---- inspect one bed (hover alternative, and where the task chain lives)
    st.markdown("### Inspect a bed")
    held = [h["bed_id"] for h in res["holds"]]
    all_ids = list(beds["bed_id"])                       # the same options at every time of day, so the widget keeps a valid value as holds come and go
    if st.session_state.get("board_inspect") not in all_ids:
        st.session_state["board_inspect"] = all_ids[0]   # also clears a stale value such as a label saved by an older version of the app
    pick = st.selectbox("Bed", all_ids, key="board_inspect", format_func=lambda x: x + ("  (held)" if x in held else ""))
    b = beds.set_index("bed_id").loc[pick]
    h = hold_by_bed.get(pick)
    key = _bed_state(b)
    st.markdown(f"##### {pick}")
    st.caption(f"{dict(UNIT_ORDER).get(b['unit'], b['unit'])} · {b['bed_type'].replace('_', ' ')}" + (" · isolation-capable" if bool(b["isolation_capable"]) else "") + f" · {STATUS_NAME.get(b['status'], b['status'])}"
               + (" (discharge pending)" if key == "discharge" else ""))
    if pick in assigned:
        a_ = assigned[pick]
        st.write(f"Assigned to **{a_['patient_id']}** ({a_['level'].replace('_', '-')}) by {a_['role']} at {a_['time']}" + (" (override)" if a_["overridden"] else "") + ".")
    occ = PW["blk"].get(pick)
    if occ:
        st.write("Open blockers: " + (", ".join(f"**{f['blocker_type'].replace('_', ' ')}** ({f['owner_role'].replace('_', ' ')})" for f in occ["findings"]) or "none in the latest note"))
        with st.expander(f"Latest note ({occ['note_time']:%H:%M})"):
            st.markdown(highlight(occ["note_text"], [f["evidence_phrase"] for f in occ["findings"]]), unsafe_allow_html=True)
    if h:
        st.write(f"{HOLD_NAME[h['kind']]} for **{who_label(h['visit_id'])}**" + ("" if h["p_latest"] is None else f", admit score {h['p_latest']:.0%}")
                 + f". Assumed {h['level_assumed'].replace('_', '-')} ({h['level_confidence']:.0%} of similar admits).")
        mine = tasks_by_hold.get(h["hold_id"], [])
        if mine:
            stt = EP.task_state(mine, tnow, st.session_state.get("prep_done", {}).get(h["hold_id"], []))
            ROLE_LABEL = {"inpatient_physician": "Inpatient physician", "environmental_services": "Environmental services (cleaning)", "inpatient_nurse": "Inpatient nurse",
                          "case_management": "Case management", "pharmacy": "Pharmacy", "PT/OT": "PT/OT", "social_work": "Social work"}
            st.dataframe(pd.DataFrame([{"Task": k["task_type"].replace("_", " "), "Owner": ROLE_LABEL.get(k["owner_role"], k["owner_role"]), "Start by": f"{k['start_by_time']:%H:%M}",
                                        "Due": f"{k['due_time']:%H:%M}", "Minutes": int(k["duration_min"]), "Status": stt[k["task_key"]]} for k in mine]),
                         hide_index=True, width="stretch")
            ready_t = max(k["due_time"] for k in mine if k["task_type"] == "bed_cleaning")
            st.write(f"Bed ready (end of cleaning): **{ready_t:%H:%M}**" + ("  · misses the earliest plan, but is still sooner than starting after the decision" if h["critical"] else "") + ".")
            open_tasks = [k for k in mine if stt[k["task_key"]] in ("open", "in_progress")]
            if open_tasks:
                tk = st.selectbox("Mark a task done now (to see the chain move)", [k["task_key"] for k in open_tasks],
                                  format_func=lambda x: next(k["task_type"].replace("_", " ") for k in open_tasks if k["task_key"] == x), key="prep_done_pick")
                if st.button("Mark done at this moment", key="prep_done_btn"):
                    moved, new_ready = EP.mark_done(mine, tk, tnow)
                    st.success(f"Done at {tnow:%H:%M}. The tasks after it re-time; the bed is now expected ready at {new_ready:%H:%M} (was {ready_t:%H:%M}).")

    with st.expander("All holds and early bed requests (tables)"):
        ratio = []
        for hh in res["holds"]:
            margin = None if hh["kind"] in ("decided", "pooled") else round((hh["needed_by_time"] - hh["realistic_ready_time"]).total_seconds() / 60)
            ratio.append({"Patient": who_label(hh["visit_id"]), "Type": HOLD_NAME[hh["kind"]], "Admit score": None if hh["p_latest"] is None else round(hh["p_latest"], 2),
                          "Bed assumed": f"{hh['level_assumed'].replace('_', '-')} ({hh['level_confidence']:.0%})", "Bed": hh["bed_id"],
                          "Supply": {1: "ready now", 2: "being cleaned", 3: "discharge-pending"}[hh["tier"]], "Realistic ready": f"{hh['realistic_ready_time']:%H:%M}",
                          "Needed by": "" if hh["kind"] == "pooled" else f"{hh['needed_by_time']:%H:%M}", "Margin (min)": margin})
        if ratio:
            st.dataframe(pd.DataFrame(ratio), hide_index=True, width="stretch")
        else:
            st.caption("Nobody is flagged yet at this moment.")
        if res["bed_requests"]:
            st.dataframe(pd.DataFrame([{"Patient": who_label(r["visit_id"]), "Bed type": r["level"].replace("_", "-"), "Isolation": "yes" if r["isolation"] else "no",
                                        "For bed": r["bed_id"], "Status": "send now (60%+)" if r["sent_early"] else "drafted only (40-60%)"} for r in res["bed_requests"]]),
                         hide_index=True, width="stretch")
        st.caption("Needed by = the earliest the doctor will likely decide (25th percentile for this many pending results). Realistic ready uses published task durations, "
                   "not the unit's stated ready time.")
        who = st.selectbox("Confirming as", list(ROLES), format_func=lambda r: ROLES[r], key="prep_role")
        if st.button("Confirm these prompts (writes holds, tasks and an audit record)", type="primary", key="prep_confirm"):
            done = E.record_prep(res, who, now=tnow)
            st.success(f"Recorded {done['holds']} holds, {done['tasks']} tasks and {done['audit']} audit rows (Audit trail page). No bed was changed and nobody was discharged.")

    with st.expander("How much time could this save across all admits? (scenarios, not measurements)"):
        sc = C.SCENARIO_MEDIAN_BOARDING_HOURS
        st.dataframe(pd.DataFrame([
            {"Scenario": "Today (brief)", "Median boarding (h)": sc["baseline"], "Change": "-"},
            {"Scenario": "Conservative: half the prompts acted on", "Median boarding (h)": sc["conservative"], "Change": f"{(sc['conservative'] / sc['baseline'] - 1):.0%}"},
            {"Scenario": "Base case (headline)", "Median boarding (h)": sc["base"], "Change": f"{(sc['base'] / sc['baseline'] - 1):.0%}"},
            {"Scenario": "Optimistic: every prompt acted on", "Median boarding (h)": sc["optimistic"], "Change": f"{(sc['optimistic'] / sc['baseline'] - 1):.0%}"}]),
            hide_index=True, width="stretch")
        st.caption("Each test-split admit's simulated boarding time, minus a head start only where the model had flagged the patient at 60% before the decision. "
                   "The saving comes from starting existing work sooner; nobody is assumed to work faster.")


# ------------------------------------------------------------------ separate pages (opened from the sidebar in a new browser tab)
LABELS = {"mean_admit_score": "Mean admit score", "calibration_gap": "Calibration gap (score minus actual admit rate)",
          "false_negative_rate": "Missed admits (last score 'unlikely')", "median_boarding_hours": "Median boarding hours",
          "lwbs_rate": "Left without being seen", "median_door_to_provider_min": "Door-to-provider minutes"}
MODEL_METRICS = ["mean_admit_score", "calibration_gap", "false_negative_rate"]


@st.cache_data(show_spinner="Scoring the held-out test visits...")
def _fairness():
    import os as _os
    p = f"{C.DATA_DIR}/fairness_report.csv"
    if _os.path.exists(p):
        return pd.read_csv(p), None
    df, overall, _ = FR.run()
    return df, overall


def page_extractor():
    st.write("It reads each note and must name the blocker, quote the exact words, and say who owns it. We score it against a hidden answer key, "
             "including two traps: notes with no blocker, and notes that mention something already fixed.")
    which = st.radio("Notes", ["36 template notes", "12 differently worded notes (fairer test)"], horizontal=True)
    use_rules = st.checkbox("Score the keyword-rules baseline instead of the AI", value=not (st_["key_present"] or st_["cache_only"]))
    if st.button("Run the test"):
        if "12" in which:
            nn, ll = pd.read_csv(f"{C.DATA_DIR}/notes_unseen.csv"), pd.read_csv(f"{C.DATA_DIR}/notes_unseen_labels.csv")
            mm = nn[["note_id", "style", "has_resolved_distractor"]]
        else:
            nn, ll, mm = (pd.read_csv(f"{C.DATA_DIR}/{f}.csv") for f in ("inpatient_notes", "note_labels", "note_meta"))
        fn = EB.rules_extract if use_rules else EB.extract
        res, bar = {}, st.progress(0.0)
        try:
            for i, (_, r) in enumerate(nn.iterrows()):
                res[r["note_id"]] = fn(r["note_text"])["findings"]; bar.progress((i + 1) / len(nn))
            out, det = EB.score(nn, ll, res, mm)
            q = st.columns(4)
            q[0].metric("Precision", f"{out['precision']:.0%}"); q[1].metric("Recall", f"{out['recall']:.0%}")
            q[2].metric("Invented a blocker", f"{out['false_alarm_notes']} of {out['no_blocker_notes']} clean notes")
            q[3].metric("Flagged a fixed item", f"{out.get('flagged_a_resolved_item', 0)} of {out.get('notes_with_resolved_item', 0)}")
            st.dataframe(det, hide_index=True, width="stretch")
        except L.LLMUnavailable as e:
            st.error(f"The AI reader is not available: {e}")



def page_audit():
    import os
    for name in ("audit_log", "bed_matches", "bed_holds", "tasks"):
        p = f"{C.DATA_DIR}/{name}.csv"
        st.markdown(f"### {name}")
        if os.path.exists(p):
            st.dataframe(pd.read_csv(p).tail(30), hide_index=True, width="stretch")
        else:
            st.caption("Nothing yet. Confirm or override a bed in Simulate Scenario.")



def page_fairness():
    st.caption("Payer is never an input to the model. We still measure whether results differ by payer, on visits the model never trained on. "
               f"A gap is flagged when a group differs from overall by more than {C.FAIRNESS_TOLERANCE:.0%} "
               "(calibration gap: more than 2 points). FINDING = the 95% interval excludes zero and the group has 30+ cases.")
    df, overall = _fairness()
    st.markdown("### Does the model score payer groups differently?")
    show = df.assign(metric=df["metric"].map(LABELS))
    for title, mets in (("Model behaviour", MODEL_METRICS), ("How the ED runs (observed data, not model output)",
                                                           ["median_boarding_hours", "lwbs_rate", "median_door_to_provider_min"])):
        st.markdown(f"#### {title}")
        t = show[df["metric"].isin(mets)][["metric", "group", "group_value", "overall_value", "difference", "ci_low", "ci_high", "n", "status"]]
        st.dataframe(t.rename(columns={"metric": "Metric", "group": "Payer group", "group_value": "Group", "overall_value": "Overall",
                                       "difference": "Difference", "ci_low": "95% low", "ci_high": "95% high", "n": "Cases", "status": "Status"}),
                     hide_index=True, width="stretch")
    st.markdown("### Same patient, different payer")
    tw = FR.twin_check()
    if tw:
        st.write(f"Demo twins (identical clinically; {tw['commercial']} commercial vs {tw['uninsured']} uninsured) scored "
                 f"**{tw['scores'][0]:.4f}** and **{tw['scores'][1]:.4f}**. Payer cannot change a score because it is not an input.")
    ts = twin_story()
    if ts:
        st.markdown("#### What each twin got")
        st.dataframe(pd.DataFrame(ts), hide_index=True, width="stretch")
        st.write("**Same score, same rule, different bed supply.** Both twins went through the identical matching rule. The first to be decided took the best bed that was free; "
                 "the second was decided five minutes later, when that bed was gone, so the next ICU bed (ICU-05, ready at 12:00) set the wait. Swap the payers and the same gap appears for whoever decides second. "
                 "Payer is not an input to the score or to the matching rule.")
    st.info("Not measured yet: override rates by payer. Needs real decisions in the audit trail; the demo has too few to test.")
    st.download_button("Download fairness_report.csv", df.to_csv(index=False), "fairness_report.csv")



PAGES = {"extractor": ("Extractor test", page_extractor), "audit": ("Audit trail", page_audit), "fairness": ("Fairness check", page_fairness)}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Boarding Command Center")
    st.caption("Demo day: Wed Nov 4, from 8:00 AM. All data is synthetic. Do not type real patient information.")
    if st_["key_present"]:
        st.success(f"AI reader: {st_['provider']} / {st_['model']}")
    elif st_["cache_only"]:
        st.info("AI reader: replaying saved answers (offline mode)")
    else:
        st.warning(f"No AI key found. Set {C.LLM_PROVIDERS[st_['provider']]['key_env']} to turn on the AI reader. "
                   "You can still fill the form by hand; blocker reading falls back to simple keyword rules.")
    st.caption("The AI proposes. A named person decides. Every override needs a written reason.")
    st.markdown("<div style='height:22vh'></div>", unsafe_allow_html=True)
    st.divider()
    st.caption("More pages (open in a new tab)")
    for k, (name, _fn) in PAGES.items():
        st.markdown(f"<a href='?page={k}' target='_blank' style='text-decoration:none'>{name} ↗</a>", unsafe_allow_html=True)

# ------------------------------------------------------------------ route
_page = st.query_params.get("page")
if _page in PAGES:
    st.title(PAGES[_page][0])
    st.markdown("<a href='.' target='_self'>← Back to the main app</a>", unsafe_allow_html=True)
    PAGES[_page][1]()
else:
    tab_sim, tab_beds, tab_demo, tab_board = st.tabs(["Simulate Patient Scenario", "Simulate Bed Assignments", "Demo Scenarios", "Bed Board"])
    with tab_sim:
        simulate_patient()
    with tab_beds:
        simulate_beds()
    with tab_demo:
        demo()
    with tab_board:
        board()