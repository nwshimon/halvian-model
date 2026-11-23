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
def _orders_frame(orders):
    """Orders table with fixed column types (an empty table with untyped columns crashes st.data_editor)."""
    df = pd.DataFrame(orders or [], columns=["name", "resulted", "abnormal"])
    return df.astype({"name": str, "resulted": bool, "abnormal": bool})


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
    s = st.session_state
    s.update(f)
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
    s["f_admit"] = c["admit_decided"]
    if c["requested_level_of_care"]:
        s["f_level"] = c["requested_level_of_care"]
    s["f_iso"] = c["isolation_required"]
    s["f_orders"] = _orders_frame(c["orders"])
    s["orders_ver"] += 1
    s["decision"] = None


def current_case():
    s = st.session_state
    orders = [{"name": r["name"], "resulted": bool(r["resulted"]), "abnormal": bool(r["abnormal"]) and bool(r["resulted"])}
              for _, r in s["orders_table"].iterrows() if r["name"] in I.ORDERS]
    return {"age_band": s["f_age"], "complaint_category": s["f_complaint"], "esi": s["f_esi"],
            "vitals": {"hr": s["f_hr"], "spo2": s["f_spo2"], "sbp": s["f_sbp"], "rr": s["f_rr"], "temp_c": s["f_temp"]},
            "orders": orders, "minutes_since_arrival": s["f_arr"], "minutes_since_last_reassessment": s["f_last"],
            "admit_decided": s["f_admit"], "requested_level_of_care": s["f_level"] if s["f_admit"] else None, "isolation_required": s["f_iso"]}


def highlight(text, phrases):
    out = html.escape(text)
    for p in phrases:
        out = out.replace(html.escape(p), f"<mark style='background:#FDE68A;padding:1px 3px;border-radius:3px'>{html.escape(p)}</mark>", 1)
    return out


def badge(text, color):
    return f"<span style='background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:0.85em;font-weight:600'>{html.escape(text)}</span>"



# ------------------------------------------------------------------ shared pieces
import json as _json
import early_prep as EP

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
    for col, (k, label, fmt) in zip(cols, (("hr", "Heart rate (worst)", "{:.0f}"), ("spo2", "SpO2 % (lowest)", "{:.1f}"), ("sbp", "Systolic BP (lowest)", "{:.0f}"),
                                           ("rr", "Resp. rate (worst)", "{:.0f}"), ("temp_c", "Temp °C (highest)", "{:.1f}"))):
        col.metric(label, fmt.format(vit[k]))
        if flags.get(k):
            col.caption(":red[⚠ outside the normal range]")


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
            st.markdown(f"**#{r['rank']} {r['bed_id']}**" + ("  ·  best fit" if r["rank"] == 1 else ""))
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


def early_prep_preview(case, p, pending, board, now):
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
    mt = E.match_beds(guess, board, now=now, top_n=1)
    st.success(f"No decision yet, but the score ({p:.0%}) is in the {TIER_TEXT[tier]}")
    st.write(f"Assumed bed type: **{lv.replace('_', '-')}** ({share:.0%} of similar admits)" + (", **isolation**" if iso else "")
             + f". Earliest likely decision: about **{needed:.0f} min** from now (25th percentile, with {int(pending)} result(s) pending).")
    if mt["ranked"]:
        r = mt["ranked"][0]
        when = "ready now" if r["status"] == "ready" else f"ready in ~{r['minutes_until_ready']} min"
        st.write(f"Bed it would hold: **{r['bed_id']}** ({r['unit']}, {r['status']}, {when}). A person confirms before anything is sent.")
    else:
        st.warning("No suitable bed can be prepared in time.")


def matching_step(case, p, pending, board, now, record_key=None, case_id=None):
    """Step 5. Decided -> ranked beds (and, for the typed case, the confirm/override form). Not decided -> early-prep preview."""
    if not case["admit_decided"]:
        early_prep_preview(case, p, pending, board, now)
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
    st.markdown("**A named person confirms or overrides**")
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


# ------------------------------------------------------------------ tab 1: Simulate Scenario
def simulate():
    st.caption("Type or pick a case and follow it left to right through the data model, one step at a time. "
               "Assume the note is written in a standard form that contains everything the model needs; the AI only copies it into the form.")
    # ---- step 1
    step_title(1, "The typed case and what the AI filled in")
    c1, c2 = st.columns([3, 1])
    with c2:
        st.selectbox("Example cases", list(EXAMPLES), key="example_pick", on_change=load_example)
    with c1:
        st.text_area("Case (as a clinician would write it, including the last assessed vitals)", key="f_text", height=120)
    st.button("Read the case with AI", type="primary", on_click=do_parse)
    info = st.session_state["parse_info"]
    if info and "error" in info:
        st.error(f"The AI reader is not available ({info['error']}). Fill in the form by hand; every step after this still works.")
    elif info:
        st.caption(f"Read by {info['model']} ({info['source']}). Check the form and correct anything that is wrong.")
        for w_ in info["warnings"]:
            st.warning(w_)
    st.markdown("**Form the AI filled** (edit anything)")
    a, b, c = st.columns(3)
    a.selectbox("Age band", C.AGE_BANDS, key="f_age")
    b.selectbox("Chief complaint", I.COMPLAINTS, key="f_complaint")
    c.selectbox("ESI triage level", [1, 2, 3, 4, 5], key="f_esi")
    v = st.columns(5)
    v[0].number_input("Heart rate (worst)", 20.0, 250.0, key="f_hr")
    v[1].number_input("SpO2 % (lowest)", 50.0, 100.0, key="f_spo2")
    v[2].number_input("Systolic BP (lowest)", 40.0, 260.0, key="f_sbp")
    v[3].number_input("Resp. rate (worst)", 4.0, 60.0, key="f_rr")
    v[4].number_input("Temp °C (highest)", 30.0, 43.0, key="f_temp")
    m = st.columns(2)
    m[0].number_input("Minutes since arrival", 0.0, 4320.0, key="f_arr")
    m[1].number_input("Minutes since last vitals check", 0.0, 4320.0, key="f_last")
    if info and "evidence" in info and info["evidence"]:
        st.caption("Where the AI found each value: " + "; ".join(f"{k}: “{p}”" for k, p in info["evidence"].items()))

    # ---- step 2
    step_title(2, "Orders so far and their results", "Add an order, then tick “Result back” and “Abnormal” as results arrive. The score in step 3 updates.")
    edited = st.data_editor(st.session_state["f_orders"], key=f"orders_editor_{st.session_state['orders_ver']}", num_rows="dynamic", hide_index=True, width="stretch",
                            column_config={"name": st.column_config.SelectboxColumn("Order", options=I.ORDERS, required=True),
                                           "resulted": st.column_config.CheckboxColumn("Result back"),
                                           "abnormal": st.column_config.CheckboxColumn("Abnormal")})
    st.session_state["orders_table"] = edited
    case, assumed = E.complete_case(current_case())

    # ---- step 3
    step_title(3, "Admit score and how it was calculated")
    sc = E.score_case(case, W)
    show_score(sc, assumed)

    # ---- step 4
    step_title(4, "The doctor's admit decision", "Tick this when the doctor decides. It turns the score into a fact and opens a bed request.")
    d = st.columns(3)
    d[0].checkbox("Doctor has decided to admit", key="f_admit")
    d[1].selectbox("Level of care needed", list(C.LEVEL_OF_CARE_SHARE), key="f_level", disabled=not st.session_state["f_admit"])
    d[2].checkbox("Isolation required", key="f_iso", disabled=not st.session_state["f_admit"])
    case, assumed = E.complete_case(current_case())
    if case["admit_decided"]:
        show_request(case)
    else:
        st.info("Not decided yet. Until then the system only forecasts.")

    # ---- step 5
    step_title(5, "Bed matching")
    pending = int(sc["features"]["pending_results_so_far"])
    matching_step(case, sc["p"], pending, W["beds"], E.NOW, record_key="sim", case_id="CASE-" + pd.Timestamp.now().strftime("%H%M%S"))


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
        rows.append({"Order": str(name).replace("_", " "), "Ordered at": f"{e['event_time']:%H:%M}",
                     "Result back": "n/a" if no_res else ("yes" if len(res) else "pending"),
                     "Abnormal": "" if (no_res or not len(res)) else ("yes" if str(res.iloc[0]["result_abnormal"]) == "True" else "no")})
    return rows


def demo():
    st.caption("Our seeded patients, replayed one event at a time. At each time the same five steps as Simulate Scenario are shown, using only what was known at that moment.")
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
    st.markdown(f"### Time {i + 1} · {t:%H:%M}   {cur['text']}")

    step_title(1, "The case so far")
    if row is None:
        st.info("The patient has arrived but nothing the model uses is charted yet. The first score comes at triage.")
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

    step_title(5, "Bed matching")
    tb = max(t, E.NOW)
    st.caption(f"Bed board as of {tb:%H:%M}: beds that finish cleaning by then count as ready; other changes after the demo's {C.DEMO_NOW_TIME} are not simulated.")
    matching_step(case, float(row["p"]), int(row["pending_results_so_far"]), EP.board_at(W["beds"], tb), tb)


# ------------------------------------------------------------------ tab 3: Bed Board (with prepare-ahead merged in)
@st.cache_resource(show_spinner="Reading the discharge notes for every bed that is waiting on a discharge...")
def prep_world():
    pri = priors()
    man = _json.load(open(f"{C.DATA_DIR}/demo_manifest.json"))
    try:
        blk = E.prep_blockers(W["beds"], W["notes"], use_llm=True)
    except Exception:
        blk = E.prep_blockers(W["beds"], W["notes"], use_llm=False)
    times = ["10:30", "10:46", "11:30", "12:05", "12:28"]
    stamps = [pd.Timestamp(f"{C.DEMO_DATE} {t}") for t in times]
    states = EP.replay(stamps, journey_data(), W["beds"], blk, pri, man.get("forecast_at_now"), now=E.NOW)
    return {"pri": pri, "man": man, "blk": blk, "times": times, "stamps": stamps, "states": states}


UNIT_ORDER = [("icu", "ICU"), ("cardiac", "Cardiac (telemetry)"), ("pulmonary", "Pulmonary"), ("neuro", "Neuro"), ("general", "General (med-surg)")]
STATUS_RGB = {"ready": (34, 197, 94), "cleaning": (14, 165, 233), "discharge": (245, 158, 11), "occupied": (156, 163, 175), "blocked": (100, 116, 139)}
HOLD_STYLE = {"soft": ("#7C3AED", "dashed"), "hard": ("#7C3AED", "solid"), "decided": ("#1D4ED8", "solid"), "pooled": ("#0F766E", "dotted")}
HOLD_NAME = {"soft": "Soft hold (40-60%)", "hard": "Expedited hold (60%+)", "decided": "Held for a decided patient", "pooled": "Pooled hold (no patient yet)"}


def _bed_state(b):
    if b["status"] == "occupied" and bool(b["discharge_pending"]):
        return "discharge"
    return b["status"]


def _bed_box(b, tnow, hold, blk, tasks):
    key = _bed_state(b)
    mins = None
    if key in ("cleaning", "discharge") and pd.notna(b["expected_ready_time"]):
        mins = max(0, round((b["expected_ready_time"] - tnow).total_seconds() / 60))
    alpha = 1.0 if key in ("ready", "occupied", "blocked") else max(0.4, 1.0 - min(mins or 240, 240) / 240 * 0.6)
    r, g, bl = STATUS_RGB[key]
    border = "1px solid rgba(0,0,0,.25)"
    tag = ""
    if hold:
        col, sty = HOLD_STYLE[hold["kind"]]
        border = f"3px {sty} {col}"
        tag = f"<div style='font-size:10px;color:{col};font-weight:700'>→ {html.escape(str(hold['visit_id'] or 'pooled'))[-5:] if hold['visit_id'] else 'pool'}</div>"
    tip = [f"{b['bed_id']} · {b['unit']} · {b['bed_type']}" + (" · isolation-capable" if bool(b["isolation_capable"]) else ""),
           {"ready": "Ready now", "cleaning": f"Being cleaned, ready in ~{mins} min", "discharge": f"Occupied, discharge pending, expected ready in ~{mins} min",
            "occupied": "Occupied, no discharge planned", "blocked": "Blocked (out of service)"}[key]]
    if isinstance(b["current_visit_id"], str) and key in ("occupied", "discharge"):
        tip.append(f"Patient {b['current_visit_id']}")
    if blk and blk.get("findings"):
        tip.append("Open blockers: " + ", ".join(f"{f['blocker_type'].replace('_', ' ')} ({f['owner_role'].replace('_', ' ')})" for f in blk["findings"]))
    if hold:
        tip.append(f"{HOLD_NAME[hold['kind']]}: {who_label(hold['visit_id'])}"
                   + ("" if hold["p_latest"] is None else f", admit score {hold['p_latest']:.0%}")
                   + f", assumed {hold['level_assumed'].replace('_', '-')} ({hold['level_confidence']:.0%})")
        if hold["kind"] != "pooled":
            tip.append(f"Needed by {hold['needed_by_time']:%H:%M}; realistic ready {hold['realistic_ready_time']:%H:%M}" + ("; misses earliest plan" if hold["critical"] else ""))
        if tasks:
            tip.append("Tasks: " + " → ".join(f"{k['task_type'].replace('_', ' ')} {k['start_by_time']:%H:%M}-{k['due_time']:%H:%M}" for k in tasks))
    label = {"ready": "ready", "cleaning": f"clean {mins}m", "discharge": f"leaves ~{mins}m", "occupied": "occupied", "blocked": "blocked"}[key]
    return (f"<div title=\"{html.escape(chr(10).join(tip), quote=True)}\" style='width:92px;height:62px;border-radius:8px;border:{border};"
            f"background:rgba({r},{g},{bl},{alpha:.2f});padding:4px 6px;font-size:12px;line-height:1.25;color:#111;box-sizing:border-box;overflow:hidden'>"
            f"<b>{html.escape(b['bed_id'])}</b><br>{label}{tag}</div>")


def _legend():
    chips = []
    for name, key in (("Ready now", "ready"), ("Being cleaned", "cleaning"), ("Occupied, discharge pending", "discharge"), ("Occupied", "occupied"), ("Blocked", "blocked")):
        r, g, b = STATUS_RGB[key]
        chips.append(f"<span style='display:inline-flex;align-items:center;margin-right:14px;font-size:13px'><span style='width:16px;height:16px;border-radius:4px;"
                     f"background:rgba({r},{g},{b},1);border:1px solid rgba(0,0,0,.25);margin-right:5px'></span>{name}</span>")
    chips.append("<span style='font-size:13px;margin-right:14px'>Lighter tint = longer until ready</span>")
    for k, (col, sty) in HOLD_STYLE.items():
        chips.append(f"<span style='display:inline-flex;align-items:center;margin-right:14px;font-size:13px'><span style='width:16px;height:16px;border-radius:4px;"
                     f"border:3px {sty} {col};margin-right:5px;box-sizing:border-box'></span>{HOLD_NAME[k]}</span>")
    st.markdown("<div style='line-height:2'>" + "".join(chips) + "</div>", unsafe_allow_html=True)


def board():
    PW = prep_world()
    st.caption("Every inpatient bed at a glance. Hover a bed for details (or pick one below). Purple and teal outlines show beds the system would prepare ahead "
               "for patients who are likely to be admitted but not yet decided.")
    idx = st.select_slider("Moment in the demo day", options=list(range(len(PW["times"]))), value=0,
                           format_func=lambda i: PW["times"][i] + ("  (demo 'now')" if i == 0 else ""), key="prep_t")
    res, tnow = PW["states"][idx], PW["stamps"][idx]
    beds = EP.board_at(W["beds"], tnow)
    hold_by_bed = {h["bed_id"]: h for h in res["holds"]}
    tasks_by_hold = {}
    for k in res["tasks"]:
        tasks_by_hold.setdefault(k["hold_id"], []).append(k)
    named = [h for h in res["holds"] if h["visit_id"] and h["kind"] in ("soft", "hard")]
    c = st.columns(5)
    c[0].metric("Ready beds", int((beds["status"] == "ready").sum()))
    c[1].metric("Held for decided patients", sum(h["kind"] == "decided" for h in res["holds"]))
    c[2].metric("Prepared ahead (likely patients)", len(named))
    c[3].metric("Pooled holds", len(res["pool"]))
    c[4].metric("Tasks started early", len(res["tasks"]))
    bn = res["banner"]
    if bn.get("forecast_used"):
        (st.warning if bn["shortfall_beds"] >= C.PREP_POOL_MIN_SHORTFALL else st.success)(
            f"Next {bn['horizon_hours']} hours: about {bn['forecast_admits']:.1f} new admits expected. After beds already claimed, "
            + ", ".join(f"{k.replace('_', '-')} short by {max(0, v['shortfall']):.1f}" for k, v in bn["levels"].items()) + ".")
    _legend()
    for unit, title in UNIT_ORDER:
        ub = beds[beds["unit"] == unit]
        if ub.empty:
            continue
        boxes = "".join(_bed_box(r, tnow, hold_by_bed.get(r["bed_id"]), PW["blk"].get(r["bed_id"]),
                                 tasks_by_hold.get(hold_by_bed[r["bed_id"]]["hold_id"]) if r["bed_id"] in hold_by_bed else None) for _, r in ub.iterrows())
        st.markdown(f"**{title}**")
        st.markdown(f"<div style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px'>{boxes}</div>", unsafe_allow_html=True)

    # ---- inspect one bed (hover alternative, and where the task chain lives)
    st.markdown("**Inspect a bed**")
    held = [h["bed_id"] for h in res["holds"]]
    pick = st.selectbox("Bed", held + [b for b in beds["bed_id"] if b not in held], key="board_inspect",
                        format_func=lambda x: x + ("  (held)" if x in held else ""))
    b = beds.set_index("bed_id").loc[pick]
    h = hold_by_bed.get(pick)
    key = _bed_state(b)
    st.write(f"**{pick}** · {b['unit']} · {b['bed_type']}" + (" · isolation-capable" if bool(b["isolation_capable"]) else "") + f" · {STATUS_NAME.get(b['status'], b['status'])}"
             + (" (discharge pending)" if key == "discharge" else ""))
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
        st.dataframe(pd.DataFrame(ratio), hide_index=True, width="stretch") if ratio else st.caption("Nobody is flagged yet at this moment.")
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
        st.markdown(f"**{name}**")
        if os.path.exists(p):
            st.dataframe(pd.read_csv(p).tail(30), hide_index=True, width="stretch")
        else:
            st.caption("Nothing yet. Confirm or override a bed in Simulate Scenario.")



def page_fairness():
    st.caption("Payer is never an input to the model. We still measure whether results differ by payer, on visits the model never trained on. "
               f"A gap is flagged when a group differs from overall by more than {C.FAIRNESS_TOLERANCE:.0%} "
               "(calibration gap: more than 2 points). FINDING = the 95% interval excludes zero and the group has 30+ cases.")
    df, overall = _fairness()
    st.markdown("**Does the model score payer groups differently?**")
    show = df.assign(metric=df["metric"].map(LABELS))
    for title, mets in (("Model behaviour", MODEL_METRICS), ("How the ED runs (observed data, not model output)",
                                                           ["median_boarding_hours", "lwbs_rate", "median_door_to_provider_min"])):
        st.markdown(f"**{title}**")
        t = show[df["metric"].isin(mets)][["metric", "group", "group_value", "overall_value", "difference", "ci_low", "ci_high", "n", "status"]]
        st.dataframe(t.rename(columns={"metric": "Metric", "group": "Payer group", "group_value": "Group", "overall_value": "Overall",
                                       "difference": "Difference", "ci_low": "95% low", "ci_high": "95% high", "n": "Cases", "status": "Status"}),
                     hide_index=True, width="stretch")
    st.markdown("**Same patient, different payer**")
    tw = FR.twin_check()
    if tw:
        st.write(f"Demo twins (identical clinically; {tw['commercial']} commercial vs {tw['uninsured']} uninsured) scored "
                 f"**{tw['scores'][0]:.4f}** and **{tw['scores'][1]:.4f}**. Payer cannot change a score because it is not an input.")
    st.info("Not measured yet: override rates by payer. Needs real decisions in the audit trail; the demo has too few to test.")
    st.download_button("Download fairness_report.csv", df.to_csv(index=False), "fairness_report.csv")



PAGES = {"extractor": ("Extractor test", page_extractor), "audit": ("Audit trail", page_audit), "fairness": ("Fairness check", page_fairness)}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Boarding Command Center")
    st.caption(f"Demo board as of Wed Nov 4, {C.DEMO_NOW_TIME}. All data is synthetic. Do not type real patient information.")
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
    tab_sim, tab_demo, tab_board = st.tabs(["Simulate Scenario", "Demo Scenarios", "Bed Board"])
    with tab_sim:
        simulate()
    with tab_demo:
        demo()
    with tab_board:
        board()
