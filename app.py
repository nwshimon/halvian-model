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
                                 "Doctor has decided to admit to telemetry.",
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


def _defaults():
    return {"f_text": EXAMPLES["A: looks mild, then isn't"], "f_age": "35-49", "f_complaint": "abdominal_pain", "f_esi": 3,
            "f_hr": 92.0, "f_spo2": 96.0, "f_sbp": 128.0, "f_rr": 18.0, "f_temp": 37.4, "f_arr": 60.0, "f_last": 52.0,
            "f_admit": False, "f_level": "telemetry", "f_iso": False, "orders_ver": 0,
            "f_orders": _orders_frame([{"name": "lab_panel", "resulted": False, "abnormal": False}]),
            "parse_info": None, "decision": None}


for k, v in _defaults().items():
    st.session_state.setdefault(k, v)


def load_example():
    st.session_state["f_text"] = EXAMPLES[st.session_state["example_pick"]]


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


# ------------------------------------------------------------------ sidebar
st_ = L.status()
with st.sidebar:
    st.header("Boarding Command Center")
    st.caption(f"Demo board as of Wed Nov 4, {C.DEMO_NOW_TIME}. All data is synthetic. Do not type real patient information.")
    if st_["key_present"]:
        st.success(f"AI reader: {st_['provider']} / {st_['model']}")
    elif st_["cache_only"]:
        st.info("AI reader: replaying saved answers (offline mode)")
    else:
        st.warning(f"No AI key found. Set {C.LLM_PROVIDERS[st_['provider']]['key_env']} to turn on the AI reader. "
                   "You can still fill the form by hand and score cases; blocker reading falls back to simple keyword rules.")
    st.divider()
    st.caption("The AI proposes. A named person decides. Every override needs a written reason.")

tab_case, tab_journey, tab_board, tab_test, tab_audit, tab_fair = st.tabs(["Case demo", "Patient journey", "Bed board", "Extractor test", "Audit trail", "Fairness"])

# ------------------------------------------------------------------ tab 1: case demo
with tab_case:
    st.subheader("1. Type a case")
    c1, c2 = st.columns([3, 1])
    with c2:
        st.selectbox("Example cases", list(EXAMPLES), key="example_pick", on_change=load_example)
    with c1:
        st.text_area("Case (as a clinician would write it)", key="f_text", height=130)
    st.button("Read the case with AI", type="primary", on_click=do_parse)
    info = st.session_state["parse_info"]
    if info and "error" in info:
        st.error(f"The AI reader is not available ({info['error']}). Fill in the form below by hand; everything after this step still works.")
    elif info:
        st.caption(f"Read by {info['model']} ({info['source']}). Check the form below and correct anything that is wrong.")
        for w_ in info["warnings"]:
            st.warning(w_)

    with st.expander("The form the AI filled (edit anything)", expanded=True):
        a, b, c = st.columns(3)
        a.selectbox("Age band", C.AGE_BANDS, key="f_age")
        b.selectbox("Chief complaint", I.COMPLAINTS, key="f_complaint")
        c.selectbox("ESI triage level", [1, 2, 3, 4, 5], key="f_esi")
        v = st.columns(5)
        v[0].number_input("Heart rate (worst)", 20.0, 250.0, key="f_hr")
        v[1].number_input("SpO2 % (lowest)", 50.0, 100.0, key="f_spo2")
        v[2].number_input("Systolic BP (lowest)", 40.0, 260.0, key="f_sbp")
        v[3].number_input("Resp. rate (worst)", 4.0, 60.0, key="f_rr")
        v[4].number_input("Temp C (highest)", 30.0, 43.0, key="f_temp")
        m = st.columns(2)
        m[0].number_input("Minutes since arrival", 0.0, 4320.0, key="f_arr")
        m[1].number_input("Minutes since last vitals check", 0.0, 4320.0, key="f_last")
        st.caption("Orders so far")
        edited = st.data_editor(st.session_state["f_orders"], key=f"orders_editor_{st.session_state['orders_ver']}", num_rows="dynamic", hide_index=True,
                       column_config={"name": st.column_config.SelectboxColumn("Order", options=I.ORDERS, required=True),
                                      "resulted": st.column_config.CheckboxColumn("Result back"),
                                      "abnormal": st.column_config.CheckboxColumn("Abnormal")})
        st.session_state["orders_table"] = edited
        d = st.columns(3)
        d[0].checkbox("Doctor has decided to admit", key="f_admit")
        d[1].selectbox("Level of care needed", list(C.LEVEL_OF_CARE_SHARE), key="f_level", disabled=not st.session_state["f_admit"])
        d[2].checkbox("Isolation required", key="f_iso")
        if info and "evidence" in info and info["evidence"]:
            st.caption("Where the AI found each value: " + "; ".join(f"{k}: “{p}”" for k, p in info["evidence"].items()))

    case, assumed = E.complete_case(current_case())

    # ---- 2. admit score
    st.subheader("2. Admit score and how it was calculated")
    sc = E.score_case(case, W)
    k1, k2 = st.columns([1, 2])
    with k1:
        st.markdown(f"<div style='font-size:3em;font-weight:700;color:{BAND_COLOR[sc['band']]}'>{sc['p']:.0%}</div>"
                    + badge(f"{sc['band']} admit", BAND_COLOR[sc["band"]]), unsafe_allow_html=True)
        st.caption("Chance this patient is admitted, from what is known right now.")
        st.markdown("**Top reasons**")
        for r in sc["top3"]:
            st.write(f"{'▲' if r['push'] >= 0 else '▼'} {r['feature']} ({r['push']:+.2f})")
    with k2:
        st.caption(f"Starting point for a typical patient: {sc['baseline_logodds']:+.2f} (log-odds). Each row pushes the score up or down. "
                   f"Total {sc['total_logodds']:+.2f} → {sc['p']:.1%}.")
        tab = pd.DataFrame(sc["rows"])
        tab["value"] = tab["value"].map(lambda x: f"{x:.2f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x))
        st.bar_chart(tab.set_index("feature")["push"], horizontal=True, height=300)
        with st.expander("Show the numbers"):
            st.dataframe(tab.rename(columns={"feature": "Factor", "value": "Value used", "push": "Push (log-odds)"}), hide_index=True, width="stretch")
    if assumed:
        st.info("Not stated, so typical values were used: " + ", ".join(assumed))
    st.caption("Payer, race and language are never used by the score.")

    # ---- 3. bed request + matches
    st.subheader("3. Bed matching")
    if not case["admit_decided"]:
        st.info("No admit decision yet, so no bed request. Tick “Doctor has decided to admit” to see matches. "
                "(Before a decision the tool only forecasts demand: this patient counts toward the expected admits, at the probability above.)")
    else:
        mt = E.match_beds(case, W["beds"])
        pat = mt["patient"]
        st.write(f"Needs **{pat['requested_level_of_care']}**" + (" with **isolation**" if pat["isolation_required"] else "")
                 + (f", preferred unit **{pat['preferred_unit']}**" if pat["preferred_unit"] else "")
                 + f". {mt['n_feasible']} of {mt['n_beds']} beds can take this patient now or soon.")
        if not mt["ranked"]:
            st.error("No bed fits. Escalate to the bed manager.")
        else:
            blockers = E.annotate_blockers(mt["ranked"], W["beds"], W["notes"], use_llm=st_["key_present"] or st_["cache_only"])
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

            # ---- 4. decision
            st.subheader("4. Decision")
            if st.session_state["decision"]:
                dcs = st.session_state["decision"]
                st.success(f"{'Override' if dcs['overridden'] else 'Confirmed'} by {dcs['role']} at {dcs['time']}: {dcs['bed']}. "
                           f"Wrote {dcs['matches']} match rows, {dcs['tasks']} tasks, {dcs['audit']} audit entries.")
            ids = [r["bed_id"] for r in mt["ranked"]]
            others = [b for b in W["beds"]["bed_id"] if b not in ids]
            who = st.radio("Decided by", list(ROLES), format_func=ROLES.get, horizontal=True)
            pick = st.radio("Bed", ids + ["(another bed)"], horizontal=True)
            bed_id = st.selectbox("Which bed?", others) if pick == "(another bed)" else pick
            override = bed_id != ids[0]
            reason = st.text_input("Reason (required when you do not take the AI's top pick)") if override else ""
            if st.button(f"{'Override and assign' if override else 'Confirm'} {bed_id}", type="primary", disabled=override and not reason.strip()):
                res = E.record_decision("CASE-" + pd.Timestamp.now().strftime("%H%M%S"), mt["ranked"], bed_id, who, reason, blockers)
                st.session_state["decision"] = {**res, "role": ROLES[who], "bed": bed_id, "time": pd.Timestamp.now().strftime("%H:%M:%S")}
                st.rerun()

# ------------------------------------------------------------------ tab 2: board
with tab_board:
    st.subheader("Inpatient bed board (40-bed demo slice)")
    b = W["beds"].copy()
    b["ready_in_min"] = [0 if s == "ready" else (None if pd.isna(t) else max(0, round((t - E.NOW).total_seconds() / 60))) for s, t in zip(b["status"], b["expected_ready_time"])]
    st.dataframe(b[["bed_id", "unit", "bed_type", "isolation_capable", "status", "discharge_pending", "ready_in_min"]], hide_index=True, width="stretch", height=520)
    st.caption(b["status"].value_counts().to_dict())

# ------------------------------------------------------------------ tab 3: extractor test
with tab_test:
    st.subheader("How good is the blocker reader?")
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

# ------------------------------------------------------------------ tab 4: audit trail
with tab_audit:
    st.subheader("Audit trail")
    import os
    for name in ("audit_log", "bed_matches", "tasks"):
        p = f"{C.DATA_DIR}/{name}.csv"
        st.markdown(f"**{name}**")
        if os.path.exists(p):
            st.dataframe(pd.read_csv(p).tail(30), hide_index=True, width="stretch")
        else:
            st.caption("Nothing yet. Confirm or override a bed on the Case demo tab.")


# ------------------------------------------------------------------ tab 5: fairness
LABELS = {"mean_admit_score": "Mean admit score", "calibration_gap": "Calibration gap (score minus actual admit rate)",
          "false_negative_rate": "Missed admits (last score 'unlikely')", "median_boarding_hours": "Median boarding hours",
          "lwbs_rate": "Left without being seen", "median_door_to_provider_min": "Door-to-provider minutes"}
MODEL_METRICS = ["mean_admit_score", "calibration_gap", "false_negative_rate"]

@st.cache_data(show_spinner="Scoring the held-out test visits...")
def _fairness():
    import os as _os
    p = f"{C.DATA_DIR}/fairness_report.csv"
    if _os.path.exists(p):                      # precomputed by fairness.py; the hosted copy has no full history to recompute from
        return pd.read_csv(p), None
    df, overall, _ = FR.run()
    return df, overall

with tab_fair:
    st.subheader("Fairness check by payer group")
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


# ------------------------------------------------------------------ tab: patient journey
@st.cache_resource(show_spinner="Scoring every step of the demo patients...")
def journey_data():
    return JR.load(W["model"])


TITLES = {}
try:
    import json as _json
    for _k, _v in _json.load(open(f"{C.DATA_DIR}/demo_manifest.json"))["scripts"].items():
        TITLES[_v["visit_id"]] = f"{_k}: {_v['title']}"
except Exception:
    pass


def _jreset():
    st.session_state["jstep"] = 1


def _jmove(d, n):
    st.session_state["jstep"] = int(min(max(st.session_state.get("jstep", 1) + d, 1), n))


with tab_journey:
    st.subheader("Patient journey: watch each new piece of information change the score")
    st.caption("Replays a synthetic patient's ED stay one event at a time. At each step the same trained model rescores the patient using "
               "only what was known at that moment, and we show how that moves the whole ED's expected admissions.")
    JD = journey_data()
    ids = JR.patients(JD)
    ids = sorted(ids, key=lambda v: (v not in TITLES, TITLES.get(v, v)))
    pick = st.selectbox("Patient", ids, format_func=lambda v: TITLES.get(v, f"{v} ({JD['visits'].loc[v].get('complaint_category', '')})"),
                        key="jpatient", on_change=_jreset)
    S_ = JR.steps(JD, pick)
    n = len(S_)
    st.session_state.setdefault("jstep", 1)
    st.session_state["jstep"] = min(st.session_state["jstep"], n)
    b1, b2, b3, _sp = st.columns([1, 1, 1, 5])
    b1.button("Back", on_click=_jmove, args=(-1, n), disabled=st.session_state["jstep"] <= 1, key="jback")
    b2.button("Next", on_click=_jmove, args=(1, n), disabled=st.session_state["jstep"] >= n, type="primary", key="jnext")
    b3.button("Restart", on_click=_jreset, key="jrestart")
    st.slider("Step", 1, max(n, 2), key="jstep") if n > 1 else None
    i = st.session_state["jstep"] - 1
    cur = S_[i]
    t = cur["event"]["event_time"]

    st.markdown(f"### {t:%H:%M}   {cur['text']}")
    if t > pd.Timestamp(E.NOW):
        st.caption(f"Replay: this happens after the demo's {C.DEMO_NOW_TIME} 'now'.")

    row, idx = cur["snap"], cur["snap_idx"]
    prev_row = S_[i - 1]["snap"] if i > 0 else None
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**What the system knows so far**")
        if row is None:
            st.info("Nothing the model uses is known yet. The first score comes at triage.")
        else:
            now_f = JR.facts(row)
            old_f = JR.facts(prev_row) if prev_row is not None else {}
            tab = pd.DataFrame([{"Fact": k, "Value": (f"{v:.1f}" if isinstance(v, float) else str(v)),
                                 "Change": ("new" if k not in old_f else ("changed" if str(old_f[k]) != str(v) else ""))}
                                for k, v in now_f.items()
                                if k not in ("Time since arrival (min)", "Time since last check (min)", "Inpatient occupancy")])
            st.dataframe(tab, hide_index=True, width="stretch", height=380)
    with right:
        if row is None:
            st.metric("Admit score", "no score yet")
        else:
            p = float(row["p"]); prev_p = float(prev_row["p"]) if prev_row is not None else None
            band = str(AM.AdmitModel.category([p])[0])
            st.metric("Admit score", f"{p:.0%}", None if prev_p is None else f"{(p - prev_p) * 100:+.0f} points", delta_color="off")
            st.markdown(badge(band, BAND_COLOR.get(band, "#6B7280")), unsafe_allow_html=True)
            if not cur["new_score"]:
                st.caption("No new score at this step: the model rescores at triage, each order and each result.")
            hist = [(s["event"]["event_time"], float(s["snap"]["p"])) for s in S_[: i + 1] if s["snap"] is not None]
            st.line_chart(pd.DataFrame(hist, columns=["time", "Admit score"]).set_index("time"), height=200)

    if row is not None:
        st.markdown("**Why the score is what it is** (each fact pushes the odds up or down; they add up to the score)")
        cw = JD["contrib"].loc[idx]
        pushes = pd.DataFrame({"push": [float(cw[f]) for f in JR.FIELDS]}, index=[JR.LABEL[f] for f in JR.FIELDS])
        pushes = pushes[pushes["push"].abs() > 0.005].sort_values("push")
        st.bar_chart(pushes, horizontal=True, height=max(180, 28 * len(pushes)))
        top = pushes.reindex(pushes["push"].abs().sort_values(ascending=False).index).head(C.TOP_FACTORS_SHOWN)
        st.write("Top reasons: " + "; ".join(f"{k} ({'+' if v > 0 else ''}{v:.2f})" for k, v in top["push"].items()))

    tot, npat = JR.ed_total(JD, t)
    prev_tot = JR.ed_total(JD, S_[i - 1]["event"]["event_time"])[0] if i > 0 else None
    st.markdown("**Effect on the ED as a whole**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Expected admissions from patients in the ED now", f"{tot:.1f}", None if prev_tot is None else f"{tot - prev_tot:+.1f} since last step", delta_color="off")
    m2.metric("Patients still awaiting a decision", f"{npat}")
    m3.metric("This patient's share", "none yet" if row is None else f"{float(row['p']):.2f} of {tot:.1f}")
    sp = JR.split(tot)
    st.write("Beds that expected admits would need (using planning shares of level of care): " +
             ", ".join(f"{k.replace('_', '-')} {v:.1f}" for k, v in sp.items()))
    st.caption("This total also moves when other patients arrive or get decided, so not every change comes from the selected patient. "
               "At the admit decision the score becomes a fact and the bed request opens: continue in the Case demo or Bed board tabs.")