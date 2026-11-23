"""
schema.py -- single source of truth for every table's columns.

Design rules
- Category lists (payers, complaints, bed types, blocker types...) are IMPORTED from
  config.py, never retyped here. Change a category in config.py and the schema follows.
- Snapshot feature columns are built from config.MODEL_FEATURES, so the model's feature
  list and the table can never drift apart (an import-time check enforces it).
- Generators, model, matcher and demo import column names via cols("table"). No hard-coded strings.
- ALL data is synthetic. Never load rows from public datasets.

Search for "TODO(spec)" for names to reconcile against the team spec's field lists (Step 9).
"""
from dataclasses import dataclass
import pandas as pd
import config as C


@dataclass(frozen=True)
class Col:
    name: str
    dtype: str          # "str" | "int" | "float" | "bool" | "datetime" | "json"
    desc: str = ""
    enum: tuple = ()    # allowed values (nulls always allowed)


# ------------------------------------------------------------------ enums (from config)
PAYERS = tuple(C.PAYER_MIX)
AGE_BANDS = tuple(C.AGE_BANDS)
COMPLAINTS = tuple(C.COMPLAINT_MIX)
BED_TYPES = tuple(C.BEDS_DEMO["unit_counts"])           # med_surg, telemetry, icu
BED_STATUS = tuple(C.BED_STATUSES)                      # occupied, cleaning, ready, blocked
BLOCKERS = tuple(C.BLOCKER_TYPES)
BLOCKER_OWNERS = tuple(sorted(set(C.BLOCKER_TYPES.values())))
ORDER_NAMES = tuple(C.ORDER_BASE_PROB)
SPLITS = tuple(C.SPLIT) + ("demo",)                     # train, validation, test, demo

# Not defined in config.py. TODO(spec): confirm these with the team spec.
DISPOSITIONS = ("admit", "discharge", "lwbs")
ROLES = ("ed_charge_nurse", "bed_manager", "ed_physician", "inpatient_nurse", "ai", "inpatient_physician", "environmental_services") + BLOCKER_OWNERS   # last two added 2026-11-23 (D5)
MATCH_STATUS = ("proposed", "accepted", "overridden", "rejected")
TASK_STATUS = ("open", "waiting", "in_progress", "done", "cancelled")   # waiting = a dependency is not done (added 2026-11-23)
TASK_TYPES = BLOCKERS + tuple(C.PREP_TAIL_TASKS)                          # blocker types plus discharge_signoff, patient_leaves, bed_cleaning
HOLD_KINDS = ("soft", "hard", "pooled", "decided")
HOLD_STATUS = ("active", "converted", "released", "expired")
HOLD_RELEASE_REASONS = ("p_dropped", "discharged", "lwbs", "wrong_level", "expired", "superseded", "left_ed")
# Includes the boarding-stage events behind config.STALL_LIMIT_MIN and SCORE_ON_EVENTS.
# 'transported' = an ADMITTED patient leaves for the inpatient bed; 'departed' = a discharged
# or LWBS patient leaves. Either one defines visits.left_ed_time (added 2026-11-22).
EVENT_TYPES = ("arrival", "triage", "provider_seen", "order", "result", "reassessment",
               "admit_decision", "bed_requested", "bed_matched", "bed_ready",
               "report_given", "transported", "departed")

# dtype of each model feature (keys must equal config.MODEL_FEATURES)
FEATURE_DTYPES = {
    "age_band": "str", "esi": "int", "complaint_category": "str",
    "max_hr_so_far": "float", "min_spo2_so_far": "float", "min_sbp_so_far": "float",
    "max_rr_so_far": "float", "max_temp_so_far": "float",
    "abnormal_vital_count_so_far": "int",
    "orders_so_far": "int", "ct_orders_so_far": "int",
    "abnormal_results_so_far": "int", "pending_results_so_far": "int",
    "mins_since_arrival": "float", "mins_since_last_reassessment": "float",
    "inpatient_occupancy_at_snapshot": "float",
}
_FEATURE_ENUMS = {"age_band": AGE_BANDS, "complaint_category": COMPLAINTS, "esi": (1, 2, 3, 4, 5)}

# ------------------------------------------------------------------ tables
SCHEMA = {
    # ---- Step 1: one per-visit generator loop writes these three ----------
    "visits": [
        Col("visit_id", "str", "Primary key"),
        Col("arrival_time", "datetime"),
        Col("age_band", "str", enum=AGE_BANDS),
        Col("payer", "str", "FAIRNESS ONLY, never a feature", enum=PAYERS),
        Col("complaint_category", "str", enum=COMPLAINTS),
        Col("esi", "int", "1 = most urgent", enum=(1, 2, 3, 4, 5)),
        # --- bridge to bed matching: generated from complaint/ESI/age, NOT from the admit label
        Col("requested_level_of_care", "str", "Admits only (null otherwise)", enum=BED_TYPES),
        Col("isolation_required", "bool", "Admits only (null otherwise)"),
        Col("preferred_unit", "str", "Admits only; feeds MATCH_WEIGHTS['preferred_unit']"),
        # --- outcome columns: labels or post-decision. All are in config.PROHIBITED_FEATURES
        Col("disposition", "str", enum=DISPOSITIONS),
        Col("admitted", "bool", "LABEL. disposition == 'admit'"),
        Col("lwbs", "bool", "disposition == 'lwbs'"),
        Col("door_to_provider_min", "float", "Null for LWBS"),
        Col("disposition_time", "datetime", "Admit decision time. Null unless admitted"),
        Col("bed_requested_time", "datetime"),
        Col("bed_matched_time", "datetime"),
        Col("bed_ready_time", "datetime"),
        Col("left_ed_time", "datetime", "Time of the 'transported' event (admits) or 'departed' event (everyone else)"),
        Col("boarding_hours", "float", "left_ed_time - disposition_time"),
        Col("split", "str", "Assigned per visit so no visit spans splits", enum=SPLITS),
    ],
    "events": [
        Col("event_id", "str", "Primary key"),
        Col("visit_id", "str", "FK visits"),
        Col("event_time", "datetime", "Snapshots may only use events with event_time <= snapshot_time"),
        Col("event_type", "str", enum=EVENT_TYPES),
        Col("order_name", "str", "Set when event_type is 'order' or 'result'", enum=ORDER_NAMES),
        Col("result_abnormal", "bool", "Set when event_type is 'result'"),
        Col("actor_role", "str", enum=ROLES),
    ],
    "vitals": [
        Col("vitals_id", "str", "Primary key"),
        Col("visit_id", "str", "FK visits"),
        Col("recorded_time", "datetime"),
    ] + [Col(v, "float", "vital sign") for v in C.VITAL_BASELINE],      # hr, spo2, sbp, rr, temp_c

    # ---- hospital-level context for the inpatient_occupancy feature -------
    "occupancy_hourly": [
        Col("hour_start", "datetime", "Primary key"),
        Col("inpatient_occupancy", "float", "0 to 1. Mean config.INPATIENT_OCCUPANCY; a deliberate null effect on admission"),
    ],

    # ---- Step 2: derived ---------------------------------------------------
    "snapshots": [
        Col("snapshot_id", "str", "Primary key"),
        Col("visit_id", "str", "FK visits"),
        Col("snapshot_time", "datetime", "The 'known as of' time"),
        Col("trigger_event_id", "str", "FK events"),
        Col("trigger_type", "str", enum=tuple(C.SCORE_ON_EVENTS)),
    ] + [Col(f, FEATURE_DTYPES[f], "model feature", _FEATURE_ENUMS.get(f, ())) for f in C.MODEL_FEATURES] + [
        Col("label_admit", "bool", "visits.admitted. LABEL, never a feature"),
        Col("split", "str", "Inherited from visits.split", enum=SPLITS),
    ],

    # ---- Step 5: demo setup ------------------------------------------------
    "beds": [
        Col("bed_id", "str", "Primary key"),
        Col("unit", "str"),
        Col("bed_type", "str", enum=BED_TYPES),
        Col("isolation_capable", "bool"),
        Col("status", "str", enum=BED_STATUS),
        Col("discharge_pending", "bool", "Display state (amber). Only meaningful when occupied"),
        Col("current_visit_id", "str", "Null unless occupied"),
        Col("status_updated_time", "datetime"),
        Col("expected_ready_time", "datetime", "Drives MATCH_WEIGHTS['minutes_until_ready']"),
    ],

    # ---- Step 6: test only -------------------------------------------------
    "inpatient_notes": [
        Col("note_id", "str", "Primary key"),
        Col("visit_id", "str", "FK visits"),
        Col("bed_id", "str", "FK beds"),
        Col("note_time", "datetime"),
        Col("author_role", "str", enum=ROLES),
        Col("note_text", "str"),
        Col("in_demo", "bool", "One of the config.N_NOTES_DEMO notes shown live"),
    ],
    "note_labels": [   # the answer key; never shown to the extractor
        Col("note_id", "str", "FK inpatient_notes. One row per planted blocker; notes with no blocker have NO rows"),
        Col("blocker_type", "str", enum=BLOCKERS),
        Col("evidence_phrase", "str", "Must appear verbatim in note_text"),
        Col("owner_role", "str", enum=BLOCKER_OWNERS),
    ],

    # ---- written live by the demo, start empty -----------------------------
    "bed_matches": [
        Col("match_id", "str", "Primary key"),
        Col("visit_id", "str"),
        Col("bed_id", "str"),
        Col("created_time", "datetime"),
        Col("rank", "int", f"1..{C.MATCH_TOP_N}"),
        Col("match_score", "float"),
        Col("reason_codes", "json", "Keys of config.MATCH_REASON_CODES that fired"),
        Col("status", "str", enum=MATCH_STATUS),
        Col("decided_by_role", "str", enum=ROLES),
        Col("override_reason", "str"),
    ],
    "tasks": [
        Col("task_id", "str", "Primary key"),
        Col("visit_id", "str", "For note_extraction tasks and early-prep chains: the INPATIENT whose discharge this task works on"),
        Col("created_time", "datetime"),
        Col("task_type", "str", "A blocker type, or discharge_signoff / patient_leaves / bed_cleaning", enum=TASK_TYPES),
        Col("owner_role", "str", enum=ROLES),
        Col("due_time", "datetime"),
        Col("status", "str", enum=TASK_STATUS),
        Col("source", "str", "'note_extraction', 'early_prep' or 'manual'"),
        Col("bed_id", "str", "The bed this task frees (early_prep only)"),
        Col("hold_id", "str", "The hold this task serves (early_prep only)"),
        Col("depends_on", "str", "task_id list, comma separated (early_prep only)"),
        Col("start_by_time", "datetime"),
        Col("duration_min", "float"),
        Col("priority", "str", "standard or urgent"),
        Col("critical", "bool", "The chain cannot finish by the claimant's needed-by time"),
        Col("completed_time", "datetime"),
    ],
    "bed_holds": [
        Col("hold_id", "str", "Primary key"),
        Col("visit_id", "str", "The ED patient the bed is held for. Empty for a pooled hold (held for a level of care, not a named patient)"),
        Col("bed_id", "str"),
        Col("tier", "int", "1 ready, 2 cleaning, 3 discharge-pending"),
        Col("kind", "str", enum=HOLD_KINDS),
        Col("level_assumed", "str", enum=BED_TYPES),
        Col("level_confidence", "float", "Share from the level prior (1.0 once the doctor decides)"),
        Col("p_at_hold", "float"),
        Col("p_latest", "float"),
        Col("needed_by_time", "datetime", "From the decision-time prior (25th percentile)"),
        Col("realistic_ready_time", "datetime", "Blockers + sign-off + patient leaves + cleaning, from config durations"),
        Col("critical", "bool"),
        Col("status", "str", enum=HOLD_STATUS),
        Col("release_reason", "str", enum=HOLD_RELEASE_REASONS),
        Col("created_time", "datetime"),
        Col("updated_time", "datetime"),
    ],
    "audit_log": [
        Col("log_id", "str", "Primary key"),
        Col("log_time", "datetime"),
        Col("actor", "str", "ai or a role", enum=ROLES),
        Col("action", "str"),
        Col("entity_type", "str"),
        Col("entity_id", "str"),
        Col("detail", "json"),
    ],
}

FOREIGN_KEYS = {   # (child_table, child_col) -> (parent_table, parent_col)
    ("events", "visit_id"): ("visits", "visit_id"),
    ("vitals", "visit_id"): ("visits", "visit_id"),
    ("snapshots", "visit_id"): ("visits", "visit_id"),
    ("snapshots", "trigger_event_id"): ("events", "event_id"),
    ("inpatient_notes", "visit_id"): ("visits", "visit_id"),
    ("inpatient_notes", "bed_id"): ("beds", "bed_id"),
    ("note_labels", "note_id"): ("inpatient_notes", "note_id"),
    ("bed_matches", "visit_id"): ("visits", "visit_id"),
    ("bed_matches", "bed_id"): ("beds", "bed_id"),
    ("tasks", "visit_id"): ("visits", "visit_id"),
    ("bed_holds", "bed_id"): ("beds", "bed_id"),
}
LIVE_TABLES = ("bed_matches", "tasks", "audit_log", "bed_holds")
COMPOSITE_KEY_TABLES = ("note_labels",)                   # no single primary key
# Never allowed as model features: config's list plus the snapshot label.
LEAKY_COLUMNS = set(C.PROHIBITED_FEATURES) | {"label_admit"}

_DTYPES = {"str": "object", "int": "Int64", "float": "float64",
           "bool": "boolean", "datetime": "datetime64[ns]", "json": "object"}
_SQL = {"str": "TEXT", "int": "INTEGER", "float": "REAL",
        "bool": "INTEGER", "datetime": "TEXT", "json": "TEXT"}


# ------------------------------------------------------------------ helpers
def cols(table: str) -> list:
    return [c.name for c in SCHEMA[table]]


def feature_cols() -> list:
    """The snapshot columns the model may use (exactly config.MODEL_FEATURES)."""
    return list(C.MODEL_FEATURES)


def empty_frame(table: str) -> pd.DataFrame:
    return pd.DataFrame({c.name: pd.Series(dtype=_DTYPES[c.dtype]) for c in SCHEMA[table]})


def validate(df: pd.DataFrame, table: str) -> list:
    """Problems found in a generated table (empty list = OK)."""
    problems = []
    expected = cols(table)
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    if missing:
        problems.append(f"{table}: missing columns {missing}")
    if extra:
        problems.append(f"{table}: unexpected columns {extra}")
    for c in SCHEMA[table]:
        if c.enum and c.name in df.columns:
            bad = set(df[c.name].dropna().unique()) - set(c.enum)
            if bad:
                problems.append(f"{table}.{c.name}: values outside enum {sorted(map(str, bad))}")
    pk = expected[0]
    if table not in COMPOSITE_KEY_TABLES and pk in df.columns and df[pk].duplicated().any():
        problems.append(f"{table}.{pk}: duplicate primary keys")
    if table == "visits" and not missing and len(df):
        for must_exist in ("disposition", "admitted", "lwbs", "split"):
            if df[must_exist].isna().any():
                problems.append(f"visits.{must_exist}: has nulls")
        if (df["admitted"].eq(True) != (df["disposition"] == "admit")).any():
            problems.append("visits: admitted disagrees with disposition")
        if (df["lwbs"].eq(True) != (df["disposition"] == "lwbs")).any():
            problems.append("visits: lwbs disagrees with disposition")
        non_admit = df[df["disposition"] != "admit"]
        if non_admit[["requested_level_of_care", "boarding_hours"]].notna().any().any():
            problems.append("visits: bed/boarding fields set on non-admitted visits")
    return problems


def check_foreign_keys(tables: dict) -> list:
    problems = []
    for (child, ccol), (parent, pcol) in FOREIGN_KEYS.items():
        if child in tables and parent in tables and len(tables[child]):
            orphans = ~tables[child][ccol].dropna().isin(tables[parent][pcol])
            if orphans.any():
                problems.append(f"{child}.{ccol}: {int(orphans.sum())} rows not in {parent}.{pcol}")
    return problems


VISIT_TIME_FROM_EVENT = {            # visits column -> event_type it must equal
    "arrival_time": "arrival",
    "disposition_time": "admit_decision",
    "bed_requested_time": "bed_requested",
    "bed_matched_time": "bed_matched",
    "bed_ready_time": "bed_ready",
}


def check_visit_event_consistency(visits: pd.DataFrame, events: pd.DataFrame) -> list:
    """visits timestamps must be DERIVED from events, so the two tables can never disagree.
    Also checks boarding_hours = left_ed_time - disposition_time."""
    problems = []
    v = visits.set_index("visit_id")

    def first_time(types):
        return events[events["event_type"].isin(types)].groupby("visit_id")["event_time"].min()

    pairs = {col: first_time([et]) for col, et in VISIT_TIME_FROM_EVENT.items()}
    pairs["left_ed_time"] = first_time(["transported", "departed"])
    for col, from_events in pairs.items():
        both = pd.concat([v[col], from_events], axis=1, keys=["visits", "events"])
        same = (both["visits"] == both["events"]) | (both["visits"].isna() & both["events"].isna())
        if (~same).any():
            problems.append(f"visits.{col} disagrees with events for {int((~same).sum())} visits")
    admits = v[v["admitted"].eq(True)]
    if len(admits):
        hours = (admits["left_ed_time"] - admits["disposition_time"]).dt.total_seconds() / 3600
        if ((hours - admits["boarding_hours"]).abs() > 0.01).any():
            problems.append("visits.boarding_hours != left_ed_time - disposition_time")
    return problems


def check_snapshots(snapshots: pd.DataFrame, visits: pd.DataFrame, events: pd.DataFrame) -> list:
    """Rules a snapshots table must obey (added 2026-11-22).
    1 the label never changes within a visit and equals visits.admitted
    2 split is inherited from visits, so no visit spans two splits
    3 the trigger event exists, belongs to the same visit and happened at snapshot_time
    4 a snapshot is taken before the admit decision (admits) and before the patient leaves (everyone)
    5 every visit has exactly one triage snapshot, and it is the visit's first"""
    problems = []
    v = visits.set_index("visit_id")
    if (snapshots.groupby("visit_id")["label_admit"].nunique() > 1).any():
        problems.append("snapshots: label_admit changes within a visit")
    if (snapshots["label_admit"].astype(bool).to_numpy() != v["admitted"].astype(bool).reindex(snapshots["visit_id"]).to_numpy()).any():
        problems.append("snapshots: label_admit disagrees with visits.admitted")
    if (snapshots["split"].to_numpy() != v["split"].reindex(snapshots["visit_id"]).to_numpy()).any():
        problems.append("snapshots: split differs from visits.split")
    e = events.set_index("event_id")
    trig = e.reindex(snapshots["trigger_event_id"])
    if trig["visit_id"].isna().any():
        problems.append("snapshots: trigger_event_id not found in events")
    else:
        if (trig["visit_id"].to_numpy() != snapshots["visit_id"].to_numpy()).any():
            problems.append("snapshots: trigger event belongs to a different visit")
        if (trig["event_time"].to_numpy() != snapshots["snapshot_time"].to_numpy()).any():
            problems.append("snapshots: snapshot_time != trigger event time")
        if (trig["event_type"].to_numpy() != snapshots["trigger_type"].to_numpy()).any():
            problems.append("snapshots: trigger_type != trigger event's event_type")
    dt = v["disposition_time"].reindex(snapshots["visit_id"]).to_numpy()
    st = snapshots["snapshot_time"].to_numpy()
    if (~pd.isna(dt) & (st >= dt)).any():
        problems.append("snapshots: a snapshot at or after the admit decision (disposition_time)")
    if (st > v["left_ed_time"].reindex(snapshots["visit_id"]).to_numpy()).any():
        problems.append("snapshots: a snapshot after the patient left")
    tri = snapshots[snapshots["trigger_type"] == "triage"]
    if tri["visit_id"].duplicated().any() or set(tri["visit_id"]) != set(visits["visit_id"]):
        problems.append("snapshots: not exactly one triage snapshot per visit")
    first = snapshots.sort_values(["visit_id", "snapshot_time"]).groupby("visit_id")["trigger_type"].first()
    if (first != "triage").any():
        problems.append("snapshots: a visit's first snapshot is not its triage snapshot")
    return problems


def sqlite_ddl() -> str:
    out = []
    for t, columns in SCHEMA.items():
        body = ",\n  ".join(
            f"{c.name} {_SQL[c.dtype]}" + (" PRIMARY KEY" if i == 0 and t not in COMPOSITE_KEY_TABLES else "")
            for i, c in enumerate(columns))
        out.append(f"CREATE TABLE IF NOT EXISTS {t} (\n  {body}\n);")
    return "\n\n".join(out)


def check_against_config():
    """Fails loudly if schema and config drift apart. Runs at import."""
    assert set(FEATURE_DTYPES) == set(C.MODEL_FEATURES), "FEATURE_DTYPES != config.MODEL_FEATURES"
    assert set(C.SCORE_ON_EVENTS) <= set(EVENT_TYPES), "SCORE_ON_EVENTS has an event type not in EVENT_TYPES"
    stage_events = {s for k in C.STALL_LIMIT_MIN for s in k.split("_to_")}
    stage_events = {s.replace("_decided", "_decision") for s in stage_events}
    assert stage_events <= set(EVENT_TYPES), f"stall stages need events: {stage_events - set(EVENT_TYPES)}"
    snapshot_leaks = LEAKY_COLUMNS & set(C.MODEL_FEATURES)
    assert not snapshot_leaks, f"leaky columns in MODEL_FEATURES: {snapshot_leaks}"
    assert set(C.PROHIBITED_FEATURES) - {"race", "ethnicity", "language"} <= set(cols("visits")), \
        "PROHIBITED_FEATURES names a column that visits does not have"
    for t in SCHEMA:
        names = cols(t)
        assert len(names) == len(set(names)), f"duplicate column names in {t}"


check_against_config()

if __name__ == "__main__":
    for t in SCHEMA:
        assert not validate(empty_frame(t), t), t
        print(f"{t:17s} {len(SCHEMA[t]):2d} columns")
    print("\nOK: schema matches config; all empty frames validate.")
