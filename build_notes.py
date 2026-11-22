"""
build_notes.py -- Step 6: synthetic inpatient notes + the answer key for discharge-blocker extraction.

What this makes (test data only: nothing here trains anything; it is the exam for the extractor)
  data/inpatient_notes.csv   36 short progress notes on the 30 occupied demo beds (6 beds get a second note)
  data/note_labels.csv       the ANSWER KEY: one row per planted blocker. Notes with no blocker have no rows.
  data/note_meta.csv         test metadata (style, number of blockers, whether the note has a resolved distractor)

Design choices (all counts come from config.py)
  * 25% of notes have NO blocker, so an extractor that invents one fails the test.
  * Some notes mention an item that is already RESOLVED ("PT evaluation was completed this morning"). That is not a
    blocker; an extractor that flags it fails the test. These are the hard negatives.
  * 35% of notes are written in a messy hand-typed style (abbreviations, fragments); the rest are clean.
  * Every evidence phrase appears word-for-word in its note (config.EXTRACTION_REQUIRE_VERBATIM_EVIDENCE).
  * Which blockers a patient gets depends on age and complaint (older patients: PT, placement; breathing problems:
    home oxygen; children: family). It never depends on payer, race, language or anything else protected.
  * The demo shows the 12 notes on the 6 discharge-pending beds (two notes each).

Run after build_demo.py:  python build_notes.py
"""
import os

import numpy as np
import pandas as pd

import config as C
import schema as S

NOW = pd.Timestamp(C.DEMO_DATE) + pd.Timedelta(hours=int(C.DEMO_NOW_TIME[:2]), minutes=int(C.DEMO_NOW_TIME[3:]))

# ---- phrase banks: (clean, messy). The evidence phrase is exactly one of these strings.
ACTIVE = {
    "pt_evaluation": (["physical therapy evaluation has not been completed", "awaiting PT evaluation for safe mobility at home",
                       "PT/OT has not yet assessed whether the patient can manage stairs"],
                      ["PT eval pending", "pt/ot not seen yet", "no PT clearance yet"]),
    "transport": (["no ride home has been arranged", "transport home is not yet confirmed",
                   "medical transport has not been booked"],
                  ["no ride home", "transport not set up", "need transport 2 go home"]),
    "home_oxygen_equipment": (["home oxygen has not been delivered", "needs home oxygen set up before discharge",
                               "oxygen equipment for home is still being arranged"],
                              ["home O2 not set up", "awaiting DME o2", "o2 equip not delivered"]),
    "medications": (["discharge medications are awaiting prior authorization", "new anticoagulant has not yet been filled by pharmacy",
                     "medication reconciliation is not finished"],
                    ["meds prior auth pending", "waiting on pharmacy for dc meds", "med rec not done"]),
    "family": (["family meeting about the discharge plan has not taken place", "no caregiver has been identified to stay with the patient at home",
                "family has not yet completed discharge teaching"],
               ["fam mtg not done", "no caregiver at home", "fam teaching not done"]),
    "paperwork": (["discharge paperwork is incomplete", "home health referral form has not been signed",
                   "insurance authorization form is still missing a signature"],
                  ["dc paperwork incomplete", "HH referral unsigned", "auth form not signed"]),
    "placement": (["skilled nursing facility placement is pending acceptance", "awaiting a bed offer from the rehabilitation facility",
                   "facility has not yet accepted the patient"],
                  ["SNF bed pending", "rehab acceptance pending", "no facility accept yet"]),
}
RESOLVED = {   # hard negatives: a past blocker that is already fixed. These must never be labelled.
    "pt_evaluation": (["PT evaluation was completed this morning and cleared the patient for home"], ["PT eval done, cleared"]),
    "transport": (["transport has been arranged for this afternoon"], ["ride set up"]),
    "home_oxygen_equipment": (["home oxygen was delivered and teaching is done"], ["o2 delivered"]),
    "medications": (["prior authorization was approved and medications are filled"], ["meds filled"]),
    "family": (["family meeting was held and the family agrees with the plan"], ["fam mtg done"]),
    "paperwork": (["discharge paperwork has been signed"], ["paperwork signed"]),
    "placement": (["rehabilitation facility has accepted the patient"], ["rehab accepted"]),
}
DX = {"shortness_of_breath": ["COPD exacerbation", "pneumonia", "heart failure"], "chest_pain": ["chest pain, observation", "NSTEMI ruled out"],
      "abdominal_pain": ["small bowel obstruction", "pancreatitis"], "neuro": ["TIA", "seizure"],
      "infection_fever": ["urinary tract infection", "cellulitis"], "injury_trauma": ["hip fracture", "rib fractures"],
      "psychiatric": ["medical clearance after overdose"], "other": ["dehydration", "electrolyte imbalance"]}
STATUS_CLEAN = ["Clinically stable and medically ready for discharge.", "Vitals stable overnight and tolerating diet; medically cleared.",
                "No acute events overnight; medically ready to leave when logistics allow."]
STATUS_MESSY = ["VSS, medically stable.", "stable, no events overnight.", "VSS. tolerating diet. med clear."]
LEAD = ["Remaining issues: ", "Not yet ready because ", "Holding discharge: ", "Per team, "]
CLOSE_BLOCKED = ["Will reassess this afternoon.", "Case management aware.", "Plan to discharge once these are complete."]
CLOSE_FREE = ["Discharge orders to be entered; patient may go home today.", "No barriers to discharge identified."]
AUTHORS = (["inpatient_nurse"] * 7, ["case_management", "social_work", "PT/OT", "pharmacy"])


def weights(age_band, complaint):
    old = age_band in ("65-74", "75+")
    w = {k: 1.0 for k in ACTIVE}
    w["pt_evaluation"] = 2.5 if (old or complaint in ("injury_trauma", "neuro")) else 0.6
    w["placement"] = 3.0 if age_band == "75+" else (0.3 if old else 0.0)
    w["home_oxygen_equipment"] = 3.0 if complaint == "shortness_of_breath" else 0.3
    w["family"] = 3.0 if age_band == "0-17" else 1.0
    w["paperwork"] = 1.6
    if age_band == "0-17":
        w["pt_evaluation"] = w["home_oxygen_equipment"] = 0.0
    return w


def pick_types(rng, age_band, complaint, n):
    w = weights(age_band, complaint)
    keys = [k for k in w if w[k] > 0]
    p = np.array([w[k] for k in keys]); p = p / p.sum()
    return [str(k) for k in rng.choice(keys, size=n, replace=False, p=p)]


def write_note(rng, hd, age_band, complaint, active, resolved, messy):
    dx = str(rng.choice(DX[complaint]))
    phrases = [(t, str(rng.choice(ACTIVE[t][1 if messy else 0]))) for t in active]
    neg = [str(rng.choice(RESOLVED[t][1 if messy else 0])) for t in resolved]
    if messy:
        parts = [f"HD{hd} {dx}, {age_band}yo. {rng.choice(STATUS_MESSY)}"]
        if phrases:
            parts.append(" | ".join(p for _, p in phrases) + ".")
        if neg:
            parts.append("done: " + ", ".join(neg) + ".")
        parts.append("f/u pm." if phrases else "ok for dc.")
    else:
        parts = [f"Hospital day {hd}. Patient (age band {age_band}) admitted with {dx}. {rng.choice(STATUS_CLEAN)}"]
        if phrases:
            parts.append(rng.choice(LEAD) + "; and ".join(p for _, p in phrases) + ".")
        if neg:
            parts.append("Already resolved: " + "; ".join(neg) + ".")
        parts.append(rng.choice(CLOSE_BLOCKED if phrases else CLOSE_FREE))
    return " ".join(parts), phrases


def generate(beds, visits, seed=C.RANDOM_SEED + 2):
    rng = np.random.default_rng(seed)
    occ = beds[beds["status"] == "occupied"].merge(visits[["visit_id", "age_band", "complaint_category", "left_ed_time"]],
                                                    left_on="current_visit_id", right_on="visit_id")
    pend = occ[occ["discharge_pending"]]
    rest = occ[~occ["discharge_pending"]]
    assert len(pend) * 2 == C.N_NOTES_DEMO and len(occ) + len(pend) == C.N_NOTES_TOTAL, "bed counts do not match note counts in config"

    slots = []                                              # (bed row, in_demo, note_time)
    for _, r in pend.iterrows():
        slots.append((r, True, NOW - pd.Timedelta(hours=1) * float(rng.uniform(13, 16))))   # evening note, day before
        slots.append((r, True, NOW - pd.Timedelta(hours=1) * float(rng.uniform(1.0, 3.0))))  # morning note
    for _, r in rest.iterrows():
        slots.append((r, False, NOW - pd.Timedelta(hours=1) * float(rng.uniform(1.0, 5.0))))
    n = len(slots)
    demo_idx = [i for i, s in enumerate(slots) if s[1]]
    other_idx = [i for i, s in enumerate(slots) if not s[1]]

    def plan(idx, n_none, n_none_hardneg, n_double, n_messy):
        a = list(rng.permutation(idx))
        none, withb = a[:n_none], a[n_none:]
        kind = {i: 0 for i in none}
        for j, i in enumerate(withb):
            kind[i] = 2 if j < n_double else 1
        hard = set(none[:n_none_hardneg])
        messy = set(list(rng.permutation(idx))[:n_messy])
        return kind, hard, messy
    total_none = round(C.NOTE_NO_BLOCKER_FRACTION * C.N_NOTES_TOTAL)            # 9
    total_messy = round(C.NOTE_MESSY_FRACTION * C.N_NOTES_TOTAL)                # 13
    k1, h1, m1 = plan(demo_idx, 2, 1, 3, round(total_messy * len(demo_idx) / n))
    k2, h2, m2 = plan(other_idx, total_none - 2, 4, 4, total_messy - round(total_messy * len(demo_idx) / n))
    kind, hard, messy = {**k1, **k2}, h1 | h2, m1 | m2

    notes, labels, meta = [], [], []
    order = sorted(range(n), key=lambda i: slots[i][2])
    for num, i in enumerate(order, start=1):
        r, in_demo, t = slots[i]
        nid = f"N-{num:03d}"
        nb = kind[i]
        active = pick_types(rng, r["age_band"], r["complaint_category"], nb) if nb else []
        pool = [k for k in ACTIVE if k not in active and weights(r["age_band"], r["complaint_category"])[k] > 0]
        want_resolved = (i in hard) or (nb and rng.random() < 0.30)
        resolved = [str(rng.choice(pool))] if want_resolved else []
        hd = max(1, (t.normalize() - r["left_ed_time"].normalize()).days + 1)
        text, phrases = write_note(rng, hd, r["age_band"], r["complaint_category"], active, resolved, i in messy)
        role = str(rng.choice(AUTHORS[0] + AUTHORS[1]))
        notes.append({"note_id": nid, "visit_id": r["current_visit_id"], "bed_id": r["bed_id"], "note_time": t.floor("min"),
                      "author_role": role, "note_text": text, "in_demo": bool(in_demo)})
        for typ, ph in phrases:
            labels.append({"note_id": nid, "blocker_type": typ, "evidence_phrase": ph, "owner_role": C.BLOCKER_TYPES[typ]})
        meta.append({"note_id": nid, "style": "messy" if i in messy else "clean", "n_blockers": nb, "has_resolved_distractor": bool(resolved)})
    return pd.DataFrame(notes), pd.DataFrame(labels, columns=S.cols("note_labels")), pd.DataFrame(meta)


def check(notes, labels, meta, beds, visits):
    bad = S.validate(notes, "inpatient_notes") + S.validate(labels, "note_labels")
    bad += S.check_foreign_keys({"visits": visits, "beds": beds, "inpatient_notes": notes, "note_labels": labels})
    if len(notes) != C.N_NOTES_TOTAL:
        bad.append(f"{len(notes)} notes, config says {C.N_NOTES_TOTAL}")
    if int(notes["in_demo"].sum()) != C.N_NOTES_DEMO:
        bad.append("demo note count differs from config")
    merged = labels.merge(notes[["note_id", "note_text"]], on="note_id")
    for _, r in merged.iterrows():
        if C.EXTRACTION_REQUIRE_VERBATIM_EVIDENCE and r["evidence_phrase"] not in r["note_text"]:
            bad.append(f"evidence not verbatim in {r['note_id']}: {r['evidence_phrase']!r}")
        if r["note_text"].count(r["evidence_phrase"]) != 1:
            bad.append(f"evidence phrase appears more than once in {r['note_id']}")
    if labels.duplicated(["note_id", "blocker_type"]).any():
        bad.append("a blocker type is labelled twice in one note")
    none_share = (meta["n_blockers"] == 0).mean()
    if abs(none_share - C.NOTE_NO_BLOCKER_FRACTION) > 0.02:
        bad.append(f"no-blocker share {none_share:.2f} vs config {C.NOTE_NO_BLOCKER_FRACTION}")
    if abs((meta["style"] == "messy").mean() - C.NOTE_MESSY_FRACTION) > 0.03:
        bad.append("messy share differs from config")
    for _, r in meta[meta["has_resolved_distractor"]].iterrows():
        txt = notes.set_index("note_id").loc[r["note_id"], "note_text"]
        if not any(p in txt for bank in RESOLVED.values() for p in bank[0] + bank[1]):
            bad.append(f"{r['note_id']} says it has a resolved item but none is in the text")
    if (notes["note_time"] >= NOW).any():
        bad.append("a note is time-stamped after DEMO_NOW")
    return bad


def main():
    beds = pd.read_csv(f"{C.DATA_DIR}/beds.csv", parse_dates=["status_updated_time", "expected_ready_time"])
    visits = pd.read_csv(f"{C.DATA_DIR}/visits.csv", parse_dates=["arrival_time", "left_ed_time"])
    notes, labels, meta = generate(beds, visits)
    problems = check(notes, labels, meta, beds, visits)

    print(f"=== Step 6: {len(notes)} notes, {len(labels)} planted blockers ===")
    print(f"in demo: {int(notes['in_demo'].sum())} | no blocker: {(meta.n_blockers == 0).sum()} ({(meta.n_blockers == 0).mean():.0%}) | "
          f"messy: {(meta['style'] == 'messy').sum()} ({(meta['style'] == 'messy').mean():.0%}) | "
          f"with a resolved distractor: {int(meta.has_resolved_distractor.sum())}")
    print("blockers per note:", meta["n_blockers"].value_counts().sort_index().to_dict())
    print("blocker types:", labels["blocker_type"].value_counts().to_dict())
    print("author roles:", notes["author_role"].value_counts().to_dict())
    print("\nExamples (note -> answer key):")
    for style, nb, hard in [("clean", 2, None), ("messy", 1, None), ("clean", 0, True), ("messy", 0, None)]:
        m = meta[(meta["style"] == style) & (meta["n_blockers"] == nb)]
        if hard:
            m = m[m["has_resolved_distractor"]]
        if m.empty:
            continue
        nid = m["note_id"].iloc[0]
        print(f"\n[{nid}] ({style}, {nb} blocker{'s' if nb != 1 else ''}{', resolved distractor' if hard else ''})")
        print("  " + notes.set_index("note_id").loc[nid, "note_text"])
        for _, r in labels[labels["note_id"] == nid].iterrows():
            print(f"  KEY: {r['blocker_type']} <- {r['evidence_phrase']!r} -> {r['owner_role']}")
        if nb == 0:
            print("  KEY: (none)")
    print("\nPROBLEMS:", problems if problems else "none")

    os.makedirs(C.DATA_DIR, exist_ok=True)
    notes[S.cols("inpatient_notes")].to_csv(f"{C.DATA_DIR}/inpatient_notes.csv", index=False)
    labels.to_csv(f"{C.DATA_DIR}/note_labels.csv", index=False)
    meta.to_csv(f"{C.DATA_DIR}/note_meta.csv", index=False)
    print(f"\nWrote inpatient_notes.csv, note_labels.csv, note_meta.csv to ./{C.DATA_DIR}/")
    return problems


if __name__ == "__main__":
    import build_notes
    build_notes.main()