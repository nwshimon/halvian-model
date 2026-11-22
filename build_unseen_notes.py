"""
build_unseen_notes.py -- 12 hand-written notes in DIFFERENT wording from the 36 template notes (test only).

Why: the 36 main notes were written from a fixed phrase list, so any system built around those phrases would score
unrealistically well. These 12 use wording the extractor has never been tuned on. Report both sets honestly.
Includes 3 no-blocker notes (2 with resolved items, 1 with a booked post-discharge plan that is NOT a blocker).
Run:  python build_unseen_notes.py
"""
import pandas as pd

import config as C

# (note_id, style, text, [(blocker_type, evidence_phrase), ...], has_resolved_distractor)
NOTES = [
    ("U-01", "plain", "Day 3 after hip fracture repair. Pain controlled and eating well. She lives alone with stairs to her bedroom and "
     "no one from therapy has come to assess her mobility yet, so we cannot safely send her home.",
     [("pt_evaluation", "no one from therapy has come to assess her mobility yet")], False),
    ("U-02", "plain", "Medically ready for discharge. Vitals stable. His son has already been updated on the plan, but "
     "the van company hasn't confirmed a pickup time.",
     [("transport", "the van company hasn't confirmed a pickup time")], False),
    ("U-03", "plain", "COPD patient, saturating 93% on room air when walking, so she qualifies for home O2. "
     "However the supplier hasn't dropped off the concentrator at her apartment.",
     [("home_oxygen_equipment", "the supplier hasn't dropped off the concentrator at her apartment")], False),
    ("U-04", "plain", "Starting apixaban at discharge. Insurance rejected the first fill and "
     "the pharmacy is still waiting on the prior auth from the plan.",
     [("medications", "the pharmacy is still waiting on the prior auth from the plan")], False),
    ("U-05", "plain", "Patient has dementia and is otherwise at baseline. Her daughter lives out of state and "
     "nobody has been named to supervise her at home after discharge.",
     [("family", "nobody has been named to supervise her at home after discharge")], False),
    ("U-06", "plain", "Home health agency needs the face-to-face attestation signed by the physician before they will start services. "
     "Chart currently has no signed copy.",
     [("paperwork", "needs the face-to-face attestation signed by the physician")], False),
    ("U-07", "plain", "Plan is short-term rehab. Two facilities were sent referrals on Monday and neither has gotten back to us.",
     [("placement", "neither has gotten back to us")], False),
    ("U-08", "plain", "Cleared by medicine. OT hasn't signed off on her kitchen safety, and there's no wheelchair van booked for tomorrow.",
     [("pt_evaluation", "OT hasn't signed off on her kitchen safety"), ("transport", "there's no wheelchair van booked for tomorrow")], False),
    ("U-09", "plain", "Walked the hallway twice with PT, who signed off for home. Oxygen was delivered yesterday and the family has already "
     "attended the teaching class. Discharge summary is signed and meds are in hand. Going home this afternoon.",
     [], True),
    ("U-10", "plain", "Afebrile for 48 hours and the wound is clean. Eating independently. Discharge instructions reviewed with the patient "
     "and her husband. Nothing outstanding.",
     [], False),
    ("U-11", "plain", "Will need outpatient physical therapy after discharge; the referral is placed and the first appointment is booked "
     "for next week. No barriers to going home today.",
     [], True),
    ("U-12", "messy", "snf accepted yesterday but bed not avail til fri; auth from insurance not back",
     [("placement", "bed not avail til fri"), ("paperwork", "auth from insurance not back")], False),
]


def main():
    notes = pd.DataFrame([{"note_id": n, "note_text": t, "style": s, "has_resolved_distractor": h} for n, s, t, _, h in NOTES])
    labels = pd.DataFrame([{"note_id": n, "blocker_type": b, "evidence_phrase": e, "owner_role": C.BLOCKER_TYPES[b]}
                           for n, _, _, ls, _ in NOTES for b, e in ls])
    bad = []
    for _, r in labels.merge(notes, on="note_id").iterrows():
        if r["evidence_phrase"] not in r["note_text"]:
            bad.append(f"{r['note_id']}: evidence not verbatim: {r['evidence_phrase']!r}")
        if r["blocker_type"] not in C.BLOCKER_TYPES:
            bad.append(f"{r['note_id']}: bad type")
    notes.to_csv(f"{C.DATA_DIR}/notes_unseen.csv", index=False)
    labels.to_csv(f"{C.DATA_DIR}/notes_unseen_labels.csv", index=False)
    print(f"{len(notes)} unseen notes, {len(labels)} planted blockers, {(~notes.note_id.isin(labels.note_id)).sum()} with no blocker")
    print("PROBLEMS:", bad if bad else "none")


if __name__ == "__main__":
    main()
