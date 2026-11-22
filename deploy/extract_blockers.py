"""
extract_blockers.py -- the discharge-blocker extractor (Step 7).

  An LLM reads an inpatient progress note and returns the reasons a medically ready patient has NOT left:
      blocker_type (closed list), evidence_phrase (word-for-word from the note), owner_role, confidence

Guardrails (the LLM never has the last word on what is shown)
  * blocker_type must be on the closed list in config.BLOCKER_TYPES, or the finding is dropped.
  * evidence_phrase must appear verbatim in the note, or the finding is dropped. This is what lets a nurse verify it.
  * owner_role is looked up from config, never chosen by the LLM.
  * Already-resolved items ("PT evaluation was completed") are NOT blockers; the prompt says so and the scorer tests it.
  * If the LLM is unavailable the app can fall back to rules_extract() (simple keyword rules), clearly labelled.

Score it against the answer key (needs an API key, or a filled cache):
    python extract_blockers.py --score            # the 36 main notes
    python extract_blockers.py --score --unseen   # also the 12 differently worded notes (the fairer test)
    python extract_blockers.py --score --rules    # the rules-only fallback, for comparison
"""
import argparse
import re

import pandas as pd

import config as C
import llm_client as L

DEFINITIONS = {
    "pt_evaluation": "a physical or occupational therapy assessment/clearance is still missing or pending",
    "transport": "no ride home or medical transport has been arranged or confirmed",
    "home_oxygen_equipment": "home oxygen or other medical equipment is not yet set up or delivered",
    "medications": "discharge medications are unresolved (prior authorization, not filled, reconciliation unfinished)",
    "family": "family meeting, teaching or a home caregiver is missing or not yet arranged",
    "paperwork": "discharge, referral or insurance forms/authorizations are incomplete or unsigned (the physician simply entering the routine discharge order is NOT paperwork)",
    "placement": "a facility placement (skilled nursing, rehab) is pending acceptance or a bed",
}
SYSTEM = (
    "You extract discharge blockers from a hospital inpatient progress note. A discharge blocker is a specific, still-unresolved "
    "item that is stopping a medically ready patient from leaving the hospital.\n\n"
    "Allowed blocker_type values (use exactly these strings):\n"
    + "\n".join(f"- {k}: {v}" for k, v in DEFINITIONS.items())
    + "\n\nRules:\n"
    "1. Report ONLY items that are still open. If the note says something was completed, delivered, signed, approved, arranged, "
    "accepted or held, it is resolved: do NOT report it.\n"
    "2. Do not report plans for after discharge (for example an outpatient appointment that is already booked).\n"
    "3. evidence_phrase must be copied word for word from the note: one contiguous span, as short as possible, no paraphrasing.\n"
    "4. If nothing is blocking discharge, return an empty list. Do not invent blockers. A routine step that will happen as the patient leaves "
    "(for example 'discharge orders to be entered; patient may go home today') is NOT a blocker.\n"
    "5. At most one entry per blocker_type.\n"
    "6. confidence is a number from 0 to 1.\n\n"
    'Reply with JSON only, exactly: {"blockers": [{"blocker_type": "...", "evidence_phrase": "...", "confidence": 0.0}]}'
)


def _find_verbatim(phrase, text):
    """Return the exact span of `text` equal to `phrase` (ignoring case and extra spaces), or None."""
    if not phrase:
        return None
    if phrase in text:
        return phrase
    norm = re.sub(r"\s+", " ", phrase.strip())
    m = re.search(re.escape(norm).replace(r"\ ", r"\s+"), text, flags=re.I)
    return m.group(0) if m else None


def validate(raw, note_text):
    """Turn raw LLM output into (findings, dropped). Only valid findings are ever shown to staff."""
    findings, dropped, seen = [], [], set()
    items = raw.get("blockers", []) if isinstance(raw, dict) else []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        typ = str(it.get("blocker_type", "")).strip()
        if typ not in C.BLOCKER_TYPES:
            dropped.append({"item": it, "why": "blocker_type is not on the closed list"}); continue
        span = _find_verbatim(str(it.get("evidence_phrase", "")), note_text)
        if span is None:
            dropped.append({"item": it, "why": "evidence_phrase is not in the note"}); continue
        if typ in seen:
            dropped.append({"item": it, "why": "duplicate blocker_type"}); continue
        seen.add(typ)
        try:
            conf = min(1.0, max(0.0, float(it.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        findings.append({"blocker_type": typ, "evidence_phrase": span, "owner_role": C.BLOCKER_TYPES[typ], "confidence": round(conf, 2)})
    return findings, dropped


def extract(note_text):
    """LLM extraction. Raises llm_client.LLMUnavailable if there is no key / network / cache entry."""
    raw, meta = L.chat_json(SYSTEM, f"NOTE:\n{note_text}")
    findings, dropped = validate(raw, note_text)
    return {"findings": findings, "dropped": dropped, "source": meta["source"], "model": f"{meta['provider']}/{meta['model']}"}


# ---------------------------------------------------------------- offline fallback (rules only, clearly labelled)
_KEYWORDS = {
    "pt_evaluation": r"(?:\bPT\b|\bOT\b|PT/OT|physical therapy|therapy)[^.;|]*?(?:pending|not (?:been )?(?:seen|completed|assessed|done)|awaiting|not yet|hasn't|no .*clearance)|no PT clearance",
    "transport": r"no ride|transport[^.;|]*(?:not|need)|(?:van|transport)[^.;|]*(?:not|no )",
    "home_oxygen_equipment": r"(?:home )?(?:O2|oxygen)[^.;|]*?(?:not|awaiting|still being|needs)|awaiting DME",
    "medications": r"prior auth[^.;|]*(?:pending|awaiting)|awaiting prior authorization|not yet (?:been )?filled|med rec(?:onciliation)? (?:is )?not|waiting on pharmacy",
    "family": r"(?:fam|family)[^.;|]*(?:not (?:done|taken)|has not)|no caregiver",
    "paperwork": r"paperwork[^.;|]*incomplete|referral (?:form )?(?:has not been signed|unsigned)|(?:form|auth)[^.;|]*(?:not signed|missing)",
    "placement": r"(?:SNF|skilled nursing|rehab\w*|facility)[^.;|]*?(?:pending|awaiting|not yet|no .*accept)|awaiting a bed offer",
}
_RESOLVED = re.compile(r"\b(?:completed|was completed|done|delivered|signed|approved|accepted|arranged|set up|filled|held|cleared)\b", re.I)


def rules_extract(note_text):
    findings = []
    for clause in re.split(r"(?<=[.;|])\s+|\s\|\s", note_text):
        if _RESOLVED.search(clause):
            continue
        for typ, pat in _KEYWORDS.items():
            m = re.search(pat, clause, flags=re.I)
            if m and typ not in {f["blocker_type"] for f in findings}:
                findings.append({"blocker_type": typ, "evidence_phrase": m.group(0).strip(), "owner_role": C.BLOCKER_TYPES[typ], "confidence": 0.5})
    return {"findings": findings, "dropped": [], "source": "rules-fallback", "model": "keyword rules (not an LLM)"}


# ---------------------------------------------------------------- scoring against the answer key
def score(notes, labels, results, meta=None):
    """notes: note_id,note_text. labels: answer key. results: {note_id: findings}. Type-level matching."""
    key = labels.groupby("note_id")["blocker_type"].apply(set).to_dict()
    rows = []
    for nid in notes["note_id"]:
        truth, pred = key.get(nid, set()), {f["blocker_type"] for f in results.get(nid, [])}
        rows.append({"note_id": nid, "tp": len(truth & pred), "fp": len(pred - truth), "fn": len(truth - pred),
                     "truth_n": len(truth), "pred_n": len(pred)})
    d = pd.DataFrame(rows)
    if meta is not None:
        d = d.merge(meta, on="note_id", how="left")
    tp, fp, fn = d.tp.sum(), d.fp.sum(), d.fn.sum()
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    out = {"notes": len(d), "blockers_planted": int(d.truth_n.sum()), "precision": prec, "recall": rec,
           "f1": 2 * prec * rec / (prec + rec) if prec + rec else float("nan"),
           "false_alarm_notes": int(((d.truth_n == 0) & (d.pred_n > 0)).sum()), "no_blocker_notes": int((d.truth_n == 0).sum())}
    if meta is not None and "has_resolved_distractor" in d:
        h = d[d.has_resolved_distractor == True]  # noqa: E712
        out["flagged_a_resolved_item"] = int(h.fp.gt(0).sum()); out["notes_with_resolved_item"] = len(h)
    if meta is not None and "style" in d:
        out["recall_by_style"] = {s: float(g.tp.sum() / max(1, g.tp.sum() + g.fn.sum())) for s, g in d.groupby("style")}
    return out, d


def run_set(notes, labels, meta, fn, label):
    results, errors = {}, 0
    import time as _t
    total, t0 = len(notes), _t.time()
    for i, (_, r) in enumerate(notes.iterrows(), 1):
        try:
            results[r["note_id"]] = fn(r["note_text"])["findings"]
            print(f"  [{i}/{total}] {r['note_id']} done ({_t.time() - t0:.0f}s elapsed)", flush=True)
        except L.LLMUnavailable as e:
            errors += 1
            print(f"  {r['note_id']}: unavailable ({e})")
            if errors >= 3:
                print("  stopping after 3 failures"); return None
    out, d = score(notes, labels, results, meta)
    wrong = d[(d.fp > 0) | (d.fn > 0)]
    if len(wrong):
        print(f"\n-- {label}: notes the AI got wrong (the first thing to look at) --")
        txt = notes.set_index("note_id")["note_text"].to_dict()
        keymap = labels.groupby("note_id")["blocker_type"].apply(set).to_dict()
        for nid in wrong["note_id"]:
            pred = {f["blocker_type"]: f.get("evidence_phrase", "") for f in results.get(nid, [])}
            truth = keymap.get(nid, set())
            for t in sorted(set(pred) - truth):
                print(f"  {nid} FALSE ALARM {t}: AI quoted {pred[t]!r}")
            for t in sorted(truth - set(pred)):
                print(f"  {nid} MISSED {t}")
            print(f"     note: {txt[nid][:260]}")
    print(f"\n== {label}: {out['notes']} notes, {out['blockers_planted']} planted blockers ==")
    print(f"precision {out['precision']:.2f} | recall {out['recall']:.2f} | F1 {out['f1']:.2f}")
    print(f"invented a blocker in {out['false_alarm_notes']} of {out['no_blocker_notes']} no-blocker notes")
    if "flagged_a_resolved_item" in out:
        print(f"false alarm in {out['flagged_a_resolved_item']} of {out['notes_with_resolved_item']} notes that mention a resolved item (check the list above: it may be a different item, not the resolved one)")
    if "recall_by_style" in out:
        print("recall by style:", {k: round(v, 2) for k, v in out["recall_by_style"].items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true"); ap.add_argument("--unseen", action="store_true"); ap.add_argument("--rules", action="store_true")
    a = ap.parse_args()
    if not a.score:
        ap.print_help(); return
    fn = rules_extract if a.rules else extract
    st = L.status()
    print("extractor:", "rules fallback" if a.rules else f"{st['provider']}/{st['model']} (key present: {st['key_present']})")
    notes = pd.read_csv(f"{C.DATA_DIR}/inpatient_notes.csv"); labels = pd.read_csv(f"{C.DATA_DIR}/note_labels.csv")
    meta = pd.read_csv(f"{C.DATA_DIR}/note_meta.csv")
    run_set(notes, labels, meta, fn, "main notes")
    if a.unseen:
        un = pd.read_csv(f"{C.DATA_DIR}/notes_unseen.csv"); ul = pd.read_csv(f"{C.DATA_DIR}/notes_unseen_labels.csv")
        run_set(un, ul, un[["note_id", "style", "has_resolved_distractor"]], fn, "unseen notes (different wording)")


if __name__ == "__main__":
    import extract_blockers
    extract_blockers.main()