# Boarding Command Center: MVP data model and demo

Everything is synthetic. Do not type real patient information anywhere.
Python 3.13. Run every command from inside this folder, with your virtual environment active.

## 0. One-time setup

```
pip install -r requirements.txt
```

Create a file named `.env` in this folder (copy `.env.example`):

```
GROQ_API_KEY=your_key_here
LLM_PROVIDER=groq
```

No quotes, no spaces. The key is only needed for the language-model steps (3 and "Read the case with AI"). Everything else works without it, including the Demo Scenarios tab.

## 1. Build the data and models (about 5 to 8 minutes, run once)

Run these in this exact order. Each prints a "PROBLEMS: none" style check at the end.

```
python generate_visits.py      # 85,000 ED visits over one year
python build_snapshots.py      # training rows (one per triage/order/result)
python admit_model.py          # trains the admit model
python demand_forecast.py      # trains the 2/4/8-hour forecaster
python build_demo.py           # the demo world: Nov 4 10:30, 30 patients, 40 beds
python build_snapshots.py      # run again: picks up the demo rows
python build_priors.py         # level-of-care, decision-time and isolation priors (from the training split only) -> data/priors.json
python scenario_model.py       # boarding-time scenarios and the model's lead time -> data/scenario_results.json
python build_notes.py          # 36 inpatient notes with an answer key
python build_unseen_notes.py   # 12 differently worded test notes
python fairness.py             # fairness report by payer group
```

Important: if you ever rerun `generate_visits.py`, rerun everything after it. It wipes the demo.

## 2. What each command shows you

| Command | What you should see |
|---|---|
| `python admit_model.py` | Accuracy (AUC about 0.89), calibration table, and a table of effects the model learned beside the true ones we planted |
| `python demand_forecast.py` | Forecast error vs a simple baseline (about 26% / 17% / 9% better at 2 / 4 / 8 hours) and a sample forecast |
| `python build_demo.py` | The score arc for patient A (14% at triage rising to 77% when the lab result returns), the payer twins scoring identically, bed recommendations for waiting patients |
| `python fairness.py` | Table by payer group. Look for FINDING (real gap), uncertain, or ok. Writes `data/fairness_report.csv` |
| `python build_priors.py` | The decision-time priors (minutes from a snapshot to the admit decision, by results pending) and level-of-care shares |
| `python scenario_model.py` | How early the model flags admits (at 60%: about half of admits, median lead about 111 min) and three boarding-time scenarios, clearly labelled as scenarios |
| `python test_early_prep.py` | 10 checks on the early-prep loop (no bed held twice, release on a drop, payer never used, tasks reach the right roles, and more). Should end `ALL PASSED` |
| `python extract_blockers.py --score --rules` | Keyword baseline with no AI: about 47% recall on the 36 notes. Needs no key |
| `python extract_blockers.py --score --unseen --rules` | Same baseline on the 12 reworded notes: 0% recall (this is why we use the language model) |

## 3. Test the language model (needs the key)

```
python extract_blockers.py --score --unseen
```

The first run calls Groq for all 48 notes and takes about 6 to 10 minutes. You will see a progress line per note (`[7/36] N-007 done (48s elapsed)`) and nothing else until it finishes, so do not stop it. Every answer is saved to `data/llm_cache.json`, so a rerun is instant.

Our results on Nov 22 (Groq, `openai/gpt-oss-120b`):

| Test | Found | False alarms |
|---|---|---|
| 36 template notes | 34 of 34 blockers | 0 of 9 no-blocker notes |
| 12 reworded notes | 11 of 11 blockers | 0 of 3 no-blocker notes |

Read these carefully: the instructions were tightened once after the first run, because the AI read "discharge orders to be entered; patient may go home today" as a blocker. That means the 36-note score is partly tuned. The 12 reworded notes were never used for tuning, but 12 is a small test. Report both as a first check.

If the scorer lists notes under `-- notes the AI got wrong`, those are the notes to look at first. If you change the instructions in `extract_blockers.py`, the saved answers no longer match and the next run calls Groq again.

Errors: `429` means a rate limit (it waits and retries by itself). `403` means access was refused (check the key, or try the curl test in section 7). `model not found` means set `LLM_MODEL` in `.env`.

## 4. Run the demo app

```
streamlit run app.py
```

Opens in your browser. Tabs:

1. **Simulate Patient Scenario (doctor / ED side):** type a case note (or pick example A, B, C or D, which fills every step without the AI key; "How do I write this?" shows the template). Step 1: the note and the form the AI filled. Step 2: order results (pending, reassuring or concerning, with free-text notes). Step 3: admit decision (level of care, isolation, approve) and **Submit Bed Request**, which puts the patient in the queue with a code such as SIM-001 and freezes the score panel until **New patient scenario**. A fixed panel on the right shows the admit score so far and how it changed as you edited the case. **Submit to AI Queue** reads the note with the AI and puts the patient on the Bed Board as a likely patient (before any admit decision), so soft and expedited holds appear as the score rises. It stays greyed out until you submit the bed request or start a new patient scenario (which treats the patient as having left). The key is only needed for the AI reading.
2. **Simulate Bed Assignments (nurse / bed management side):** the queue of submitted patients. Pick one, review the top 3 matching beds, check the discharge blockers of the selected bed, and **Confirm Bed Match** (a reason is required if you do not take the top pick). The bed is then treated as taken for the next queued patient and appears on the Bed Board.
3. **Demo Scenarios:** our seeded patients A, B and C replayed Time 1, 2, 3 and so on. At each time the doctor's view (left) and the bed manager's view (right) show only what was known at that moment. Needs no key.
4. **Bed Board:** a grid of all 40 beds by unit. Colour shows status, lighter tint means longer until ready, outlines show beds the early-prep loop holds, and indigo boxes are beds you assigned in the simulation. Hover a bed, or pick it below, for details, its blocker note, its task chain and a mark-done button. Pick a moment in the demo day with the slider.
**Sidebar pages (open in a new browser tab, bottom of the left sidebar):** **Extractor test** (run the blocker reader on the 36 template notes or the 12 reworded ones, AI or keyword baseline), **Audit trail** (every decision made in the app) and **Fairness check** (scores, boarding, walkouts and waits by payer group). They are the same app with `?page=extractor`, `?page=audit` or `?page=fairness` on the address.

What to look for in the journey for patient A (a plain med-surg case): 14% at triage, 21% after the CT order, 77% when the lab result returns abnormal at 10:46, 79% after the CT result at 12:05, then the admit decision at 12:28. In Prepare ahead, A is flagged at 10:46 (a 102 minute lead), a ready med-surg bed is held and the bed request is ready to send; at 12:28 the hold converts.

## 5. Recording the demo video

Replay saved AI answers with no internet and no rate limits:

- Mac / Linux: `LLM_CACHE_ONLY=1 streamlit run app.py`
- Windows PowerShell: `$env:LLM_CACHE_ONLY="1"; streamlit run app.py`

Click "Read the case with AI" on every example you plan to show once with the key first (so the answers are saved), then record in this mode. An uncached case shows an error in this mode instead of calling out. Demo Scenarios, Bed Board, Audit trail and Fairness never need the key.

## 6. Where things are

| Folder or file | Contents |
|---|---|
| `config.py` | Every setting, tagged [SPEC], [LIT] or [ASSUMPTION] |
| `data/` | Generated tables, notes, `fairness_report.csv`, `demo_manifest.json`, AI answer cache |
| `models/` | `admit_model.joblib` and `demand_forecaster.joblib` (the trained models; only load your own files), plus accuracy files |
| `app.py`, `case_engine.py` | The demo app and the logic behind the Simulate Scenario tab |
| `journey.py` | The step-by-step replay behind the Demo Scenarios tab |
| `early_prep.py`, `build_priors.py`, `scenario_model.py` | The early-prep loop engine (tiered holds, task chains, pooled holds), its priors and the boarding-time scenarios |
| `test_early_prep.py` | The 10 loop checks |
| `intake.py`, `extract_blockers.py`, `llm_client.py` | The language-model readers and client |
| `fairness.py` | The fairness report |

## 7. If something breaks

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` with the virtual environment active |
| `No such file` for data or models | Do step 1 in order |
| `no API key` | Check `.env` is in this folder and says `GROQ_API_KEY=...` |
| `429` / rate limit | Wait a minute, rerun; saved answers are reused |
| `403` | Test the key: `curl -s -o /dev/null -w "%{http_code}\n" https://api.groq.com/openai/v1/models -H "Authorization: Bearer YOUR_KEY"`. `200` means the key is fine; `401` or `403` means make a new key. Gemini works as a fallback: set `GEMINI_API_KEY` and `LLM_PROVIDER=gemini` |
| `model not found` | Set `LLM_MODEL=` in `.env` to a model name from your Groq console |
| Looks frozen after the extractor command | It is not: 48 notes take 6 to 10 minutes the first time. Watch for the progress lines |
| App shows old numbers after rerunning data | Stop Streamlit (Ctrl+C) and start it again |
| `demo_manifest.json` errors | Rerun `python build_demo.py` |