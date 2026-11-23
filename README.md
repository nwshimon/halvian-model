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

No quotes, no spaces. The key is only needed for the language-model steps (3 and the Case demo tab). Everything else works without it, including the Patient journey tab.

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

1. **Case demo:** type a patient case in plain words (or pick example A, B, C or D). The AI fills a form, then shows the admit score with its math, the top 3 beds, discharge blockers for occupied beds, and a confirm/override step (an override needs a typed reason). Needs the key for the typed-case reading.
2. **Patient journey:** the story tab. Pick a patient (start with A, "Looks mild, then isn't") and click **Next** to step through their ED stay: triage, each order, each result, the admit decision. Every step rescores the patient with the real model and shows the facts known so far, the new score and why, and how the ED's total expected admissions and beds move. Needs no key.
3. **Prepare ahead:** the early-prep loop. Pick a moment in the demo day; see who is flagged and what the system would do (soft hold, early bed request, task chains for a discharge-pending bed, pooled holds for expected shortfalls), mark a task done to watch the chain re-time, and confirm the prompts (writes `bed_holds`, `tasks` and audit rows). Needs no key if the notes were read before (it reads the six discharge-pending notes with the language model, falling back to the keyword reader).
4. **Bed board:** the 40 beds and their status.
5. **Extractor test:** run the blocker reader on the 36 template notes or the 12 reworded ones, AI or keyword baseline.
6. **Audit trail:** every decision you made in the Case demo and Prepare ahead tabs.
7. **Fairness:** scores, boarding, walkouts and waits by payer group.

What to look for in the journey for patient A (a plain med-surg case): 14% at triage, 21% after the CT order, 77% when the lab result returns abnormal at 10:46, 79% after the CT result at 12:05, then the admit decision at 12:28. In Prepare ahead, A is flagged at 10:46 (a 102 minute lead), a ready med-surg bed is held and the bed request is ready to send; at 12:28 the hold converts.

## 5. Recording the demo video

Replay saved AI answers with no internet and no rate limits:

- Mac / Linux: `LLM_CACHE_ONLY=1 streamlit run app.py`
- Windows PowerShell: `$env:LLM_CACHE_ONLY="1"; streamlit run app.py`

Click through every Case demo example you plan to show once with the key first (so the answers are saved), then record in this mode. An uncached case shows an error in this mode instead of calling out. The Patient journey, Bed board, Audit trail and Fairness tabs never need the key.

## 6. Where things are

| Folder or file | Contents |
|---|---|
| `config.py` | Every setting, tagged [SPEC], [LIT] or [ASSUMPTION] |
| `data/` | Generated tables, notes, `fairness_report.csv`, `demo_manifest.json`, AI answer cache |
| `models/` | `admit_model.joblib` and `demand_forecaster.joblib` (the trained models; only load your own files), plus accuracy files |
| `app.py`, `case_engine.py` | The demo app and the logic behind the Case demo tab |
| `journey.py` | The step-by-step replay behind the Patient journey tab |
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
