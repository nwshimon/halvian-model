"""
make_deploy_folder.py -- builds ./deploy/, a small copy of the app you can push to GitHub and host.

  python make_deploy_folder.py

It copies the code and trained models, keeps only the DEMO rows of the big history files (so the folder is a few MB, not 130 MB),
copies the notes, fairness report and saved AI answers, and never copies your .env or any key.
The full history stays on your computer; the hosted app does not need it.
"""
import os
import shutil

import pandas as pd

OUT = "deploy"
CODE = ["app.py", "case_engine.py", "journey.py", "intake.py", "extract_blockers.py", "llm_client.py", "fairness.py",
        "admit_model.py", "bed_rules.py", "config.py", "schema.py", "requirements.txt", "demand_forecast.py", ".gitignore"]
SMALL = ["beds.csv", "demo_manifest.json", "inpatient_notes.csv", "note_labels.csv", "note_meta.csv", "notes_unseen.csv",
         "notes_unseen_labels.csv", "occupancy_hourly.csv", "fairness_report.csv", "llm_cache.json"]
BIG = ["visits.csv", "events.csv", "snapshots.csv", "vitals.csv"]

if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(f"{OUT}/data"); os.makedirs(f"{OUT}/models")
for f in CODE:
    if os.path.exists(f):
        shutil.copy(f, f"{OUT}/{f}")
for f in os.listdir("models"):
    if f.endswith((".joblib", ".json")):
        shutil.copy(f"models/{f}", f"{OUT}/models/{f}")
for f in SMALL:
    if os.path.exists(f"data/{f}"):
        shutil.copy(f"data/{f}", f"{OUT}/data/{f}")
demo_ids = None
for f in BIG:
    if not os.path.exists(f"data/{f}"):
        continue
    df = pd.read_csv(f"data/{f}")
    if f == "visits.csv":
        demo_ids = set(df.loc[df["split"] == "demo", "visit_id"])
    keep = df[df["visit_id"].isin(demo_ids)] if "visit_id" in df and demo_ids else df.head(0)
    keep.to_csv(f"{OUT}/data/{f}", index=False)
    print(f"{f}: kept {len(keep):,} of {len(df):,} rows")

with open(f"{OUT}/.gitignore", "w") as g:
    g.write(".env\n.streamlit/secrets.toml\n__pycache__/\n")
size = sum(os.path.getsize(os.path.join(r, x)) for r, _, fs in os.walk(OUT) for x in fs) / 1e6
print(f"\nwrote ./{OUT}/  ({size:.1f} MB)  -> push THIS folder to GitHub, not your full project")