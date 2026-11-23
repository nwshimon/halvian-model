"""
config.py -- every design decision for the Halvian ED Boarding MVP, in one place.

The simulator, feature builder, model, bed matcher and fairness module all READ
from this file. To change a decision, change it here and re-run.

Source tags on every parameter:
  [BRIEF]       number given in the case brief (as quoted in the team spec)
  [SPEC]        number or rule proposed in the team spec
  [LIT]         shape taken from published data (source named in the comment)
  [ASSUMPTION]  our own placeholder. Replace if a clinician or source gives better.

Run `python config.py` to validate (sums to 1, brief baselines reproduced, etc.).

LOCKED 2026-11-22 (do not change without telling the team):
  - History = one full year (2025-11-01, 365 days, ~85,000 visits). HISTORY_START_DATE / HISTORY_DAYS
    are the inputs; the visit count is derived from the brief's annual volume, not typed in.
  - Winter respiratory season (+20% volume, Dec-Mar) is built into the daily volumes.

STILL OPEN (defaults below are my recommendations; search this file for "DECISION"):
  1. PLANT_PAYER_BOARDING_GAP        -- should the fairness tab have a real gap to find?
  2. INCLUDE_PEDIATRIC               -- are patients under 18 in scope?
  3. VITAL_ABNORMAL                  -- DONE 2026-11-22: published NEWS2 normal bands
  4. PAYER_MIX["other"]              -- spec's payer mix sums to 98%; I added 2% "other"
"""

# ---------------------------------------------------------------------------
# 0. Run settings
# ---------------------------------------------------------------------------
RANDOM_SEED = 42                      # [ASSUMPTION] fixed so results are reproducible
HISTORY_START_DATE = "2025-11-01"     # [ASSUMPTION] LOCKED 2026-11-22: history is ONE FULL YEAR at the brief's
                                      #   volume, so every season shows. The uninsured test group is then ~1,360
                                      #   visits / ~218 admits (the old 40,000-visit plan covered only ~172 days: ~640 / ~100).
HISTORY_DAYS = 365                    # [LOCKED] 2025-11-01 to 2026-10-31 inclusive
DEMO_DATE = "2026-11-04"              # [ASSUMPTION] a Wednesday after the history window; cosmetic, for demo timestamps
N_DEMO_PATIENTS = 30                  # [SPEC] about 30 ED patients for the demo
N_SCRIPTED_TIMELINES = 3              # [SPEC] three scripted patients for the live demo
SPLIT = {"train": 0.60, "validation": 0.20, "test": 0.20}   # [ASSUMPTION] split by visit, never by snapshot row
DATA_DIR = "data"                     # where generated CSVs are written

# ---------------------------------------------------------------------------
# 1. Hospital baselines (the generator must reproduce these)
# ---------------------------------------------------------------------------
ED_VISITS_PER_YEAR = 85_000           # [BRIEF]
N_HISTORY_VISITS_EXPECTED = round(ED_VISITS_PER_YEAR * HISTORY_DAYS / 365)   # derived: 85,000. Not a free parameter
ADMIT_RATE_TARGET = 0.16              # [BRIEF] 16% of visits are admitted (~37 admits/day)
MEDIAN_BOARDING_HOURS = 3.5           # [BRIEF] admit decision -> leaves ED
TREATMENT_ROOMS = 32                  # [BRIEF]
INPATIENT_BEDS = 320                  # [BRIEF]
INPATIENT_OCCUPANCY = 0.88            # [BRIEF] about 40 beds open on average
DOOR_TO_PROVIDER_MIN_OVERALL = 55     # [BRIEF] median minutes
LWBS_RATE_OVERALL = 0.04              # [BRIEF] 4.0% leave without being seen

# Payer mix. [SPEC] gives 35 / 22 / 33 / 8 = 98%. "other" is the missing 2%.
PAYER_MIX = {
    "medicaid_chip": 0.35,            # [SPEC]
    "medicare":      0.22,            # [SPEC]
    "commercial":    0.33,            # [SPEC]
    "uninsured":     0.08,            # [SPEC]
    "other":         0.02,            # [ASSUMPTION] DECISION: remainder so the mix sums to 1
}
# Median door-to-provider minutes by payer.
DOOR_TO_PROVIDER_MIN_BY_PAYER = {
    "uninsured":     64,              # [BRIEF] outcome-gaps table
    "medicaid_chip": 58,              # [BRIEF] outcome-gaps table
    "commercial":    48,              # [BRIEF]
    "medicare":      55,              # [ASSUMPTION] set to overall; not in the spec
    "other":         55,              # [ASSUMPTION]
}
# Leave-without-being-seen rate by payer.
LWBS_BY_PAYER = {
    "uninsured":     0.058,           # [BRIEF]
    "commercial":    0.030,           # [BRIEF]
    "medicaid_chip": 0.046,           # [BRIEF] outcome-gaps table (was 0.048, which contradicted the brief)
    "medicare":      0.039,           # [ASSUMPTION] solved so the weighted overall is 4.0% given the brief's other values
    "other":         0.040,           # [ASSUMPTION]
}

# ---------------------------------------------------------------------------
# 2. Who arrives, and when (generated in this causal order:
#    age -> complaint -> vitals -> ESI -> orders -> results -> admit)
# ---------------------------------------------------------------------------
INCLUDE_PEDIATRIC = True              # DECISION: [ASSUMPTION] brief/spec only say "age band"; set False if adults only

AGE_BANDS = ["0-17", "18-34", "35-49", "50-64", "65-74", "75+"]
AGE_MIX = {"0-17": 0.18, "18-34": 0.25, "35-49": 0.17,
           "50-64": 0.18, "65-74": 0.11, "75+": 0.11}      # [ASSUMPTION]
# If INCLUDE_PEDIATRIC is False the generator drops "0-17" and renormalises.

COMPLAINT_MIX = {                     # [ASSUMPTION] base mix before the age tilt below
    "chest_pain": 0.10, "shortness_of_breath": 0.08, "abdominal_pain": 0.12,
    "infection_fever": 0.09, "injury_trauma": 0.20, "neuro": 0.07,
    "psychiatric": 0.05, "minor_other": 0.17, "other": 0.12,
}
# Multipliers applied to COMPLAINT_MIX, then renormalised. Direction is clinically
# sensible (older patients: more cardiac/respiratory/neuro); sizes are [ASSUMPTION].
COMPLAINT_AGE_TILT = {
    "0-17":  {"injury_trauma": 1.3, "infection_fever": 1.3, "chest_pain": 0.3, "neuro": 0.5},
    "65-74": {"chest_pain": 1.5, "shortness_of_breath": 1.8, "neuro": 1.4, "infection_fever": 1.3, "minor_other": 0.5},
    "75+":   {"chest_pain": 1.5, "shortness_of_breath": 1.8, "neuro": 1.4, "infection_fever": 1.3, "minor_other": 0.5},
}

# Hourly arrival weights, hour 0..23 (midnight first). Low overnight, two peaks.
# [BRIEF] "Arrivals peak on weekday mornings and evenings" -> peaks at 10:00 and 18:00, dip at 14:00.
# [ASSUMPTION] exact heights; normalised below. Daily total = ED_VISITS_PER_YEAR / 365.
_ARRIVAL_HOURLY_RAW = [0.022, 0.018, 0.015, 0.014, 0.014, 0.017, 0.026, 0.040,
                       0.058, 0.072, 0.074, 0.068, 0.062, 0.058, 0.056, 0.058,
                       0.064, 0.072, 0.074, 0.066, 0.054, 0.042, 0.032, 0.026]
_DAY_OF_WEEK_RAW = [1.10, 1.00, 0.98, 0.97, 1.00, 0.95, 0.95]   # Mon..Sun [ASSUMPTION]


def _normalize(values):
    total = sum(values)
    return [v / total for v in values]


ARRIVAL_HOURLY_WEIGHTS = _normalize(_ARRIVAL_HOURLY_RAW)
DAY_OF_WEEK_WEIGHTS = _normalize(_DAY_OF_WEEK_RAW)
MEAN_VISITS_PER_DAY = ED_VISITS_PER_YEAR / 365          # ~233 (annual mean)

# Winter respiratory season. [BRIEF] "adds about 20% to volume".
# The annual total must stay 85,000, so the generator scales non-winter days DOWN so that
# (winter days x 1.20 + other days x 1.0) still averages MEAN_VISITS_PER_DAY.
WINTER_VOLUME_MULTIPLIER = 1.20       # [BRIEF]
WINTER_MONTHS = (12, 1, 2, 3)         # [ASSUMPTION] brief does not say which months
WINTER_COMPLAINT_TILT = {"shortness_of_breath": 1.4, "infection_fever": 1.4}   # [ASSUMPTION] applied in winter months


def expected_daily_visits():
    """{date: expected visits} across the history window.
    Shape = day-of-week weights x winter multiplier, scaled so the window totals exactly
    N_HISTORY_VISITS_EXPECTED. The generator draws each day's arrivals around these values,
    then spreads them over the day with ARRIVAL_HOURLY_WEIGHTS."""
    from datetime import date, timedelta
    start = date.fromisoformat(HISTORY_START_DATE)
    days = [start + timedelta(days=i) for i in range(HISTORY_DAYS)]
    raw = {d: 7 * DAY_OF_WEEK_WEIGHTS[d.weekday()]
              * (WINTER_VOLUME_MULTIPLIER if d.month in WINTER_MONTHS else 1.0) for d in days}
    scale = N_HISTORY_VISITS_EXPECTED / sum(raw.values())
    return {d: v * scale for d, v in raw.items()}

# ---------------------------------------------------------------------------
# 3. Vitals
# ---------------------------------------------------------------------------
VITAL_BASELINE = {                    # (mean, sd) in a typical adult ED patient [ASSUMPTION]
    "hr": (82, 14), "spo2": (97.5, 1.5), "sbp": (130, 20), "rr": (17, 3), "temp_c": (36.8, 0.5),
}
# Shifts added to the baseline mean for each complaint. [ASSUMPTION]
VITAL_SHIFT_BY_COMPLAINT = {
    "chest_pain":          {"hr": 5},
    "shortness_of_breath": {"spo2": -2.5, "rr": 4, "hr": 8},
    "infection_fever":     {"temp_c": 1.0, "hr": 15, "rr": 3},
    "abdominal_pain":      {"hr": 4},
}
# A hidden per-visit "severity" (standard normal) is the main reason vitals, ESI,
# orders and admission correlate with each other. Never shown to the model.
LATENT_SEVERITY_SD = 1.0              # [ASSUMPTION]
SEVERITY_LOADING_ON_VITALS = {"hr": 6.0, "spo2": -1.2, "sbp": -6.0, "rr": 2.0, "temp_c": 0.2}  # [ASSUMPTION]

# Abnormal-vital flag = the vital falls OUTSIDE the "normal" band of the Royal College of Physicians'
# NEWS2 chart (a 0 score on that vital). (low, high) inclusive; None = no limit on that side.   [LIT] NEWS2, RCP 2017:
#   heart rate 51-90 | systolic BP 111-219 | respiratory rate 12-20 | SpO2 (scale 1) >= 96 | temperature 36.1-38.0 C
# SIRS (a different, older sepsis screen) uses HR > 90, RR > 20, temp > 38 or < 36, so the two agree on the high side.
VITAL_ABNORMAL = {
    "hr": (51, 90), "spo2": (96, None), "sbp": (111, 219), "rr": (12, 20), "temp_c": (36.1, 38.0),
}

# ---------------------------------------------------------------------------
# 4. ESI (triage level 1 = most urgent ... 5 = least urgent)
# ---------------------------------------------------------------------------
ESI_MIX = {1: 0.02, 2: 0.17, 3: 0.45, 4: 0.28, 5: 0.08}   # [ASSUMPTION] lower-acuity than Stony Brook's
                                                          #   so the overall rate lands near 16%, not 23.6%
# ESI is assigned by ranking this score and cutting at the ESI_MIX quantiles. [ASSUMPTION]
ESI_SCORE_WEIGHTS = {"severity": 1.0, "abnormal_vitals": 0.5, "age_75plus": 0.3, "noise_sd": 0.7}

# Admit rate by ESI -- SHAPE only. [LIT] Stony Brook ED poster, 737,584 visits, one
# hospital, 2018-2024: 71.6% / 40.3% / 16.5% / 1.9% / 0.6% for ESI 1-5 (23.6% overall).
# Levels differ by hospital, so the generator shifts an intercept to hit ADMIT_RATE_TARGET.
ESI_ADMIT_RATE_SHAPE = {1: 0.72, 2: 0.40, 3: 0.165, 4: 0.019, 5: 0.006}

# ---------------------------------------------------------------------------
# 5. TRUE effects in the simulated world (log-odds added on top of the ESI shape)
#    These are the "ground truth" the trained model should RECOVER (Step 3 check).
#    Directions: [LIT] older and abnormal-vital patients are admitted more, beyond ESI
#    (Brazil triage study; adult demographic-factors study). Sizes: [ASSUMPTION].
# ---------------------------------------------------------------------------
ADMIT_LOGODDS = {
    "age_band": {"0-17": -0.8, "18-34": -0.4, "35-49": 0.0, "50-64": 0.4, "65-74": 0.8, "75+": 1.2},
    "complaint": {"chest_pain": 0.5, "shortness_of_breath": 0.7, "abdominal_pain": 0.2,
                  "infection_fever": 0.4, "injury_trauma": -0.5, "neuro": 0.3,
                  "psychiatric": 0.0, "minor_other": -1.0, "other": 0.0},
    "per_abnormal_vital": 0.35,
    "per_order": 0.12,
    "per_abnormal_result": 0.80,
    "severity": 0.50,
    "inpatient_occupancy": 0.0,       # deliberate NULL effect: the model should learn ~0 for this
    "noise_sd": 0.60,                 # random noise so AUC lands in the high 0.8s, not ~1.0
}
# DECISION: a hidden real-world payer effect on admission, never given to the model.
# Stony Brook shows lower admission for self-pay at the same ESI (observational).
PAYER_ADMIT_LOGODDS = {"medicaid_chip": 0.0, "medicare": 0.0, "commercial": 0.0,
                       "uninsured": 0.0, "other": 0.0}   # [ASSUMPTION] off; e.g. set uninsured to -0.5 to test proxies

# ---------------------------------------------------------------------------
# 6. Orders and results (time in minutes)
# ---------------------------------------------------------------------------
ORDER_BASE_PROB = {                   # probability a visit gets this order, at ESI 3 [ASSUMPTION]
    "ECG": 0.25, "lab_panel": 0.55, "troponin": 0.18, "chest_xray": 0.22,
    "CT": 0.15, "ultrasound": 0.07, "IV_medication": 0.30,
}
ORDER_ESI_MULTIPLIER = {1: 1.6, 2: 1.4, 3: 1.0, 4: 0.5, 5: 0.2}   # [ASSUMPTION]
ORDER_DELAY_AFTER_PROVIDER_MIN = (2, 40)           # [ASSUMPTION] uniform range
RESULT_LATENCY_MIN = {                             # [ASSUMPTION] (min, max)
    "ECG": (5, 15), "lab_panel": (30, 70), "troponin": (30, 70), "chest_xray": (30, 60),
    "CT": (60, 120), "ultrasound": (40, 90), "IV_medication": (0, 0),
}
RESULT_ABNORMAL_BASE_PROB = 0.20      # [ASSUMPTION] rises with severity
REASSESSMENT_INTERVAL_MIN = (60, 120) # [ASSUMPTION] nurse re-checks vitals this often

# ---------------------------------------------------------------------------
# 7. Boarding (admit decision -> leaves ED)
# ---------------------------------------------------------------------------
BOARDING_SIGMA = 0.60                 # [ASSUMPTION] lognormal spread; median = MEDIAN_BOARDING_HOURS
# DECISION: the brief gives no boarding time by payer. False = no gap (fairness tab shows
# "no gap flagged"). True = plant a modest gap so the detector has something to find.
PLANT_PAYER_BOARDING_GAP = False
BOARDING_PAYER_MULTIPLIER = {         # [ASSUMPTION] used only if PLANT_PAYER_BOARDING_GAP is True
    "uninsured": 1.15, "medicaid_chip": 1.08, "commercial": 1.00, "medicare": 1.00, "other": 1.00,
}

# ---------------------------------------------------------------------------
# 8. The admit model
# ---------------------------------------------------------------------------
MODEL_FEATURES = [                    # every feature must be knowable at the snapshot time [SPEC]
    "age_band", "esi", "complaint_category",
    "max_hr_so_far", "min_spo2_so_far", "min_sbp_so_far", "max_rr_so_far", "max_temp_so_far",
    "abnormal_vital_count_so_far",
    "orders_so_far", "ct_orders_so_far", "abnormal_results_so_far", "pending_results_so_far",
    "mins_since_arrival", "mins_since_last_reassessment",
    "inpatient_occupancy_at_snapshot",
]
# Never allowed as features. payer/language/race: fairness only [SPEC]. The rest are
# the label or happen after it (leakage).
PROHIBITED_FEATURES = [
    "payer", "language", "race", "ethnicity",
    "disposition", "disposition_time", "requested_level_of_care", "admitted",
    "bed_requested_time", "bed_matched_time", "bed_ready_time", "left_ed_time", "boarding_hours",
]
LOGREG_C = 1.0                        # [ASSUMPTION] regularisation strength; tune on validation
EXPECTED_AUC_RANGE = (0.85, 0.93)     # [LIT] a published study reached ~0.91 with full variables;
                                      #   outside this range, suspect the generator or a leak

# Category thresholds on the admit probability. [ASSUMPTION] tune on the validation set.
CATEGORY_THRESHOLDS = {"possible": 0.20, "likely": 0.60}    # <0.20 unlikely, 0.20-0.60 possible, >=0.60 likely
SCORE_ON_EVENTS = ["triage", "order", "result"]             # [SPEC] re-score on these snapshot triggers
TOP_FACTORS_SHOWN = 3                 # [SPEC]

# Demand forecast = sum of admit probabilities; variance = sum of p(1-p). [SPEC]
FORECAST_HORIZONS_HOURS = (2, 4, 8)   # [SPEC]
FORECAST_INTERVAL_Z = 1.645           # 90% range

# ---------------------------------------------------------------------------
# 9. Alerts and stalls (minutes allowed in each boarding stage before "stalled")
# ---------------------------------------------------------------------------
ALERT_ON_CATEGORY_CHANGE = True       # [SPEC] alert only when category, bed type changes or a stage stalls
STALL_LIMIT_MIN = {                   # [ASSUMPTION]
    "admit_decided_to_bed_requested": 15,
    "bed_requested_to_bed_matched": 30,
    "bed_matched_to_bed_ready": 270,   # was 45; at a 3.5 h median boarding the bed-ready wait is the long stage
    "bed_ready_to_report_given": 20,
    "report_given_to_transported": 20,
}

# ---------------------------------------------------------------------------
# 10. Beds and bed matching (rules + weights chosen by hand; auditable) [SPEC]
# ---------------------------------------------------------------------------
BEDS_DEMO = {                         # 40-bed slice for the demo [SPEC]; [ASSUMPTION] composition
    "unit_counts": {"med_surg": 22, "telemetry": 12, "icu": 6},
    "isolation_capable": {"med_surg": 3, "telemetry": 2, "icu": 2},
    # Status mix sums to 40. Note: 30/40 occupied = 75%, lower than the hospital's 88%,
    # so the demo has beds to match. Say so if asked.
    "status_counts": {"occupied": 30, "cleaning": 4, "ready": 4, "blocked": 2},
    "occupied_with_discharge_pending": 6,   # shown amber on the board (a display state, not a bed status)
}
BED_STATUSES = ["occupied", "cleaning", "ready", "blocked"]     # [SPEC]

MATCH_HARD_FILTERS = [                # a bed failing any of these is never recommended
    "bed_type_can_serve_level_of_care", "isolation_if_needed", "status_ready_or_freeing_soon",
]
MATCH_WEIGHTS = {                     # [ASSUMPTION] score = sum(weight * term); highest wins
    "ready_now": 3.0,
    "minutes_until_ready": -0.03,     # per minute: sooner is better
    "exact_level_of_care": 2.0,       # exact type beats stepping up (e.g. ICU bed for med-surg)
    "preferred_unit": 1.0,
    "isolation_bed_wasted": -0.5,     # penalty for using an isolation-capable bed when not needed
}
MATCH_REASON_CODES = {                # shown next to each recommendation [SPEC: reasoning visible]
    "ready_now": "Bed is ready now",
    "minutes_until_ready": "Becomes ready soonest",
    "exact_level_of_care": "Exact level of care match",
    "preferred_unit": "In the preferred unit",
    "isolation_bed_wasted": "Uses a scarce isolation bed",
}
MATCH_TOP_N = 3                       # [SPEC] best match plus two alternatives

# ---------------------------------------------------------------------------
# 11. Discharge-blocker extraction (LLM) and its test notes
# ---------------------------------------------------------------------------
BLOCKER_TYPES = {                     # closed list [SPEC]; owner roles [ASSUMPTION]
    "pt_evaluation":        "PT/OT",
    "transport":            "case_management",
    "home_oxygen_equipment": "case_management",
    "medications":          "pharmacy",
    "family":               "social_work",
    "paperwork":            "case_management",
    "placement":            "case_management",
}
N_NOTES_TOTAL = 36                    # [ASSUMPTION] evaluation set; no training, test only
N_NOTES_DEMO = 12                     # [SPEC] about 12 notes shown in the demo
NOTE_NO_BLOCKER_FRACTION = 0.25       # [ASSUMPTION] notes with no blocker, so the test can fail
NOTE_MESSY_FRACTION = 0.35            # [ASSUMPTION] hand-written style: abbreviations, fragments
EXTRACTION_REQUIRE_VERBATIM_EVIDENCE = True   # [SPEC] evidence phrase must appear word-for-word in the note
EXTRACTION_OUTPUT_FIELDS = ["blocker_type", "evidence_phrase", "owner_role", "confidence"]

# LLM provider settings (used by llm_client.py). All [ASSUMPTION]: check model names in your own console before recording.
# The code reads the key from an environment variable; never put a key in this file.
LLM_PROVIDERS = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "key_env": "GEMINI_API_KEY",
               "default_model": "gemini-3.8-flash"},
    "groq":   {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY",
               "default_model": "openai/gpt-oss-120b"},
}
LLM_DEFAULT_PROVIDER = "gemini"        # override with env var LLM_PROVIDER; model with LLM_MODEL
LLM_TIMEOUT_S = 40
LLM_MAX_RETRIES = 3                    # on rate limit (429) or server error, with backoff
LLM_MIN_SECONDS_BETWEEN_CALLS = 2.5    # paces calls so free-tier per-minute caps are not hit
LLM_MIN_SECONDS_BY_PROVIDER = {"groq": 8.0}   # Groq free tier ~8K tokens/min = about 8 notes a minute
LLM_CACHE_PATH = "data/llm_cache.json" # every answer is cached so the demo can replay offline

# ---------------------------------------------------------------------------
# 12. Fairness
# ---------------------------------------------------------------------------
FAIRNESS_TOLERANCE = 0.10             # [SPEC] proposal: flag if a group differs >10% from overall
FAIRNESS_GROUPS = ["medicaid_chip", "medicare", "commercial", "uninsured"]  # "other" too small to test
FAIRNESS_METRICS = [                  # [ASSUMPTION] pick and justify 1-2 in the deck
    "mean_admit_score", "false_negative_rate", "median_boarding_hours",
]
# Flag rule: abs(group_value / overall_value - 1) > FAIRNESS_TOLERANCE


# ---------------------------------------------------------------------------
# 13. Step 1 generator knobs (added 2026-11-22). All [ASSUMPTION] unless noted.
# ---------------------------------------------------------------------------
TRIAGE_DELAY_MIN = (3, 15)            # arrival -> triage (uniform)
DOOR_TO_PROVIDER_SIGMA = 0.55         # lognormal spread of door-to-provider; payer medians come from the brief
D2P_ESI_FACTOR = {1: 0.10, 2: 0.45, 3: 1.0, 4: 1.0, 5: 1.0}   # sicker patients are seen sooner; payer medians
                                                              # are re-centred afterwards so the brief still holds
LWBS_ELIGIBLE_ESI = (3, 4, 5)         # ESI 1-2 never leave without being seen; LWBS probability is scaled up
                                      # by 1 / (share of visits that are eligible) so the payer rates still hold
LWBS_LEAVE_FACTOR = (0.8, 1.6)        # an LWBS patient gives up after (would-be door-to-provider) x this
ADMIT_DECISION_DELAY_MIN = (10, 45)   # last result (or provider visit) -> admit decision
DISCHARGE_DELAY_MIN = (20, 90)        # last result (or provider visit) -> discharged patient leaves
# Boarding total = lognormal(median MEDIAN_BOARDING_HOURS, BOARDING_SIGMA). It is split across the five stages
# with Dirichlet weights; the bed-ready wait dominates, as in real boarding.
BOARDING_STAGE_ORDER = ["bed_requested", "bed_matched", "bed_ready", "report_given", "transported"]
BOARDING_STAGE_ALPHA = [0.4, 1.0, 12.0, 0.8, 0.8]   # medians about 2 / 10 / 167 / 7 / 7 min of a 3.5 h boarding
BOARDING_MIN_STAGE_MIN = 1.0          # no stage shorter than this
VITAL_REPEAT_SD_SHARE = 0.7           # share of each vital's baseline SD that is a fixed per-patient offset
VITAL_NOISE_SD_SHARE = 0.3            # per-measurement noise
VITAL_LIMITS = {"hr": (30, 190), "spo2": (70, 100), "sbp": (60, 240), "rr": (6, 45), "temp_c": (34.0, 41.5)}
RESULT_ABNORMAL_SEVERITY_SLOPE = 0.60 # log-odds per unit severity on top of RESULT_ABNORMAL_BASE_PROB
ORDERS_WITHOUT_RESULT = ("IV_medication",)   # an IV medication order has no result event
LEVEL_OF_CARE_SHARE = {"icu": 0.15, "telemetry": 0.30, "med_surg": 0.55}   # matches BEDS_DEMO unit_counts ratios
ISOLATION_PROB_ADMIT = {"infection_fever": 0.30, "shortness_of_breath": 0.10, "default": 0.02}
# DECISION: complaints with a preferred inpatient unit. Names must equal beds.unit in Step 5.
PREFERRED_UNIT_BY_COMPLAINT = {"chest_pain": "cardiac", "neuro": "neuro", "shortness_of_breath": "pulmonary"}
OCCUPANCY_DAILY_AMPLITUDE = 0.015     # afternoon peak above the mean
OCCUPANCY_AR1 = 0.995                 # hour-to-hour persistence
OCCUPANCY_NOISE_SD = 0.0035           # AR(1) shock SD; the stationary SD is about 0.035
OCCUPANCY_LIMITS = (0.70, 0.99)

# ---- demo day (added 2026-11-22) ----
DEMO_NOW_TIME = "10:30"               # [ASSUMPTION] the demo's "now" on DEMO_DATE: Wednesday morning peak
DEMO_POOL_SEED = 777                  # a second generator run supplies realistic demo patients, then they are time-shifted
# The 27 background patients (N_DEMO_PATIENTS minus the 3 scripted ones), by where they are at DEMO_NOW. [ASSUMPTION]
DEMO_ARCHETYPES = {
    "waiting_for_provider": 6,        # triaged, waiting to be seen
    "in_workup": 11,                  # orders placed or results coming back; includes the 2 payer-twin patients
    "awaiting_discharge": 4,          # results back, about to leave
    "boarding_decided": 2,            # admit decision made, bed not yet requested
    "boarding_requested": 4,          # bed requested, waiting for a match
}
DEMO_WORKUP_BANDS = {"unlikely": 3, "possible": 4, "likely": 4}    # admit-score category of the 11 in_workup patients at DEMO_NOW
DEMO_BOARDING_LEVELS = {"boarding_decided": ["med_surg", "telemetry"],
                        "boarding_requested": ["icu", "telemetry", "med_surg", "med_surg"]}
DEMO_OCCUPANT_WINDOW_DAYS = 5         # inpatients in the 30 occupied beds were admitted from the ED in the last N days of history


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _close(a, b, tol):
    return abs(a - b) <= tol


def validate_config():
    """Raise AssertionError if the config contradicts itself or the brief."""
    assert sum(DEMO_ARCHETYPES.values()) == N_DEMO_PATIENTS - N_SCRIPTED_TIMELINES, "demo archetypes must add up to N_DEMO_PATIENTS - scripted"
    assert sum(DEMO_WORKUP_BANDS.values()) == DEMO_ARCHETYPES["in_workup"], "workup bands must add up to the in_workup count"
    for k, lv in DEMO_BOARDING_LEVELS.items():
        assert len(lv) == DEMO_ARCHETYPES[k], f"DEMO_BOARDING_LEVELS[{k}] must list one level per patient"
    assert _close(sum(SPLIT.values()), 1.0, 1e-9), "SPLIT must sum to 1"
    for name, mix in [("PAYER_MIX", PAYER_MIX), ("AGE_MIX", AGE_MIX),
                      ("COMPLAINT_MIX", COMPLAINT_MIX), ("ESI_MIX", ESI_MIX)]:
        assert _close(sum(mix.values()), 1.0, 1e-9), f"{name} must sum to 1 (got {sum(mix.values()):.4f})"
    assert _close(sum(ARRIVAL_HOURLY_WEIGHTS), 1.0, 1e-9)
    assert len(ARRIVAL_HOURLY_WEIGHTS) == 24 and len(DAY_OF_WEEK_WEIGHTS) == 7
    assert set(AGE_MIX) == set(AGE_BANDS) == set(ADMIT_LOGODDS["age_band"])
    assert set(COMPLAINT_MIX) == set(ADMIT_LOGODDS["complaint"])
    assert set(PAYER_MIX) == set(LWBS_BY_PAYER) == set(DOOR_TO_PROVIDER_MIN_BY_PAYER) \
        == set(BOARDING_PAYER_MULTIPLIER) == set(PAYER_ADMIT_LOGODDS)

    # The ESI shape should land near the brief's overall admit rate before intercept tuning.
    shape_rate = sum(ESI_MIX[k] * ESI_ADMIT_RATE_SHAPE[k] for k in ESI_MIX)
    assert _close(shape_rate, ADMIT_RATE_TARGET, 0.02), f"ESI shape gives {shape_rate:.3f} vs target"

    # Weighted LWBS across payers should reproduce the brief's 4.0%.
    lwbs = sum(PAYER_MIX[p] * LWBS_BY_PAYER[p] for p in PAYER_MIX)
    assert _close(lwbs, LWBS_RATE_OVERALL, 0.002), f"Weighted LWBS is {lwbs:.4f}, brief says {LWBS_RATE_OVERALL}"

    # The brief's outcome-gaps table must be reproduced exactly where it gives a number.
    for payer, want in {"uninsured": 0.058, "medicaid_chip": 0.046, "commercial": 0.030}.items():
        assert LWBS_BY_PAYER[payer] == want, f"LWBS {payer} should be {want} (brief)"
    for payer, want in {"uninsured": 64, "medicaid_chip": 58, "commercial": 48}.items():
        assert DOOR_TO_PROVIDER_MIN_BY_PAYER[payer] == want, f"door-to-provider {payer} should be {want} (brief)"
    # Medians do not average exactly, but the mix-weighted value should sit near the overall 55.
    d2p = sum(PAYER_MIX[p] * DOOR_TO_PROVIDER_MIN_BY_PAYER[p] for p in PAYER_MIX)
    assert _close(d2p, DOOR_TO_PROVIDER_MIN_OVERALL, 2), f"weighted door-to-provider {d2p:.1f} vs {DOOR_TO_PROVIDER_MIN_OVERALL}"

    # Arrival shape: a morning peak and an evening peak, both above the mid-afternoon dip.
    w = ARRIVAL_HOURLY_WEIGHTS
    assert max(w[8:12]) > w[14] and max(w[16:20]) > w[14], "arrivals need morning and evening peaks (brief)"
    assert WINTER_VOLUME_MULTIPLIER == 1.20, "brief: winter adds about 20% to volume"

    # History window: one full year, every winter month inside it, demo day after it.
    from datetime import date, timedelta
    vol = expected_daily_visits()
    assert len(vol) == HISTORY_DAYS == 365, "history must be one full year"
    assert _close(sum(vol.values()), ED_VISITS_PER_YEAR, 1e-6), "daily volumes must total the brief's 85,000"
    assert set(WINTER_MONTHS) <= {d.month for d in vol}, "history window must include winter"
    mondays = [v for d, v in vol.items() if d.weekday() == 0 and d.month not in WINTER_MONTHS]
    sundays = [v for d, v in vol.items() if d.weekday() == 6 and d.month not in WINTER_MONTHS]
    assert sum(mondays) / len(mondays) > sum(sundays) / len(sundays), "Monday must be the busiest day (brief)"
    win = [v for d, v in vol.items() if d.month in WINTER_MONTHS]
    oth = [v for d, v in vol.items() if d.month not in WINTER_MONTHS]
    assert _close((sum(win) / len(win)) / (sum(oth) / len(oth)), WINTER_VOLUME_MULTIPLIER, 0.01), "winter days must run ~20% busier"   # weekday mix differs slightly by month
    assert date.fromisoformat(DEMO_DATE) > max(vol), "demo day must come after the history window"

    # Beds: 40 total, statuses add up, discharge-pending is a subset of occupied.
    total_beds = sum(BEDS_DEMO["unit_counts"].values())
    assert total_beds == 40 and sum(BEDS_DEMO["status_counts"].values()) == total_beds
    assert BEDS_DEMO["occupied_with_discharge_pending"] <= BEDS_DEMO["status_counts"]["occupied"]
    assert set(BEDS_DEMO["status_counts"]) == set(BED_STATUSES)

    # Notes: demo notes are a subset of the evaluation set.
    assert N_NOTES_DEMO <= N_NOTES_TOTAL

    # Leakage and fairness guard: no prohibited column may be a model feature.
    assert_features_allowed(MODEL_FEATURES)
    assert CATEGORY_THRESHOLDS["possible"] < CATEGORY_THRESHOLDS["likely"]
    return True


def assert_features_allowed(columns):
    """Call this on the model's feature list before training."""
    bad = [c for c in columns if c in PROHIBITED_FEATURES]
    assert not bad, f"Prohibited features in model input: {bad}"


if __name__ == "__main__":
    validate_config()
    shape_rate = sum(ESI_MIX[k] * ESI_ADMIT_RATE_SHAPE[k] for k in ESI_MIX)
    lwbs = sum(PAYER_MIX[p] * LWBS_BY_PAYER[p] for p in PAYER_MIX)
    print("config OK")
    print(f"  visits/day (mean)            : {MEAN_VISITS_PER_DAY:.0f}")
    print(f"  admits/day at target rate    : {MEAN_VISITS_PER_DAY * ADMIT_RATE_TARGET:.0f}")
    print(f"  admit rate from ESI shape    : {shape_rate:.3f}  (target {ADMIT_RATE_TARGET})")
    print(f"  weighted LWBS across payers  : {lwbs:.4f}  (brief {LWBS_RATE_OVERALL})")
    vol = expected_daily_visits()
    print(f"  history window               : {HISTORY_START_DATE} + {HISTORY_DAYS} days = {sum(vol.values()):,.0f} visits")
    print(f"  busiest / quietest day       : {max(vol.values()):.0f} / {min(vol.values()):.0f} visits")
    print(f"  test-set uninsured visits    : ~{int(N_HISTORY_VISITS_EXPECTED * SPLIT['test'] * PAYER_MIX['uninsured'])}")
    print(f"  test-set uninsured admits    : ~{int(N_HISTORY_VISITS_EXPECTED * SPLIT['test'] * PAYER_MIX['uninsured'] * ADMIT_RATE_TARGET)}")
    print(f"  model features               : {len(MODEL_FEATURES)}")