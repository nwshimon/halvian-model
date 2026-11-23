"""
admit_model.py -- Step 3: the admit model.

Logistic regression on snapshot rows -> calibrated admit probability -> category -> top 3 factors.

Design decisions (all visible below):
 1. Train on TRAIN rows only; tune the regularisation C and choose the calibrator on VALIDATION; touch TEST once.
 2. Each snapshot is weighted by how long it stays the patient's latest one (minutes until the next snapshot,
    or until the admit decision / departure). Admitted patients generate more snapshots, so an unweighted model
    over-predicts for "whoever is in the ED right now" (measured: 16.1% predicted vs 14.1% actual), and the demand
    forecast adds those probabilities up. Time weighting removes that bias.
 3. Features are exactly config.MODEL_FEATURES. payer and the other PROHIBITED_FEATURES can never get in.
 4. Top factors = each feature's contribution to the log-odds relative to the average ED patient.

Run:  python admit_model.py     (trains, evaluates on test once, saves ./models/admit_model.joblib)
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C
import schema as S

CATEGORICAL = ["age_band", "complaint_category", "esi"]
NUMERIC = [f for f in C.MODEL_FEATURES if f not in CATEGORICAL]
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
LOGLOSS_TOLERANCE = 0.0005             # validation log-loss differences smaller than this are treated as ties
MODEL_DIR = "models"

FEATURE_LABELS = {
    "age_band": "Age", "esi": "ESI triage level", "complaint_category": "Chief complaint",
    "max_hr_so_far": "Highest heart rate", "min_spo2_so_far": "Lowest oxygen saturation",
    "min_sbp_so_far": "Lowest systolic BP", "max_rr_so_far": "Highest respiratory rate",
    "max_temp_so_far": "Highest temperature", "abnormal_vital_count_so_far": "Abnormal vital signs",
    "orders_so_far": "Orders placed", "ct_orders_so_far": "CT ordered",
    "abnormal_results_so_far": "Abnormal results", "pending_results_so_far": "Results still pending",
    "mins_since_arrival": "Time since arrival (min)", "mins_since_last_reassessment": "Time since last check (min)",
    "inpatient_occupancy_at_snapshot": "Inpatient occupancy",
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def time_weights(snap, visits):
    """Minutes each snapshot stays the patient's latest one (>= 0.5). See design decision 2."""
    s = snap.sort_values(["visit_id", "snapshot_time"], kind="stable")
    v = visits.set_index("visit_id")
    end = v["disposition_time"].fillna(v["left_ed_time"]).reindex(s["visit_id"]).to_numpy()
    nxt = s.groupby("visit_id")["snapshot_time"].shift(-1).to_numpy()
    stop = np.where(pd.isna(nxt), end, nxt)
    mins = (pd.to_datetime(stop) - s["snapshot_time"]).dt.total_seconds().to_numpy() / 60
    return pd.Series(np.clip(mins, 0.5, None), index=s.index).reindex(snap.index)


def _design(df):
    X = df[C.MODEL_FEATURES].copy()
    X["esi"] = X["esi"].astype(int).astype(str)
    return X


class AdmitModel:
    def __init__(self):
        assert not set(C.MODEL_FEATURES) & set(C.PROHIBITED_FEATURES), "a prohibited feature is in MODEL_FEATURES"
        self.pre = None; self.lr = None; self.calibrator = None; self.calibration = "none"
        self.C = None; self.base_logit_means = None; self.col_feature = None; self.report = {}

    # ---------------------------------------------------------------- fitting
    def _make_pre(self):
        return ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
                                  ("n", StandardScaler(), NUMERIC)])

    def fit(self, snap, weights):
        tr = snap["split"] == "train"
        va = snap["split"] == "validation"
        Xtr, ytr, wtr = _design(snap[tr]), snap.loc[tr, "label_admit"].astype(int), weights[tr]
        Xva, yva, wva = _design(snap[va]), snap.loc[va, "label_admit"].astype(int), weights[va]
        self.pre = self._make_pre().fit(Xtr)
        Ztr, Zva = self.pre.transform(Xtr), self.pre.transform(Xva)

        # tune regularisation on validation (weighted log loss)
        scores = {}
        for c in C_GRID:
            m = LogisticRegression(C=c, max_iter=5000).fit(Ztr, ytr, sample_weight=wtr)
            scores[c] = log_loss(yva, m.predict_proba(Zva)[:, 1], sample_weight=wva)
        # log loss is nearly flat across C here, so do not chase noise: among the C values within LOGLOSS_TOLERANCE of the
        # best, take the one closest (in log scale) to the config default
        best = min(scores.values())
        eligible = [c for c, v in scores.items() if v <= best + LOGLOSS_TOLERANCE]
        self.C = min(eligible, key=lambda c: abs(np.log(c / C.LOGREG_C)))
        self.lr = LogisticRegression(C=self.C, max_iter=5000).fit(Ztr, ytr, sample_weight=wtr)
        self.report["C_grid_validation_logloss"] = {str(k): round(v, 5) for k, v in scores.items()}
        self.report["chosen_C"] = self.C
        self.report["config_default_C"] = C.LOGREG_C

        # centre for explanations: weighted mean of the encoded training columns
        self.base_logit_means = np.average(Ztr, axis=0, weights=wtr)
        names = self.pre.get_feature_names_out()
        self.col_feature = [next(f for f in sorted(C.MODEL_FEATURES, key=len, reverse=True) if n.split("__", 1)[1].startswith(f))
                            for n in names]

        # calibration chosen by 3-fold cross-fit (grouped by visit) inside VALIDATION, then refit on all of it
        raw_va = self.lr.predict_proba(Zva)[:, 1]
        groups = snap.loc[va, "visit_id"].to_numpy()
        cv = {"none": [], "platt": [], "isotonic": []}
        for a, b in GroupKFold(n_splits=3).split(raw_va, yva, groups):
            for kind in cv:
                cal = self._fit_calibrator(kind, raw_va[a], yva.to_numpy()[a], wva.to_numpy()[a])
                p = self._apply_calibrator(kind, cal, raw_va[b])
                cv[kind].append(brier_score_loss(yva.to_numpy()[b], p, sample_weight=wva.to_numpy()[b]))
        cv_mean = {k: float(np.mean(v)) for k, v in cv.items()}
        self.calibration = min(cv_mean, key=cv_mean.get)
        self.calibrator = self._fit_calibrator(self.calibration, raw_va, yva.to_numpy(), wva.to_numpy())
        self.report["calibration_cv_brier"] = {k: round(v, 5) for k, v in cv_mean.items()}
        self.report["calibration_chosen"] = self.calibration
        return self

    @staticmethod
    def _fit_calibrator(kind, raw, y, w):
        if kind == "none":
            return None
        if kind == "platt":
            return LogisticRegression(C=1e6, max_iter=1000).fit(logit(raw).reshape(-1, 1), y, sample_weight=w)
        return IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(raw, y, sample_weight=w)

    @staticmethod
    def _apply_calibrator(kind, cal, raw):
        if kind == "none":
            return raw
        if kind == "platt":
            return cal.predict_proba(logit(raw).reshape(-1, 1))[:, 1]
        return cal.predict(raw)

    # ---------------------------------------------------------------- scoring
    def predict_proba(self, df):
        raw = self.lr.predict_proba(self.pre.transform(_design(df)))[:, 1]
        return self._apply_calibrator(self.calibration, self.calibrator, raw)

    @staticmethod
    def category(p):
        t = C.CATEGORY_THRESHOLDS
        p = np.asarray(p)
        return np.where(p >= t["likely"], "likely", np.where(p >= t["possible"], "possible", "unlikely"))

    def contributions(self, df):
        """DataFrame (rows x features): log-odds push toward admit relative to the average patient."""
        Z = self.pre.transform(_design(df))
        per_col = (Z - self.base_logit_means) * self.lr.coef_[0]
        out = pd.DataFrame(per_col, index=df.index)
        return out.T.groupby(self.col_feature).sum().T

    def explain(self, df, k=C.TOP_FACTORS_SHOWN):
        """For each row: the k features with the largest absolute push, as 'Label (+0.8)' strings."""
        contrib = self.contributions(df)
        rows = []
        for idx in contrib.index:
            top = contrib.loc[idx].reindex(contrib.loc[idx].abs().sort_values(ascending=False).index)[:k]
            rows.append([f"{FEATURE_LABELS[f]} ({'+' if x >= 0 else '-'}{abs(x):.2f})" for f, x in top.items()])
        return pd.DataFrame(rows, index=contrib.index, columns=[f"factor_{i + 1}" for i in range(k)])

    def save(self, path=None):
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = path or f"{MODEL_DIR}/admit_model.joblib"
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path=f"{MODEL_DIR}/admit_model.joblib"):
        # A model file saved by running this file directly records its class as "__main__.AdmitModel", which a web app cannot find.
        # Point that name at the real class so both kinds of file load.
        import sys
        main = sys.modules.get("__main__")
        if main is not None and not hasattr(main, "AdmitModel"):
            main.AdmitModel = AdmitModel
        return joblib.load(path)


# ------------------------------------------------------------------ evaluation (TEST, once)
def weighted_calibration_table(p, y, w, edges=(0, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.0001)):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        rows.append({"bin": f"{lo:.2f}-{min(hi, 1):.2f}", "rows": int(m.sum()), "share_of_time": w[m].sum() / w.sum(),
                     "mean_predicted": np.average(p[m], weights=w[m]), "observed_admit_rate": np.average(y[m], weights=w[m])})
    t = pd.DataFrame(rows)
    ece = float((t["share_of_time"] * (t["mean_predicted"] - t["observed_admit_rate"]).abs()).sum())
    return t, ece


def bootstrap_auc_by_visit(visit_ids, y, p, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    order = np.argsort(visit_ids, kind="stable")
    v = visit_ids[order]
    starts = np.r_[0, np.flatnonzero(v[1:] != v[:-1]) + 1]
    ends = np.r_[starts[1:], len(v)]
    groups = [order[a:b] for a, b in zip(starts, ends)]
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def evaluate_on_test(model, snap, weights):
    te = snap[snap["split"] == "test"].copy()
    w = weights[te.index].to_numpy()
    y = te["label_admit"].astype(int).to_numpy()
    te["p"] = model.predict_proba(te)
    p = te["p"].to_numpy()
    rep = {}
    auc = roc_auc_score(y, p)
    lo, hi = bootstrap_auc_by_visit(te["visit_id"].to_numpy(), y, p)
    rep["auc"] = auc; rep["auc_ci95"] = [lo, hi]
    rep["pr_auc"] = average_precision_score(y, p); rep["brier"] = brier_score_loss(y, p)
    rep["log_loss"] = log_loss(y, p)
    rep["row_mean_p"], rep["row_rate"] = p.mean(), y.mean()
    rep["time_weighted_mean_p"], rep["time_weighted_rate"] = np.average(p, weights=w), np.average(y, weights=w)
    tbl, ece = weighted_calibration_table(p, y, w)
    rep["calibration_table"], rep["ece_time_weighted"] = tbl, ece

    # same visits, different moments: does the score improve as information arrives?
    te = te.sort_values(["visit_id", "snapshot_time"])
    has_result = te.groupby("visit_id")["trigger_type"].transform(lambda s: (s == "result").any())
    sub = te[has_result]
    pick = {"at triage": sub.groupby("visit_id").head(1),
            "at first order": sub[sub.trigger_type == "order"].groupby("visit_id").head(1),
            "at first result": sub[sub.trigger_type == "result"].groupby("visit_id").head(1),
            "at last snapshot": sub.groupby("visit_id").tail(1)}
    common = set.intersection(*[set(d["visit_id"]) for d in pick.values()])
    rep["same_visits_n"] = len(common)
    rep["same_visits_auc"] = {k: roc_auc_score(d[d.visit_id.isin(common)]["label_admit"].astype(int),
                                               d[d.visit_id.isin(common)]["p"]) for k, d in pick.items()}

    # categories (time-weighted, as the ED would experience them)
    te["category"] = model.category(te["p"].to_numpy())
    wt = weights[te.index].to_numpy()
    cats = []
    for c in ("unlikely", "possible", "likely"):
        m = (te["category"] == c).to_numpy()
        yy = te["label_admit"].astype(int).to_numpy()
        cats.append({"category": c, "rows": int(m.sum()), "share_of_time": wt[m].sum() / wt.sum(),
                     "observed_admit_rate": np.average(yy[m], weights=wt[m]) if m.any() else np.nan})
    rep["categories"] = pd.DataFrame(cats)
    lik = (te["category"] == "likely").to_numpy(); yy = te["label_admit"].astype(int).to_numpy()
    rep["likely_precision"] = np.average(yy[lik], weights=wt[lik])
    rep["likely_recall"] = (wt[lik & (yy == 1)].sum()) / wt[yy == 1].sum()
    rep["unlikely_miss_rate"] = np.average(yy[(te["category"] == "unlikely").to_numpy()], weights=wt[(te["category"] == "unlikely").to_numpy()])
    return rep, te


def recovery_table(model):
    """Learned effects (log-odds, natural units) beside the true effects planted in config.ADMIT_LOGODDS."""
    names = list(model.pre.get_feature_names_out())
    coef = dict(zip(names, model.lr.coef_[0]))
    scaler = model.pre.named_transformers_["n"]
    sd = dict(zip(NUMERIC, scaler.scale_))
    per_unit = lambda f: coef[f"n__{f}"] / sd[f]
    rows = [("per abnormal result", per_unit("abnormal_results_so_far"), C.ADMIT_LOGODDS["per_abnormal_result"]),
            ("per order (see note)", per_unit("orders_so_far"), C.ADMIT_LOGODDS["per_order"]),
            ("inpatient occupancy, per +10 points", per_unit("inpatient_occupancy_at_snapshot") * 0.10, C.ADMIT_LOGODDS["inpatient_occupancy"])]
    age_ref = "35-49"
    for a in C.AGE_BANDS:
        rows.append((f"age {a} vs {age_ref}", coef[f"c__age_band_{a}"] - coef[f"c__age_band_{age_ref}"],
                     C.ADMIT_LOGODDS["age_band"][a] - C.ADMIT_LOGODDS["age_band"][age_ref]))
    cref = "psychiatric"
    for c in C.COMPLAINT_MIX:
        rows.append((f"complaint {c} vs {cref}", coef[f"c__complaint_category_{c}"] - coef[f"c__complaint_category_{cref}"],
                     C.ADMIT_LOGODDS["complaint"][c] - C.ADMIT_LOGODDS["complaint"][cref]))
    return pd.DataFrame(rows, columns=["effect", "learned", "true"])


def main():
    from build_snapshots import load
    visits, events, vitals, occ = load()
    snap = pd.read_csv(f"{C.DATA_DIR}/snapshots.csv", parse_dates=["snapshot_time"])
    problems = S.validate(snap, "snapshots")
    assert not problems, problems
    w = time_weights(snap, visits)

    print("Training on train rows, tuning on validation ...")
    model = AdmitModel().fit(snap, w)
    print(f"chosen C = {model.C} (config default {C.LOGREG_C}); validation log loss by C: {model.report['C_grid_validation_logloss']}")
    print(f"calibration: cross-fit Brier on validation {model.report['calibration_cv_brier']} -> using '{model.calibration}'")
    model.save()

    print("\nEvaluating on TEST (first and only time) ...")
    rep, te = evaluate_on_test(model, snap, w)
    lo, hi = rep["auc_ci95"]
    ok = C.EXPECTED_AUC_RANGE[0] <= rep["auc"] <= C.EXPECTED_AUC_RANGE[1]
    print(f"AUC {rep['auc']:.3f} (95% CI {lo:.3f}-{hi:.3f}, bootstrapped by visit) | expected {C.EXPECTED_AUC_RANGE}: {'inside' if ok else 'OUTSIDE - investigate'}")
    print(f"PR-AUC {rep['pr_auc']:.3f} | Brier {rep['brier']:.3f} | log loss {rep['log_loss']:.3f}")
    print(f"mean predicted vs actual admit rate: by row {rep['row_mean_p']:.3f} vs {rep['row_rate']:.3f}; "
          f"for who is in the ED (time-weighted) {rep['time_weighted_mean_p']:.3f} vs {rep['time_weighted_rate']:.3f}")
    print(f"\ncalibration, time-weighted (expected calibration error {rep['ece_time_weighted']:.3f}):")
    print(rep["calibration_table"].round(3).to_string(index=False))
    print(f"\nsame {rep['same_visits_n']:,} test visits, scored at different moments (AUC):")
    for k, v in rep["same_visits_auc"].items():
        print(f"   {k:18s} {v:.3f}")
    print("\ncategories on test (thresholds from config):")
    print(rep["categories"].round(3).to_string(index=False))
    print(f"'likely' precision {rep['likely_precision']:.1%}, recall {rep['likely_recall']:.1%}; "
          f"admit rate hiding in 'unlikely': {rep['unlikely_miss_rate']:.1%}")
    print("\nrecovery of the true effects planted in config.ADMIT_LOGODDS (log-odds):")
    print(recovery_table(model).round(2).to_string(index=False))
    print("note: 'per order' is spread across several correlated order and result features, so it is not comparable one-to-one.")

    # example: a test admit whose category rises to "likely" only after later information arrives
    te_s = te.sort_values(["visit_id", "snapshot_time"]).copy()
    te_s["category"] = model.category(te_s["p"])
    first_cat = te_s.groupby("visit_id")["category"].first()
    last_cat = te_s.groupby("visit_id")["category"].last()
    admit_v = te_s.groupby("visit_id")["label_admit"].first()
    rise = first_cat[(first_cat != "likely") & (last_cat == "likely") & admit_v.reindex(first_cat.index)].index
    vid = rise[0] if len(rise) else te_s["visit_id"].iloc[0]
    exv = te_s[te_s["visit_id"] == vid].copy()
    exv = pd.concat([exv[["trigger_type", "mins_since_arrival", "abnormal_results_so_far", "p", "category"]], model.explain(exv)], axis=1)
    print(f"\nexample: test visit {vid} (admitted), score and top 3 factors at each snapshot")
    print(exv.round(2).to_string(index=False))

    os.makedirs(MODEL_DIR, exist_ok=True)
    metrics = {k: (v if not isinstance(v, (pd.DataFrame, np.floating)) else (v.to_dict("records") if isinstance(v, pd.DataFrame) else float(v)))
               for k, v in rep.items()}
    metrics["auc_ci95"] = [float(x) for x in rep["auc_ci95"]]
    metrics.update(model.report)
    json.dump(metrics, open(f"{MODEL_DIR}/admit_metrics.json", "w"), indent=2, default=float)
    print(f"\nSaved {MODEL_DIR}/admit_model.joblib and {MODEL_DIR}/admit_metrics.json")


if __name__ == "__main__":
    import admit_model          # run through the module so pickled classes are stored as admit_model.<Class>, not __main__.<Class>
    admit_model.main()
