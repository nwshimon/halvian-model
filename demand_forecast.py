"""
demand_forecast.py -- Step 4: how many new admit decisions (= inpatient beds needed) in the next 2, 4 and 8 hours?

For a clock time `now` and a horizon H the forecast adds three groups (all counts of ADMIT DECISIONS that fall in (now, now+H]):
  1. patients in the ED now and still undecided:  q_i = P(admitted) x P(decision lands within H | admitted, still undecided)
        P(admitted)       = the calibrated admit model (Step 3), from the patient's latest snapshot
        P(decision in H)  = a small time-to-decision model (log-normal, below), conditioned on the time already elapsed
  2. patients who have arrived but not yet been triaged (no snapshot yet): same idea from arrival-time history
  3. patients who have not arrived yet: expected arrivals (config hourly/day-of-week/winter shape) x P(admitted and decided in time)
Spec math [SPEC]: forecast = sum of per-patient probabilities, variance = sum of q(1-q) (+ Poisson variance for group 3),
90% range = forecast +/- FORECAST_INTERVAL_Z x sqrt(variance).
Patients who ALREADY have an admit decision and are boarding are reported separately as known demand.

Run:  python demand_forecast.py     (fits the time model, backtests on test visits, prints a sample forecast)
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import admit_model as AM
import config as C

MIN = 60
Z = C.FORECAST_INTERVAL_Z
HORIZONS_MIN = [h * 60 for h in C.FORECAST_HORIZONS_HOURS]


# ------------------------------------------------------------------ expected arrivals
def make_daily_expected():
    hist = C.expected_daily_visits()
    raw = lambda d: 7 * C.DAY_OF_WEEK_WEIGHTS[d.weekday()] * (C.WINTER_VOLUME_MULTIPLIER if d.month in C.WINTER_MONTHS else 1.0)
    scale = C.N_HISTORY_VISITS_EXPECTED / sum(raw(d) for d in hist)
    return lambda d: hist.get(d, raw(d) * scale)


def arrival_rate_per_min(times):
    """Expected arrivals per minute at each datetime in `times` (a DatetimeIndex)."""
    f = make_daily_expected()
    days = pd.Series(times.date)
    daily = days.map({d: f(d) for d in days.unique()}).to_numpy()
    hw = np.array(C.ARRIVAL_HOURLY_WEIGHTS)[times.hour.to_numpy()]
    return daily * hw / 60.0


# ------------------------------------------------------------------ time-to-decision model
class TimeToDecision:
    """log(minutes from a snapshot to the admit decision) ~ Normal(mu(features), sigma), fit on ADMITTED visits."""

    def fit(self, snap, visits):
        v = visits.set_index("visit_id")["disposition_time"]
        a = snap[snap["label_admit"].astype(bool)].copy()
        a["R"] = (v.reindex(a["visit_id"]).to_numpy() - a["snapshot_time"].to_numpy()) / np.timedelta64(1, "m")
        assert (a["R"] > 0).all(), "a snapshot is not before its admit decision"
        tr, va = a[a["split"] == "train"], a[a["split"] == "validation"]
        self.pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), AM.CATEGORICAL),
                                      ("n", StandardScaler(), AM.NUMERIC)]).fit(AM._design(tr))
        self.reg = Ridge(alpha=1.0).fit(self.pre.transform(AM._design(tr)), np.log(tr["R"]))
        resid = np.log(va["R"]) - self.reg.predict(self.pre.transform(AM._design(va)))
        self.sigma = float(resid.std())
        return self

    def mu(self, df):
        return self.reg.predict(self.pre.transform(AM._design(df)))

    def prob_within(self, mu, e_min, h_min):
        """P(decision within h minutes from now | not decided by e minutes after the snapshot)."""
        ze = (np.log(np.maximum(e_min, 1e-3)) - mu) / self.sigma
        zh = (np.log(np.maximum(e_min + h_min, 1e-3)) - mu) / self.sigma
        return np.clip(1.0 - np.exp(norm.logsf(zh) - norm.logsf(ze)), 0.0, 1.0)


class DemandForecaster:
    def fit(self, snap, visits):
        self.ttd = TimeToDecision().fit(snap, visits)
        tr = visits[visits["split"] == "train"]
        d = ((tr["disposition_time"] - tr["arrival_time"]).dt.total_seconds() / 60).dropna().to_numpy()
        self.arrival_to_decision = np.sort(d)
        self.n_train = len(tr)
        return self

    def G(self, minutes):
        """P(a new arrival is admitted AND the decision lands within `minutes` of arrival)."""
        return np.searchsorted(self.arrival_to_decision, minutes, side="right") / self.n_train

    # ---- group 1: patients with a snapshot
    def q_in_ed(self, rows, p, e_min, h_min):
        return p * self.ttd.prob_within(self.ttd.mu(rows), e_min, h_min)

    # ---- group 2: arrived, not yet triaged (conditional on not yet decided)
    def q_untriaged(self, e_min, h_min):
        g_e = self.G(e_min)
        return np.clip((self.G(e_min + h_min) - g_e) / np.maximum(1 - g_e, 1e-6), 0, 1)

    # ---- group 3: patients who have not arrived yet (mean, Poisson variance)
    def future_arrivals(self, now, h_min, fraction=1.0):
        minutes = np.arange(1, h_min + 1)
        rate = arrival_rate_per_min(pd.DatetimeIndex([now + pd.Timedelta(minutes=int(m)) for m in minutes]))
        return float((rate * self.G(h_min - minutes)).sum() * fraction)

    def save(self, path=f"{AM.MODEL_DIR}/demand_forecaster.joblib"):
        os.makedirs(AM.MODEL_DIR, exist_ok=True)
        joblib.dump(self, path)


# ------------------------------------------------------------------ one forecast for one clock time
def forecast_now(fc, am, now, visits, snap, fraction=1.0):
    """Forecast for the whole ED at `now` (fraction < 1 treats the visits as a sample of the ED, used by the backtest).
    Only information up to `now` is used as input; later times are compared with `now`, never read as features."""
    now = pd.Timestamp(now)
    s = snap[snap["snapshot_time"] <= now]
    last = s.sort_values("snapshot_time", kind="stable").groupby("visit_id").tail(1)
    v = visits.set_index("visit_id")
    dec, left = v["disposition_time"], v["left_ed_time"]
    in_ed = last[(left.reindex(last["visit_id"]).to_numpy() > now) &
                 (pd.isna(dec.reindex(last["visit_id"]).to_numpy()) | (dec.reindex(last["visit_id"]).to_numpy() > now))].copy()
    boarding = int(((dec <= now) & (left > now)).sum())
    untri = visits[(visits["arrival_time"] <= now) & (visits["left_ed_time"] > now) & ~visits["visit_id"].isin(s["visit_id"])]
    in_ed["p"] = am.predict_proba(in_ed) if len(in_ed) else []
    e1 = ((now - in_ed["snapshot_time"]).dt.total_seconds() / 60).to_numpy()
    e2 = ((now - untri["arrival_time"]).dt.total_seconds() / 60).to_numpy()
    out = {"now": now, "in_ed_undecided": len(in_ed), "not_yet_triaged": len(untri), "boarding_now": boarding}
    for H in HORIZONS_MIN:
        q1 = fc.q_in_ed(in_ed, in_ed["p"].to_numpy(), e1, H) if len(in_ed) else np.array([])
        q2 = fc.q_untriaged(e2, H) if len(untri) else np.array([])
        arr = fc.future_arrivals(now, H, fraction)
        mean = q1.sum() + q2.sum() + arr
        var = (q1 * (1 - q1)).sum() + (q2 * (1 - q2)).sum() + arr
        lo, hi = max(0.0, mean - Z * np.sqrt(var)), mean + Z * np.sqrt(var)
        out[f"{H // 60}h"] = {"expected_new_admits": mean, "low90": lo, "high90": hi,
                              "from_in_ed": q1.sum(), "from_not_triaged": q2.sum(), "from_future_arrivals": arr,
                              "total_beds_needed": boarding + mean,
                              "by_level_of_care": {k: mean * sh for k, sh in C.LEVEL_OF_CARE_SHARE.items()}}   # planning shares
    return out


# ------------------------------------------------------------------ backtest (test visits only)
def _expand_hours(start_s, end_s, t_lo_s, t_hi_s):
    """For each interval [start, end) every whole hour T in it, within [t_lo, t_hi]. Returns (row index, T seconds)."""
    first = np.maximum(np.ceil(start_s / 3600) * 3600, t_lo_s)
    n = np.maximum(0, np.ceil((np.minimum(end_s, t_hi_s + 1) - first) / 3600)).astype(int)
    row = np.repeat(np.arange(len(start_s)), n)
    T = first[row] + 3600 * (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    return row, T


def _secs(x):
    return pd.to_datetime(x).to_numpy().astype("datetime64[s]").astype(np.int64)


def backtest(fc, am, snap, visits):
    frac = (visits["split"] == "test").mean()
    v = visits.set_index("visit_id")
    ts = snap[snap["split"] == "test"].sort_values(["visit_id", "snapshot_time"], kind="stable").copy()
    ts["p"] = am.predict_proba(ts)
    nxt = ts.groupby("visit_id")["snapshot_time"].shift(-1).fillna(pd.Timestamp.max)
    stop = v["disposition_time"].fillna(v["left_ed_time"]).reindex(ts["visit_id"]).to_numpy()
    end = np.minimum(nxt.to_numpy(dtype="datetime64[ns]"), stop.astype("datetime64[ns]"))
    start_s, end_s = _secs(ts["snapshot_time"]), _secs(end)

    t0 = pd.Timestamp(C.HISTORY_START_DATE) + pd.Timedelta(days=1)
    t1 = pd.Timestamp(C.HISTORY_START_DATE) + pd.Timedelta(days=C.HISTORY_DAYS) - pd.Timedelta(hours=max(C.FORECAST_HORIZONS_HOURS) + 1)
    t_lo, t_hi = _secs([t0])[0], _secs([t1])[0]
    nT = int((t_hi - t_lo) // 3600) + 1
    times = pd.to_datetime(t_lo + 3600 * np.arange(nT), unit="s")

    row, T = _expand_hours(start_s, end_s, t_lo, t_hi)
    sub = ts.iloc[row]
    e1 = (T - start_s[row]) / 60.0
    ti1 = ((T - t_lo) // 3600).astype(int)

    tv = visits[visits["split"] == "test"]
    first_snap = ts.groupby("visit_id")["snapshot_time"].first().reindex(tv["visit_id"])
    row2, T2 = _expand_hours(_secs(tv["arrival_time"]), _secs(first_snap), t_lo, t_hi)
    e2 = (T2 - _secs(tv["arrival_time"])[row2]) / 60.0
    ti2 = ((T2 - t_lo) // 3600).astype(int)

    dec_s = np.sort(_secs(tv["disposition_time"].dropna()))
    rate = arrival_rate_per_min(pd.to_datetime(t_lo + 60 * np.arange(nT * 60 + 8 * 60 + 2), unit="s"))
    res = {}
    for H in HORIZONS_MIN:
        q1 = fc.q_in_ed(sub, sub["p"].to_numpy(), e1, H)
        q2 = fc.q_untriaged(e2, H)
        m = np.arange(1, H + 1)
        idx = (np.arange(nT) * 60)[:, None] + m[None, :]
        arr = (rate[idx] * fc.G(H - m)[None, :]).sum(1) * frac
        mean = np.bincount(ti1, q1, nT) + np.bincount(ti2, q2, nT) + arr
        var = np.bincount(ti1, q1 * (1 - q1), nT) + np.bincount(ti2, q2 * (1 - q2), nT) + arr
        actual = (np.searchsorted(dec_s, T_grid(t_lo, nT) + H * 60, side="right") - np.searchsorted(dec_s, T_grid(t_lo, nT), side="right"))
        lo, hi = np.maximum(0, mean - Z * np.sqrt(var)), mean + Z * np.sqrt(var)
        grp = pd.DataFrame({"a": actual, "k": times.dayofweek * 24 + times.hour})
        gs, gn = grp.groupby("k")["a"].transform("sum"), grp.groupby("k")["a"].transform("count")
        base = ((gs - grp["a"]) / (gn - 1)).to_numpy()                       # same hour-of-week, every other week
        res[H // 60] = {"mean_actual": actual.mean(), "mean_forecast": mean.mean(), "bias_pct": 100 * (mean.mean() / actual.mean() - 1),
                        "mae": np.abs(mean - actual).mean(), "mae_baseline": np.abs(base - actual).mean(),
                        "coverage90": float(((actual >= lo) & (actual <= hi)).mean()),
                        "scaled_hospital_mean": actual.mean() / frac, "n_times": nT,
                        "mean_from_in_ed": np.bincount(ti1, q1, nT).mean(), "mean_from_arrivals": arr.mean()}
    return res, frac


def T_grid(t_lo, nT):
    return t_lo + 3600 * np.arange(nT)


def main():
    from build_snapshots import load
    visits, events, vitals, occ = load()
    snap = pd.read_csv(f"{C.DATA_DIR}/snapshots.csv", parse_dates=["snapshot_time"])
    am = AM.AdmitModel.load()
    fc = DemandForecaster().fit(snap, visits)
    fc.save()
    print(f"time-to-decision model: sigma of log-minutes = {fc.ttd.sigma:.2f}")

    # time model check on test admits: predicted vs actual P(decision within H of a snapshot)
    v = visits.set_index("visit_id")["disposition_time"]
    a = snap[(snap["split"] == "test") & snap["label_admit"].astype(bool)]
    R = (v.reindex(a["visit_id"]).to_numpy() - a["snapshot_time"].to_numpy()) / np.timedelta64(1, "m")
    mu = fc.ttd.mu(a)
    print("time model on test admits (P decision within H of a snapshot): " + " | ".join(
        f"{h}h predicted {fc.ttd.prob_within(mu, 0.0, h * 60).mean():.2f} vs actual {(R <= h * 60).mean():.2f}"
        for h in C.FORECAST_HORIZONS_HOURS))

    print("\nBacktest on TEST visits (a 20% sample of the ED), every hour of the year, compared with the 'same hour of week' average")
    res, frac = backtest(fc, am, snap, visits)
    rows = []
    for h, r in res.items():
        rows.append({"horizon": f"{h}h", "actual": r["mean_actual"], "forecast": r["mean_forecast"], "bias %": r["bias_pct"],
                     "MAE": r["mae"], "MAE avg-baseline": r["mae_baseline"],
                     "improvement %": 100 * (1 - r["mae"] / r["mae_baseline"]), "90% range coverage": r["coverage90"],
                     "share from patients in ED": r["mean_from_in_ed"] / r["mean_forecast"]})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"(test visits are {frac:.0%} of the ED; multiply counts by {1 / frac:.2f} for the whole hospital: "
          + ", ".join(f"{h}h ~{r['scaled_hospital_mean']:.1f} admits on average" for h, r in res.items()) + ")")

    now = pd.Timestamp("2026-02-10 08:00")                       # a winter Tuesday morning inside the history
    f = forecast_now(fc, am, now, visits, snap)
    print(f"\nsample forecast, whole ED at {now}: {f['in_ed_undecided']} patients undecided, "
          f"{f['not_yet_triaged']} not yet triaged, {f['boarding_now']} already boarding")
    for h in C.FORECAST_HORIZONS_HOURS:
        r = f[f"{h}h"]
        print(f"   next {h}h: {r['expected_new_admits']:.1f} new admit decisions (90% range {r['low90']:.1f}-{r['high90']:.1f}) "
              f"| beds needed incl. boarding {r['total_beds_needed']:.1f} | by level of care "
              + ", ".join(f"{k} {x:.1f}" for k, x in r["by_level_of_care"].items()))
    json.dump({str(k): {a: float(b) for a, b in r.items()} for k, r in res.items()},
              open(f"{AM.MODEL_DIR}/forecast_backtest.json", "w"), indent=2)


if __name__ == "__main__":
    import demand_forecast          # run through the module so pickled classes are stored as demand_forecast.<Class>, not __main__.<Class>
    demand_forecast.main()
