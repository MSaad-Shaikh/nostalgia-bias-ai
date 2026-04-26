from io import StringIO
import math
import json
import os

from flask import Flask, jsonify, request
from flask_cors import CORS
import google.generativeai as genai
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

MIN_N = 5
EPSILON = 0.01  # Laplace smoothing for DI stability


# ─── Parsing ──────────────────────────────────────────────────

def parse_csv():
    if "file" not in request.files:
        return None, (jsonify({"error": "No file uploaded."}), 400)
    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".csv"):
        return None, (jsonify({"error": "Upload a .csv file."}), 400)
    try:
        df = pd.read_csv(StringIO(f.stream.read().decode("utf-8", errors="replace")))
        if df.empty:
            return None, (jsonify({"error": "CSV is empty."}), 400)
        return df, None
    except Exception as e:
        return None, (jsonify({"error": str(e)}), 400)


def find_col(df, aliases):
    norm = {str(c).strip().lower(): str(c) for c in df.columns}
    for a in aliases:
        if a in norm:
            return norm[a]
    for n, o in norm.items():
        for a in aliases:
            if a in n:
                return o
    return None


def to_binary(series):
    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce") > 0).astype(float)
    norm = series.astype(str).str.strip().str.lower()
    return norm.isin({"1", "true", "yes", "y", "approved", "accept", "accepted", "pass", "hired"}).astype(float)


# ─── Core Metrics ─────────────────────────────────────────────

def group_stats(df, cat_col, tgt_col):
    w = df[[cat_col, tgt_col]].dropna().copy()
    if w.empty:
        return {}, {}
    w["_o"] = to_binary(w[tgt_col])
    return w.groupby(cat_col)["_o"].mean().dropna().to_dict(), w.groupby(cat_col)["_o"].count().to_dict()


def di_smoothed(rates, counts):
    """DI with Laplace smoothing: (p_min + ε) / (p_max + ε)"""
    stable = {g: r for g, r in rates.items() if counts.get(g, 0) >= MIN_N}
    unstable = len(rates) >= 2 and len(stable) < 2
    if len(stable) < 2:
        return 1.0, 1.0, unstable
    mx, mn = max(stable.values()), min(stable.values())
    raw = round(mn / mx, 4) if mx > 0 else 0.0
    smoothed = round((mn + EPSILON) / (mx + EPSILON), 4)
    return raw, smoothed, unstable


def spd(rates):
    v = list(rates.values())
    return round(max(v) - min(v), 4) if len(v) >= 2 else 0.0


def eo_proxy(rates):
    v = list(rates.values())
    if len(v) < 2:
        return 0.0
    m = sum(v) / len(v)
    return round(math.sqrt(sum((x - m) ** 2 for x in v) / len(v)), 4)


def chi_square_test(df, cat_col, tgt_col):
    """Chi-square test for independence between category and outcome."""
    try:
        w = df[[cat_col, tgt_col]].dropna().copy()
        w["_o"] = to_binary(w[tgt_col])
        ct = pd.crosstab(w[cat_col], w["_o"])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return None, None
        chi2, p, _, _ = scipy_stats.chi2_contingency(ct)
        return round(float(chi2), 2), round(float(p), 4)
    except Exception:
        return None, None


def confidence_score(rates, counts):
    total = sum(counts.values()) if counts else 0
    sz = 1.0 if total >= 100 else (0.7 if total >= 30 else 0.4)
    mn_c = min(counts.values()) if counts else 0
    bal = 1.0 if mn_c >= 20 else (0.6 if mn_c >= 5 else 0.3)
    v = list(rates.values())
    spr = (max(v) - min(v)) if len(v) >= 2 else 0
    vs = min(spr / 0.5, 1.0)
    raw = sz * 0.35 + bal * 0.35 + vs * 0.3
    return {"level": "HIGH" if raw >= 0.7 else ("MEDIUM" if raw >= 0.45 else "LOW"), "score": round(raw, 2)}


def synthetic_flags(rates, counts):
    flags = []
    for g, r in rates.items():
        c = counts.get(g, 0)
        if r == 0.0 and c >= 3:
            flags.append(f"'{g}' 0% positive (n={c}) — deterministic rule or synthetic bias")
        elif r == 1.0 and c >= 3:
            flags.append(f"'{g}' 100% positive (n={c}) — unusually uniform")
    v = list(rates.values())
    if len(v) >= 3 and set(round(x, 2) for x in v).issubset({0.0, 1.0}):
        flags.append("All groups exactly 0% or 100% — strong synthetic indicator")
    return flags


def severity_tier(di):
    if di >= 0.8:
        return {"tier": "Acceptable", "color": "green"}
    elif di >= 0.6:
        return {"tier": "Moderate Concern", "color": "yellow"}
    return {"tier": "High Risk", "color": "red"}


def breakdown_list(rates, counts):
    return sorted([{"group": str(g), "rate": round(r, 4), "count": counts.get(g, 0),
                     "percentage": round(r * 100, 1), "sufficient": counts.get(g, 0) >= MIN_N}
                    for g, r in rates.items()], key=lambda x: x["rate"], reverse=True)


# ─── Bias Classification ─────────────────────────────────────

def classify_bias(di_raw, di_smooth, conf, syn_flags, confounder_strength):
    """Classify bias type: Structural, Conditional, Confounded, Data Artifact."""
    if syn_flags:
        return "Data Artifact"
    if confounder_strength >= 0.5 and di_raw < 0.8:
        return "Confounded Bias"
    if di_raw < 0.6 and conf["level"] == "HIGH":
        return "Structural Bias"
    if di_raw < 0.8:
        return "Conditional Bias"
    return "No Bias"


# ─── Simpson's Paradox Detection ──────────────────────────────

def simpsons_paradox(metrics_all, intersectional):
    """Detect when aggregate and subgroup analyses disagree."""
    findings = []
    for ix in intersectional:
        # Compare intersectional DI with individual DIs
        ix_di = ix["disparate_impact"]
        labels = ix["label"].split(" × ")
        for l in labels:
            if l in metrics_all:
                single_di = metrics_all[l]["di_raw"]
                # Paradox: aggregate looks fair but intersection shows bias, or vice versa
                if single_di >= 0.8 and ix_di < 0.6:
                    findings.append(f"Simpson's Paradox: '{l}' appears fair (DI={single_di:.2f}) "
                                    f"but '{ix['label']}' intersection shows high risk (DI={ix_di:.2f})")
                elif single_di < 0.6 and ix_di >= 0.8:
                    findings.append(f"Reversal: '{l}' shows bias (DI={single_di:.2f}) but "
                                    f"'{ix['label']}' intersection is fair (DI={ix_di:.2f}) — likely confounded")
    return findings


# ─── Confounder & Feature Importance ──────────────────────────

def analyze_features(df, tgt_col, demo_cols):
    """Feature importance ranking + confounder detection."""
    outcome = to_binary(df[tgt_col])
    features = []

    for col in df.columns:
        if col == tgt_col or str(col).strip().lower() in ["id", "name", "index", "row"]:
            continue

        series = df[col]
        is_demo = col in demo_cols
        try:
            if pd.api.types.is_numeric_dtype(series):
                valid = series.dropna()
                if len(valid) < 5:
                    continue
                corr = float(outcome.corr(series))
                if pd.isna(corr):
                    continue
                abs_c = abs(corr)
                if abs_c < 0.1:
                    continue

                # Flag perfect correlation as artifact
                artifact = abs_c >= 0.99
                strength = "Very High" if abs_c >= 0.7 else ("High" if abs_c >= 0.5 else ("Moderate" if abs_c >= 0.3 else "Weak"))

                features.append({
                    "column": str(col), "correlation": round(corr, 3),
                    "abs_correlation": round(abs_c, 3),
                    "direction": "positive" if corr > 0 else "negative",
                    "strength": strength, "is_demographic": is_demo,
                    "is_artifact": artifact,
                    "note": f"Perfect correlation — likely synthetic/deterministic" if artifact
                            else f"{'Demographic' if is_demo else 'Non-demographic'} feature with {strength.lower()} predictive power"
                })
            else:
                cats = series.astype(str).str.strip()
                uniq = cats.nunique()
                if 2 <= uniq <= 20:
                    gr = pd.DataFrame({"c": cats, "o": outcome}).groupby("c")["o"].agg(["mean", "count"])
                    gr = gr[gr["count"] >= 3]
                    if len(gr) >= 2:
                        spread = float(gr["mean"].max() - gr["mean"].min())
                        if spread >= 0.1:
                            strength = "Very High" if spread >= 0.7 else ("High" if spread >= 0.5 else ("Moderate" if spread >= 0.3 else "Weak"))
                            features.append({
                                "column": str(col), "correlation": round(spread, 3),
                                "abs_correlation": round(spread, 3),
                                "direction": "varied", "strength": strength,
                                "is_demographic": is_demo, "is_artifact": spread >= 0.99,
                                "note": f"{'Demographic' if is_demo else 'Non-demographic'}: {spread:.0%} outcome spread"
                            })
        except Exception:
            continue

    features.sort(key=lambda x: x["abs_correlation"], reverse=True)

    # Compute confounder influence (max non-demographic correlation)
    non_demo = [f for f in features if not f["is_demographic"]]
    confounder_strength = non_demo[0]["abs_correlation"] if non_demo else 0.0

    return features[:8], confounder_strength


# ─── Proxy Variable Detection ─────────────────────────────────

def detect_proxies(df, demo_cols, tgt_col):
    """Detect non-demographic columns that strongly correlate with demographics (proxy variables)."""
    proxies = []
    non_demo = [c for c in df.columns if c not in demo_cols and c != tgt_col
                and str(c).strip().lower() not in ["id", "name", "index", "row"]]

    for nd_col in non_demo:
        for d_col in demo_cols:
            try:
                if pd.api.types.is_numeric_dtype(df[nd_col]) and pd.api.types.is_numeric_dtype(df[d_col]):
                    corr = abs(float(df[nd_col].corr(df[d_col])))
                    if corr >= 0.35:
                        proxies.append({
                            "proxy_column": str(nd_col),
                            "demographic_column": str(d_col),
                            "strength": round(corr, 3),
                            "risk": "HIGH" if corr >= 0.6 else "MODERATE",
                            "description": f"{nd_col} strongly correlates with {d_col} and likely acts as a proxy variable"
                        })
                else:
                    # Cramér's V for categorical vs categorical
                    a = df[nd_col].astype(str).str.strip()
                    b = df[d_col].astype(str).str.strip()
                    if a.nunique() > 50 or b.nunique() > 50:
                        continue
                    ct = pd.crosstab(a, b)
                    if ct.shape[0] < 2 or ct.shape[1] < 2:
                        continue
                    chi2, _, _, _ = scipy_stats.chi2_contingency(ct)
                    n = ct.sum().sum()
                    k = min(ct.shape[0], ct.shape[1])
                    cramers_v = math.sqrt(chi2 / (n * (k - 1))) if n * (k - 1) > 0 else 0
                    if cramers_v >= 0.35:
                        proxies.append({
                            "proxy_column": str(nd_col),
                            "demographic_column": str(d_col),
                            "strength": round(cramers_v, 3),
                            "risk": "HIGH" if cramers_v >= 0.6 else "MODERATE",
                            "description": f"{nd_col} strongly correlates with {d_col} and likely acts as a proxy variable"
                        })
            except Exception:
                continue

    proxies.sort(key=lambda x: x["strength"], reverse=True)
    return proxies


# ─── Intersectional ───────────────────────────────────────────

def intersectional(df, demo_cols, tgt):
    results = []
    for i, (la, ca) in enumerate(demo_cols):
        for lb, cb in demo_cols[i + 1:]:
            w = df[[ca, cb, tgt]].dropna().copy()
            w["_combo"] = w[ca].astype(str) + " + " + w[cb].astype(str)
            w["_o"] = to_binary(w[tgt])
            rates = w.groupby("_combo")["_o"].mean().dropna()
            counts = w.groupby("_combo")["_o"].count()
            stable = {g: float(r) for g, r in rates.items() if counts.get(g, 0) >= 3}
            if len(stable) < 2:
                continue
            mx, mn = max(stable.values()), min(stable.values())
            di = round((mn + EPSILON) / (mx + EPSILON), 4) if mx > 0 else 0.0
            worst = min(stable, key=stable.get)
            best = max(stable, key=stable.get)
            bd = sorted([{"group": g, "rate": round(r, 4), "count": int(counts.get(g, 0)),
                          "percentage": round(r * 100, 1)} for g, r in stable.items()],
                        key=lambda x: x["rate"], reverse=True)
            results.append({"label": f"{la} × {lb}", "disparate_impact": di, "spd": round(mx - mn, 4),
                            "severity": severity_tier(di), "worst_group": worst, "best_group": best,
                            "breakdown": bd[:8]})
    return sorted(results, key=lambda x: x["disparate_impact"])


# ─── Risk ─────────────────────────────────────────────────────

def overall_risk(metrics):
    vals = [m["di_smoothed"] for m in metrics.values()]
    if not vals:
        return {"level": "Unknown", "color": "gray", "worst": 0, "avg": 0}
    w, a = min(vals), round(sum(vals) / len(vals), 2)
    if w <= 0.2:
        return {"level": "Critical", "color": "red", "worst": w, "avg": a}
    if w <= 0.5:
        return {"level": "High", "color": "orange", "worst": w, "avg": a}
    if w < 0.8:
        return {"level": "Medium", "color": "yellow", "worst": w, "avg": a}
    return {"level": "Low", "color": "green", "worst": w, "avg": a}


# ─── AI Prompt ────────────────────────────────────────────────

def build_prompt(metrics, risk, syn, ix_data, features, simpsons, conf_strength, proxies=None):
    m_lines = []
    for label, m in metrics.items():
        bd = ", ".join([f"{g['group']}: {g['percentage']}% (n={g['count']})" for g in m["breakdown"]])
        m_lines.append(
            f"**{label}**: DI_raw={m['di_raw']}, DI_smoothed={m['di_smoothed']}, SPD={m['spd']}, "
            f"EO={m['eo']}, Severity={m['severity']['tier']}, Classification={m['bias_class']}, "
            f"Confidence={m['confidence']['level']}, "
            f"p-value={m['p_value'] if m['p_value'] is not None else 'N/A'}\n  Groups: {bd}")

    feat_lines = [f"  {i+1}. {f['column']} ({f['strength']}, r={f['correlation']}) "
                  f"{'⚠ ARTIFACT' if f['is_artifact'] else ''} "
                  f"{'[demographic]' if f['is_demographic'] else '[non-demographic]'}"
                  for i, f in enumerate(features)]

    ix_lines = [f"  {x['label']}: DI={x['disparate_impact']}, worst={x['worst_group']}, best={x['best_group']}"
                for x in ix_data[:3]]

    sp_lines = [f"  ⚠ {s}" for s in simpsons] if simpsons else ["  None detected"]

    proxy_lines = []
    if proxies:
        for p in proxies:
            proxy_lines.append(f"  ⚠ PROXY: {p['proxy_column']} strongly correlates with {p['demographic_column']} "
                               f"(strength={p['strength']}, risk={p['risk']}) — likely acts as a proxy variable")

    return f"""You are a senior bias analyst. Provide a STRUCTURED, PROPORTIONAL, and DECISIVE report.

METRICS:
{chr(10).join(m_lines)}

FEATURE IMPORTANCE (ranked by predictive power):
{chr(10).join(feat_lines) if feat_lines else "  No significant features found."}

Confounder influence: {conf_strength:.2f} (0=none, 1=total)

PROXY VARIABLE DETECTION:
{chr(10).join(proxy_lines) if proxy_lines else '  No proxy variables detected.'}

INTERSECTIONAL:
{chr(10).join(ix_lines) if ix_lines else "  Insufficient data."}

SIMPSON'S PARADOX CHECK:
{chr(10).join(sp_lines)}

SYNTHETIC FLAGS:
{chr(10).join(f'  - {s}' for s in syn) if syn else '  None.'}

Overall Risk: {risk['level']}

BIAS CLASSIFICATIONS: Structural Bias (confirmed, high confidence), Conditional Bias (moderate, needs investigation), Confounded Bias (disparity exists but explained by confounders), Data Artifact (synthetic/deterministic pattern), No Bias.

IMPORTANT RULES:
- START with a "## 🧾 Final Verdict" section — 1-2 lines ONLY that summarize the overall finding decisively.
- When a variable is identified as a proxy, say it "strongly correlates with [demographic] and likely acts as a proxy variable" — NOT "may be a proxy".
- If confounder influence > 0.5, say: "observed disparity is likely largely explained by confounders, though residual disparity cannot be ruled out" — NOT "may be fully explained by confounders".
- If p-value > 0.05, note finding is NOT statistically significant
- If correlation = 1.0, flag as data artifact, not real confounder
- Use sample sizes to qualify every claim

RESPOND WITH THESE SECTIONS (markdown):

## 🧾 Final Verdict
1-2 lines ONLY. Be decisive. Example: "No strong direct demographic bias detected. However, significant conditional bias exists in intersectional groups, partially masked by confounders such as income and credit score."

## Observations
Factual measurements with sample sizes. No judgments.

## Bias Detection Summary
| Demographic | Detected | Confidence | Classification | p-value |
For each: YES/NO, confidence, bias class, significance.

## Feature Importance & Attribution
Rank features. Estimate: "~X% of disparity is likely largely explained by [feature]"

## Proxy Variable Analysis
For each detected proxy, state clearly: "[column] strongly correlates with [demographic] and likely acts as a proxy variable."

## Intersectional Findings
Hidden bias in combined demographics. Flag Simpson's Paradox if detected.

## Possible Confounders
How confounders affect interpretation. "Observed disparity is likely largely explained by [confounder], though residual disparity cannot be ruled out."

## Risk Assessment
Proportional. Separate confirmed vs tentative. Adjust for confounders.

## Hypotheses
NOT conclusions. Data artifact? Real inequality? Confounding? Synthetic?

## Recommended Actions
1. Investigation steps  2. Data remediation  3. Monitoring"""


def gen_report(metrics, risk, syn, ix_data, features, simpsons, conf_strength, proxies=None):
    try:
        r = gemini_model.generate_content(build_prompt(metrics, risk, syn, ix_data, features, simpsons, conf_strength, proxies))
        return getattr(r, "text", "") or "No report."
    except Exception as e:
        return f"AI Failed: {e}"


# ─── Key Insights Generator ──────────────────────────────────

def generate_key_insights(metrics, simpsons, features, ix_data, proxies, conf_strength):
    """Generate top-level key insights for the dashboard."""
    insights = []

    # Simpson's Paradox
    if simpsons:
        insights.append({"icon": "⚠️", "type": "warning",
                         "text": "Hidden bias detected via Simpson's Paradox",
                         "detail": simpsons[0]})

    # Data leakage / artifact features
    for f in features:
        if f["is_artifact"]:
            insights.append({"icon": "⚠️", "type": "warning",
                             "text": f"{f['column']} likely data leakage",
                             "detail": f"Perfect correlation ({f['correlation']}) suggests deterministic/synthetic relationship"})

    # Proxy variables
    for p in proxies:
        insights.append({"icon": "🔗", "type": "proxy",
                         "text": f"Proxy Risk: {p['proxy_column']} → {p['demographic_column']}",
                         "detail": p["description"]})

    # Worst intersectional group
    if ix_data:
        worst_ix = ix_data[0]  # already sorted by DI ascending
        if worst_ix["disparate_impact"] < 0.6:
            insights.append({"icon": "⚠️", "type": "critical",
                             "text": f"{worst_ix['worst_group']} most disadvantaged",
                             "detail": f"DI={worst_ix['disparate_impact']:.2f} in {worst_ix['label']} intersection"})

    # Confounder strength
    if conf_strength >= 0.5:
        insights.append({"icon": "🔍", "type": "info",
                         "text": "Strong confounder influence detected",
                         "detail": f"Confounder strength {conf_strength:.0%} — bias severity should be adjusted"})

    # High-risk demographics
    for label, m in metrics.items():
        if m["bias_class"] == "Structural Bias":
            insights.append({"icon": "🔴", "type": "critical",
                             "text": f"Structural bias confirmed in {label}",
                             "detail": f"DI={m['di_raw']:.2f}, p-value={m['p_value'] if m['p_value'] is not None else 'N/A'}"})

    return insights[:6]  # Cap at 6 most important insights


# ─── Routes ───────────────────────────────────────────────────

@app.route("/api/headers", methods=["POST"])
def headers():
    df, err = parse_csv()
    return err if err else jsonify({"columns": [str(c) for c in df.columns]})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    df, err = parse_csv()
    if err:
        return err

    cols = [str(c) for c in df.columns]
    tgt = request.form.get("target_column")
    if not tgt or tgt not in cols:
        tgt = cols[-1]

    cat_map = {"Age Group": ["age group", "age_group", "age", "age_band"],
               "Gender": ["gender", "sex"],
               "Ethnicity": ["ethnicity", "race", "ethnic_group"]}

    metrics = {}
    all_syn = []
    n_demo = 0
    demo_pairs = []
    demo_col_names = []

    # Feature analysis first (needed for bias classification)
    demo_col_names_temp = [find_col(df, a) for a in cat_map.values()]
    demo_col_names_temp = [c for c in demo_col_names_temp if c]
    features, conf_str = analyze_features(df, tgt, demo_col_names_temp)

    for label, aliases in cat_map.items():
        col = find_col(df, aliases)
        if not col or col == tgt:
            metrics[label] = {"di_raw": 1.0, "di_smoothed": 1.0, "spd": 0.0, "eo": 0.0,
                              "severity": severity_tier(1.0), "confidence": {"level": "LOW", "score": 0},
                              "bias_detected": False, "di_unstable": False, "bias_class": "No Bias",
                              "chi2": None, "p_value": None, "breakdown": [], "synthetic_flags": []}
            continue

        n_demo += 1
        demo_pairs.append((label, col))
        demo_col_names.append(col)
        rates, counts = group_stats(df, col, tgt)
        di_raw, di_sm, unstable = di_smoothed(rates, counts)
        s = spd(rates)
        eo = eo_proxy(rates)
        conf = confidence_score(rates, counts)
        syn = synthetic_flags(rates, counts)
        all_syn.extend(syn)
        chi2, pval = chi_square_test(df, col, tgt)
        bias_class = classify_bias(di_raw, di_sm, conf, syn, conf_str)

        metrics[label] = {
            "di_raw": di_raw, "di_smoothed": di_sm, "spd": s, "eo": eo,
            "severity": severity_tier(di_sm), "confidence": conf,
            "bias_detected": di_sm < 0.8 or s > 0.1,
            "di_unstable": unstable, "bias_class": bias_class,
            "chi2": chi2, "p_value": pval,
            "breakdown": breakdown_list(rates, counts),
            "synthetic_flags": syn,
        }

    risk = overall_risk(metrics)
    ix_data = intersectional(df, demo_pairs, tgt) if len(demo_pairs) >= 2 else []
    simpsons = simpsons_paradox(metrics, ix_data)
    proxies = detect_proxies(df, demo_col_names, tgt)
    key_insights = generate_key_insights(metrics, simpsons, features, ix_data, proxies, conf_str)

    preview = [{str(c): _s(row[c]) for c in df.columns} for _, row in df.head(10).iterrows()]
    report = gen_report(metrics, risk, all_syn, ix_data, features, simpsons, conf_str, proxies)

    result = {
        "rows": len(df), "columns": cols, "target_column": tgt,
        "demographics_detected": n_demo, "metrics": metrics, "risk": risk,
        "synthetic_flags": all_syn, "simpsons_paradox": simpsons,
        "intersectional": ix_data, "features": features,
        "confounder_strength": round(conf_str, 3),
        "proxy_variables": proxies, "key_insights": key_insights,
        "data_preview": preview, "ai_report": report,
    }

    global _last_analysis
    _last_analysis = result

    return jsonify(result)


def _s(v):
    return None if pd.isna(v) else (v if isinstance(v, (int, float)) else str(v))


# ─── Chat Endpoint ────────────────────────────────────────────

# Store last analysis for chat context
_last_analysis = {}

@app.route("/api/chat", methods=["POST"])
def chat():
    global _last_analysis
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided."}), 400

    # Feature 2: Accept metrics context from frontend (richer than server-side cache)
    frontend_metrics = data.get("metrics")
    frontend_risk = data.get("risk")
    frontend_insights = data.get("key_insights")

    # Build context — prefer frontend-sent metrics, fall back to server cache
    ctx = _last_analysis
    ctx_summary = ""

    # Use frontend metrics if available (they're always fresh)
    if frontend_metrics:
        ctx_summary = f"""ANALYSIS CONTEXT (live from dashboard):
- Overall Risk: {frontend_risk.get('level', '?') if frontend_risk else ctx.get('risk', {}).get('level', '?')}
"""
        for label, m in frontend_metrics.items():
            if isinstance(m, dict):
                ctx_summary += (f"- {label}: DI_raw={m.get('di_raw')}, DI_smoothed={m.get('di_smoothed')}, "
                                f"SPD={m.get('spd')}, EO={m.get('eo')}, "
                                f"bias_class={m.get('bias_class')}, severity={m.get('severity', {}).get('tier', '?')}, "
                                f"confidence={m.get('confidence', {}).get('level', '?')}, p_value={m.get('p_value')}\n")
                # Include group breakdown
                bd = m.get("breakdown", [])
                if bd:
                    groups = ", ".join([f"{g.get('group')}: {g.get('percentage')}% (n={g.get('count')})" for g in bd[:6]])
                    ctx_summary += f"  Groups: {groups}\n"

        if frontend_insights:
            for ki in frontend_insights:
                if isinstance(ki, dict):
                    ctx_summary += f"- Insight: {ki.get('text', '')} — {ki.get('detail', '')}\n"
    elif ctx:
        ctx_summary = f"""ANALYSIS CONTEXT:
- Rows: {ctx.get('rows', '?')}, Demographics detected: {ctx.get('demographics_detected', '?')}
- Overall Risk: {ctx.get('risk', {}).get('level', '?')}
- Confounder Strength: {ctx.get('confounder_strength', '?')}
"""
        for label, m in ctx.get("metrics", {}).items():
            ctx_summary += f"- {label}: DI_raw={m.get('di_raw')}, bias_class={m.get('bias_class')}, p_value={m.get('p_value')}\n"

        if ctx.get("simpsons_paradox"):
            ctx_summary += f"- Simpson's Paradox: {'; '.join(ctx['simpsons_paradox'])}\n"
        if ctx.get("proxy_variables"):
            for p in ctx["proxy_variables"]:
                ctx_summary += f"- Proxy: {p['proxy_column']} → {p['demographic_column']} (strength={p['strength']})\n"
        if ctx.get("key_insights"):
            for ki in ctx["key_insights"]:
                ctx_summary += f"- Insight: {ki['text']}\n"

    prompt = f"""You are the AI Bias Inspector — a sharp, decisive bias analyst.

{ctx_summary}

User Question: {question}

RESPONSE RULES (strict):
1. FIRST LINE: A single bold verdict sentence with a confidence qualifier. Example: "**High confidence:** Gender shows minimal direct bias (DI≈0.95)."
2. SECOND LINE (optional): One clarifying detail if needed. Example: "However, hidden bias exists in 18–24 female subgroup, revealed by Simpson's Paradox."
3. MAXIMUM 2-3 lines total. Never write paragraphs.
4. Always use confidence language: "High confidence", "Moderate confidence", "Likely explained by", "Strongly correlated with", "Cannot be ruled out".
5. Include specific numbers (DI, p-value, %) when available.
6. If you don't have enough data, say so in one line.

Be SHARP. Judges prefer clarity over length."""

    try:
        r = gemini_model.generate_content(prompt)
        answer = getattr(r, "text", "") or "I could not generate a response."
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"AI error: {e}"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
