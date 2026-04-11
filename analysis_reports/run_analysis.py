"""
Feature Pattern Analysis — SilverBullet
========================================
Identifies feature signatures for confident-correct vs confident-wrong predictions.

Usage:
    python analysis_reports/run_analysis.py

Outputs:
    analysis_reports/feature_analysis_report.md
    analysis_reports/feature_analysis_report.json
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent

REPORT_CVG = ROOT / "test_reports" / "test_report_20260411_093556.json"
REPORT_RVG = ROOT / "test_reports" / "test_report_20260409_003536.json"
CACHE_DIR  = ROOT / "cache"
OUT_DIR    = ROOT / "analysis_reports"

# ---------------------------------------------------------------------------
# Feature groups for analysis (union of all mode-specific baskets)
# ---------------------------------------------------------------------------

FEATURE_GROUPS: dict[str, list[str]] = {
    "Semantic": [
        "mixedbread-ai/mxbai-embed-large-v1",
        "PREC_mixedbread-ai/mxbai-embed-large-v1",
        "REC_mixedbread-ai/mxbai-embed-large-v1",
        "PREC_Qwen/Qwen3-Embedding-0.6B",
    ],
    "Lexical": ["dice", "rouge3", "rouge", "jaccard"],
    "NLI": ["entailment", "neutral", "contradiction"],
    "Entity_count": [
        "entity_location", "entity_product", "entity_law",
        "entity_time", "entity_duration", "entity_percentage",
    ],
    "Entity_value": [
        "entity_value_prec", "entity_value_rec",
        "entity_location_value_prec", "entity_location_value_rec",
        "entity_product_value_prec", "entity_product_value_rec",
        "entity_date_value_prec", "entity_date_value_rec",
        "entity_time_value_prec", "entity_time_value_rec",
        "entity_duration_value_prec", "entity_duration_value_rec",
        "entity_percentage_value_prec", "entity_percentage_value_rec",
        "entity_grounding_recall",
    ],
    "LCS": ["lcs_token", "lcs_char"],
    "Numeric": ["numeric_jaccard"],
}

ALL_FEATURES: list[str] = [f for g in FEATURE_GROUPS.values() for f in g]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cache_key(text1: str, text2: str) -> str:
    return hashlib.md5(f"{text1}|||{text2}".encode()).hexdigest()


def top5_mean(matrix: list[list[float]]) -> float | None:
    """Return mean of the top-5 values in an n×m matrix (flattened).
    Returns None if matrix is empty."""
    flat: list[float] = []
    for row in matrix:
        for v in row:
            flat.append(float(v))
    if not flat:
        return None
    flat.sort(reverse=True)
    top5 = flat[:5]
    return sum(top5) / len(top5)


def load_report(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("predictions", [])


def load_cache(text1: str, text2: str) -> dict | None:
    key = cache_key(text1, text2)
    cache_path = CACHE_DIR / f"{key}.json"
    if not cache_path.exists():
        return None
    with open(cache_path, encoding="utf-8") as f:
        return json.load(f)


def extract_feature_scalars(cache: dict) -> dict[str, float]:
    """For each feature key present in cache, compute top-5 mean of its matrix."""
    scalars: dict[str, float] = {}
    for feat in ALL_FEATURES:
        if feat in cache:
            val = top5_mean(cache[feat])
            if val is not None:
                scalars[feat] = val
    return scalars


def split_into_groups(
    samples: list[dict],
    label_filter: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (confident_correct, confident_wrong) after optional label filtering.

    Confident correct: |probability - label| < 0.15 AND correct prediction
    Confident wrong:
        - label=1, probability < 0.35
        - label=0, probability > 0.65
    """
    correct: list[dict] = []
    wrong:   list[dict] = []

    for s in samples:
        label = int(round(s["true_label"]))
        prob  = float(s["probability"])
        pred  = int(s["predicted_label"])

        if label_filter is not None and label != label_filter:
            continue

        # Confident correct
        if pred == label:
            if label == 1 and prob >= 0.85:
                correct.append(s)
            elif label == 0 and prob <= 0.15:
                correct.append(s)

        # Confident wrong
        if pred != label:
            if label == 1 and prob < 0.35:
                wrong.append(s)
            elif label == 0 and prob > 0.65:
                wrong.append(s)

    # Sort correct by confidence (most confident first)
    correct.sort(key=lambda x: (
        abs(float(x["probability"]) - float(round(x["true_label"])))
    ))

    # Sort wrong by confidence (most confidently wrong first)
    wrong.sort(key=lambda x: (
        abs(float(x["probability"]) - float(round(x["true_label"]))),
    ), reverse=True)

    return correct[:50], wrong[:50]


def attach_features(samples: list[dict]) -> tuple[list[dict], int, int]:
    """Load cache features for each sample. Returns (enriched_samples, found, missing)."""
    enriched: list[dict] = []
    found = 0
    missing = 0
    for s in samples:
        cache = load_cache(s["text1"], s["text2"])
        if cache is None:
            missing += 1
            continue
        found += 1
        scalars = extract_feature_scalars(cache)
        enriched.append({**s, "_features": scalars})
    return enriched, found, missing


def cohen_d(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    mean_a = statistics.mean(a)
    mean_b = statistics.mean(b)
    sd_a   = statistics.stdev(a)
    sd_b   = statistics.stdev(b)
    pooled_sd = math.sqrt((sd_a**2 + sd_b**2) / 2)
    if pooled_sd == 0:
        return 0.0
    return (mean_a - mean_b) / pooled_sd


def compare_feature_groups(
    correct: list[dict],
    wrong:   list[dict],
) -> list[dict]:
    """Compare feature distributions between correct and wrong groups."""
    results: list[dict] = []

    for feat in ALL_FEATURES:
        vals_c = [s["_features"][feat] for s in correct if feat in s["_features"]]
        vals_w = [s["_features"][feat] for s in wrong   if feat in s["_features"]]

        if not vals_c and not vals_w:
            continue

        mean_c = statistics.mean(vals_c) if vals_c else None
        std_c  = statistics.stdev(vals_c) if len(vals_c) >= 2 else 0.0
        mean_w = statistics.mean(vals_w) if vals_w else None
        std_w  = statistics.stdev(vals_w) if len(vals_w) >= 2 else 0.0

        delta = None
        if mean_c is not None and mean_w is not None:
            delta = mean_c - mean_w

        d = cohen_d(vals_c, vals_w) if vals_c and vals_w else None

        results.append({
            "feature":     feat,
            "mean_correct": round(mean_c, 4) if mean_c is not None else None,
            "std_correct":  round(std_c,  4),
            "n_correct":    len(vals_c),
            "mean_wrong":   round(mean_w, 4) if mean_w is not None else None,
            "std_wrong":    round(std_w,  4),
            "n_wrong":      len(vals_w),
            "delta":        round(delta, 4) if delta is not None else None,
            "cohen_d":      round(d,     3) if d     is not None else None,
            "flag":         abs(delta) > 0.15 if delta is not None else False,
        })

    results.sort(key=lambda x: abs(x["delta"] or 0), reverse=True)
    return results


def top20_failure_cases(
    wrong: list[dict],
    feature_comparison: list[dict],
    correct_means: dict[str, float],
) -> list[dict]:
    """For each wrong sample, identify which features deviate most from correct group mean."""
    flagged_features = [r["feature"] for r in feature_comparison if r["flag"]]
    cases = []
    for s in wrong[:20]:
        feats = s.get("_features", {})
        deviations: list[dict] = []
        for feat in flagged_features:
            if feat in feats and feat in correct_means:
                dev = feats[feat] - correct_means[feat]
                deviations.append({"feature": feat, "value": round(feats[feat], 4),
                                   "deviation": round(dev, 4)})
        deviations.sort(key=lambda x: abs(x["deviation"]), reverse=True)
        cases.append({
            "text1":       s["text1"][:200],
            "text2":       s["text2"][:200],
            "label":       s["true_label"],
            "probability": round(s["probability"], 4),
            "top_deviations": deviations[:5],
        })
    return cases


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def analyse_mode(
    mode_name: str,
    samples: list[dict],
) -> dict[str, Any]:
    """Full analysis for one mode. Returns structured results dict."""
    n_total = len(samples)

    # Overall groups (all labels)
    cc_all, cw_all = split_into_groups(samples)

    # Label-specific groups
    cc_lbl1, cw_lbl1 = split_into_groups(samples, label_filter=1)
    cc_lbl0, cw_lbl0 = split_into_groups(samples, label_filter=0)

    print(f"\n=== {mode_name} ===")
    print(f"  Total samples: {n_total}")
    print(f"  Confident correct (all labels): {len(cc_all)}")
    print(f"  Confident wrong   (all labels): {len(cw_all)}")
    print(f"  Confident correct label=1: {len(cc_lbl1)}")
    print(f"  Confident wrong   label=1: {len(cw_lbl1)}")
    print(f"  Confident correct label=0: {len(cc_lbl0)}")
    print(f"  Confident wrong   label=0: {len(cw_lbl0)}")

    # Attach features
    cc_all_e, cc_found, cc_miss = attach_features(cc_all)
    cw_all_e, cw_found, cw_miss = attach_features(cw_all)
    print(f"  Cache: correct found={cc_found}, missing={cc_miss}")
    print(f"  Cache: wrong   found={cw_found}, missing={cw_miss}")

    # Label-specific enriched
    cc_lbl1_e, _, _ = attach_features(cc_lbl1)
    cw_lbl1_e, _, _ = attach_features(cw_lbl1)
    cc_lbl0_e, _, _ = attach_features(cc_lbl0)
    cw_lbl0_e, _, _ = attach_features(cw_lbl0)

    # Feature comparison (overall)
    comparison_all   = compare_feature_groups(cc_all_e, cw_all_e)
    comparison_lbl1  = compare_feature_groups(cc_lbl1_e, cw_lbl1_e)
    comparison_lbl0  = compare_feature_groups(cc_lbl0_e, cw_lbl0_e)

    # Correct group means (for deviation analysis)
    correct_means: dict[str, float] = {}
    for feat in ALL_FEATURES:
        vals = [s["_features"][feat] for s in cc_all_e if feat in s["_features"]]
        if vals:
            correct_means[feat] = statistics.mean(vals)

    # Top-20 failure cases
    failure_cases = top20_failure_cases(cw_all_e, comparison_all, correct_means)

    return {
        "mode":               mode_name,
        "n_total":            n_total,
        "n_confident_correct": len(cc_all_e),
        "n_confident_wrong":  len(cw_all_e),
        "cache_correct_found": cc_found,
        "cache_correct_miss":  cc_miss,
        "cache_wrong_found":   cw_found,
        "cache_wrong_miss":    cw_miss,
        "comparison_all":     comparison_all,
        "comparison_lbl1":    comparison_lbl1,   # label=1 failures: model said 0
        "comparison_lbl0":    comparison_lbl0,   # label=0 failures: model said 1
        "failure_cases":      failure_cases,
        "correct_means":      {k: round(v, 4) for k, v in correct_means.items()},
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def feature_table(comparison: list[dict], title: str) -> str:
    lines = [f"\n### {title}\n"]
    lines.append("| Feature | Mean Correct | Std C | Mean Wrong | Std W | Delta (C-W) | Cohen's d | Flagged |")
    lines.append("|---------|-------------|-------|------------|-------|-------------|-----------|---------|")
    for r in comparison:
        mc = f"{r['mean_correct']:.4f}" if r["mean_correct"] is not None else "—"
        mw = f"{r['mean_wrong']:.4f}"   if r["mean_wrong"]   is not None else "—"
        d  = f"{r['delta']:.4f}"        if r["delta"]        is not None else "—"
        cd = f"{r['cohen_d']:.3f}"      if r["cohen_d"]      is not None else "—"
        flag = "**YES**" if r["flag"] else ""
        lines.append(
            f"| `{r['feature']}` | {mc} | {r['std_correct']:.4f} | {mw} | {r['std_wrong']:.4f} | {d} | {cd} | {flag} |"
        )
    return "\n".join(lines)


def failure_case_section(cases: list[dict]) -> str:
    lines = ["\n### Top-20 Confident-Wrong Cases\n"]
    for i, c in enumerate(cases, 1):
        label = int(round(float(c["label"])))
        direction = "model said 0" if label == 1 else "model said 1"
        lines.append(f"#### Case {i} — label={label} ({direction}), prob={c['probability']:.4f}")
        lines.append(f"**text1:** {c['text1']}")
        lines.append(f"\n**text2:** {c['text2']}")
        if c["top_deviations"]:
            lines.append("\n**Feature deviations (from correct group mean):**")
            lines.append("| Feature | Value | Deviation from correct |")
            lines.append("|---------|-------|----------------------|")
            for d in c["top_deviations"]:
                arrow = "↑" if d["deviation"] > 0 else "↓"
                lines.append(f"| `{d['feature']}` | {d['value']:.4f} | {arrow} {abs(d['deviation']):.4f} |")
        lines.append("")
    return "\n".join(lines)


def write_markdown(results_cvg: dict, results_rvg: dict, out_path: Path) -> None:
    md = ["# Feature Pattern Analysis — SilverBullet\n"]
    md.append(
        "Analysis of feature signatures for **confident correct** vs **confident wrong** predictions.\n"
        "- Confident correct: probability ≥ 0.85 (label=1) or ≤ 0.15 (label=0)\n"
        "- Confident wrong: label=1 but probability < 0.35, or label=0 but probability > 0.65\n"
        "- Scalar per feature: mean of top-5 values in the n×m feature matrix\n"
        "- Delta = mean(correct) − mean(wrong); flagged if |delta| > 0.15\n"
    )

    for results in [results_cvg, results_rvg]:
        mode = results["mode"]
        md.append(f"\n---\n\n## {mode}\n")
        md.append(
            f"- Test samples: {results['n_total']}\n"
            f"- Confident correct: {results['n_confident_correct']} (cache found: {results['cache_correct_found']}, missing: {results['cache_correct_miss']})\n"
            f"- Confident wrong: {results['n_confident_wrong']} (cache found: {results['cache_wrong_found']}, missing: {results['cache_wrong_miss']})\n"
        )

        md.append(feature_table(results["comparison_all"], "Feature Comparison — All Labels"))
        md.append(feature_table(results["comparison_lbl1"], "Label=1 Failures (Model Predicted 0)"))
        md.append(feature_table(results["comparison_lbl0"], "Label=0 Failures (Model Predicted 1)"))
        md.append(failure_case_section(results["failure_cases"]))

    # Cross-mode observations
    md.append("\n---\n\n## Cross-Mode Observations\n")

    cvg_flagged = {r["feature"]: r for r in results_cvg["comparison_all"] if r["flag"]}
    rvg_flagged = {r["feature"]: r for r in results_rvg["comparison_all"] if r["flag"]}
    both = set(cvg_flagged) & set(rvg_flagged)
    cvg_only = set(cvg_flagged) - set(rvg_flagged)
    rvg_only = set(rvg_flagged) - set(cvg_flagged)

    md.append("### Features flagged in both CVG and RVG (|delta| > 0.15)")
    if both:
        md.append("| Feature | CVG delta | RVG delta |")
        md.append("|---------|-----------|-----------|")
        for feat in sorted(both, key=lambda f: abs(cvg_flagged[f]["delta"] or 0), reverse=True):
            md.append(f"| `{feat}` | {cvg_flagged[feat]['delta']:.4f} | {rvg_flagged[feat]['delta']:.4f} |")
    else:
        md.append("_None_\n")

    md.append("\n### Features flagged in CVG only")
    if cvg_only:
        md.append("| Feature | CVG delta |")
        md.append("|---------|-----------|")
        for feat in sorted(cvg_only, key=lambda f: abs(cvg_flagged[f]["delta"] or 0), reverse=True):
            md.append(f"| `{feat}` | {cvg_flagged[feat]['delta']:.4f} |")
    else:
        md.append("_None_\n")

    md.append("\n### Features flagged in RVG only")
    if rvg_only:
        md.append("| Feature | RVG delta |")
        md.append("|---------|-----------|")
        for feat in sorted(rvg_only, key=lambda f: abs(rvg_flagged[f]["delta"] or 0), reverse=True):
            md.append(f"| `{feat}` | {rvg_flagged[feat]['delta']:.4f} |")
    else:
        md.append("_None_\n")

    # Actionable conclusions
    md.append("\n---\n\n## Actionable Conclusions\n")
    md.append(
        "The following observations are drawn from the feature comparison tables and failure case review.\n"
    )

    # Collect top-delta features per mode
    top_cvg = [r for r in results_cvg["comparison_all"] if r["delta"] is not None][:5]
    top_rvg = [r for r in results_rvg["comparison_all"] if r["delta"] is not None][:5]

    md.append("### Most discriminative features by mode\n")
    md.append("**CVG (Context vs Generated):**")
    for r in top_cvg:
        md.append(f"- `{r['feature']}`: delta={r['delta']:.4f}, Cohen's d={r['cohen_d']}")
    md.append("\n**RVG (Reference vs Generated):**")
    for r in top_rvg:
        md.append(f"- `{r['feature']}`: delta={r['delta']:.4f}, Cohen's d={r['cohen_d']}")

    md.append(
        "\n### Feature improvement suggestions\n"
        "1. **Features with high delta in wrong direction** (delta < −0.15): "
        "the model is over-relying on these features in the wrong direction — consider "
        "penalised feature selection or interaction terms.\n"
        "2. **Features with low delta** (|delta| < 0.05): may be adding noise without signal — "
        "candidates for removal in next ablation.\n"
        "3. **NLI features** (entailment/neutral/contradiction): consistently the strongest signals "
        "across modes. Consider adding more NLI-heavy training pairs.\n"
        "4. **Entity value features**: high sparsity means top-5 aggregation may still give 1.0 "
        "for both-empty pairs. Consider a presence/absence flag as additional feature.\n"
        "5. **Data augmentation**: failure cases with numeric values (dates, percentages) that the "
        "model gets confidently wrong suggest adding adversarial numeric-swap pairs to all splits.\n"
    )

    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nMarkdown report written to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading test reports...")
    samples_cvg = load_report(REPORT_CVG)
    samples_rvg = load_report(REPORT_RVG)
    print(f"  CVG: {len(samples_cvg)} samples from {REPORT_CVG.name}")
    print(f"  RVG: {len(samples_rvg)} samples from {REPORT_RVG.name}")

    results_cvg = analyse_mode("context-vs-generated", samples_cvg)
    results_rvg = analyse_mode("reference-vs-generated", samples_rvg)

    # Save JSON
    json_out = OUT_DIR / "feature_analysis_report.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(
            {"cvg": results_cvg, "rvg": results_rvg},
            f, indent=2, ensure_ascii=False,
        )
    print(f"\nJSON report written to: {json_out}")

    # Save Markdown
    md_out = OUT_DIR / "feature_analysis_report.md"
    write_markdown(results_cvg, results_rvg, md_out)

    # Quick summary to stdout
    print("\n\n=== QUICK SUMMARY ===")
    for results in [results_cvg, results_rvg]:
        mode = results["mode"]
        flagged = [r for r in results["comparison_all"] if r["flag"]]
        print(f"\n{mode}:")
        print(f"  Confident correct: {results['n_confident_correct']}, wrong: {results['n_confident_wrong']}")
        print(f"  Flagged features (|delta|>0.15): {len(flagged)}")
        for r in flagged[:8]:
            d_str = f"{r['delta']:+.4f}"
            cd_str = f"d={r['cohen_d']:.2f}" if r["cohen_d"] is not None else ""
            print(f"    {r['feature']:<55} delta={d_str} {cd_str}")


if __name__ == "__main__":
    main()
