"""
Feature ablation via clustering — SilverBullet v3.0 pipeline.

For each labelled pair, reduces each 34-feature map to its spatial mean
-> Nx34 matrix. Then applies six measures of feature-label association
and assigns each feature a tier (STRONG/MODERATE/WEAK/MARGINAL/NOISE)
and a verdict (KEEP/REVIEW/DROP) per the framework in:
    ablation_reports/framework.md

Run:
    python -m backend.ablation_cluster
    python -m backend.ablation_cluster --mode context-vs-generated --tag post-cvg
    python -m backend.ablation_cluster --split train --out ablation_reports/

All six measures + decisions are written to a timestamped experiment
directory under ablation_reports/experiments/{timestamp}_{tag}/.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from backend.feature_cache import FeatureCache
from backend.feature_registry import FEATURE_KEYS

# ---------------------------------------------------------------------------
# Framework constants (must match framework.md v1.0)
# ---------------------------------------------------------------------------
FRAMEWORK_VERSION = "1.0"
BONFERRONI_ALPHA = 0.05

# Tier thresholds
STRONG_THRESHOLDS   = dict(abs_pearson=0.25, abs_spearman=0.25, abs_cohens_d=0.50, mi_bits=0.020, mode_consistency=0.67)
MODERATE_THRESHOLDS = dict(abs_pearson=0.10, abs_spearman=0.10, abs_cohens_d=0.25, mi_bits=0.008, mode_consistency=0.33)
WEAK_THRESHOLDS     = dict(abs_pearson=0.05, p_uncorrected=0.05, abs_cohens_d=0.10, mi_bits=0.002)
MARGINAL_THRESHOLDS = dict(abs_pearson=0.02, abs_cohens_d=0.05, mi_bits=0.001)

REDUNDANCY_THRESHOLD = 0.85  # max cross-r above which a feature is considered redundant


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_pairs(mode: str | None, split: str | None) -> list[dict]:
    data_root = Path("data")
    records = []
    modes  = [mode]  if mode  else ["context-vs-generated", "reference-vs-generated", "model-vs-model"]
    splits = [split] if split else ["train", "validate", "test"]
    for m in modes:
        for s in splits:
            p = data_root / m / f"{s}.json"
            if not p.exists():
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            for row in data.get("data", []):
                records.append({"text1": row["text1"], "text2": row["text2"],
                                "label": float(row["label"]), "mode": m})
    return records


def _feature_vector(cache_entry: dict) -> np.ndarray | None:
    if not isinstance(cache_entry, dict):
        return None
    if set(cache_entry.keys()) != set(FEATURE_KEYS):
        return None
    vec = [float(np.array(cache_entry[k], dtype=np.float32).mean()) for k in FEATURE_KEYS]
    return np.array(vec, dtype=np.float32)


# ---------------------------------------------------------------------------
# Six measures
# ---------------------------------------------------------------------------

def _compute_pearson(X: np.ndarray, y: np.ndarray, bonferroni_thresh: float) -> dict:
    out = {}
    for i, key in enumerate(FEATURE_KEYS):
        r, p = stats.pearsonr(X[:, i], y)
        out[key] = {"pearson_r": float(r), "pearson_p": float(p),
                    "pearson_significant_bonferroni": bool(p < bonferroni_thresh)}
    return out


def _compute_spearman(X: np.ndarray, y: np.ndarray) -> dict:
    out = {}
    for i, key in enumerate(FEATURE_KEYS):
        r, p = stats.spearmanr(X[:, i], y)
        out[key] = {"spearman_r": float(r), "spearman_p": float(p)}
    return out


def _compute_cohens_d(X: np.ndarray, y_bin: np.ndarray) -> dict:
    out = {}
    for i, key in enumerate(FEATURE_KEYS):
        x0 = X[y_bin == 0, i]
        x1 = X[y_bin == 1, i]
        pooled = np.sqrt((x0.std(ddof=1)**2 + x1.std(ddof=1)**2) / 2)
        if pooled < 1e-12:
            out[key] = {"cohens_d": 0.0, "cohens_d_note": "zero_variance"}
        else:
            out[key] = {"cohens_d": float((x1.mean() - x0.mean()) / pooled)}
    return out


def _compute_mutual_info(X: np.ndarray, y_bin: np.ndarray) -> dict:
    mi_nats = mutual_info_classif(X, y_bin, discrete_features=False,
                                  n_neighbors=3, random_state=42)
    mi_bits = mi_nats / np.log(2)
    return {key: {"mutual_info_bits": float(mi_bits[i])} for i, key in enumerate(FEATURE_KEYS)}


def _compute_cross_correlations(X: np.ndarray) -> dict:
    assert X.shape[1] == len(FEATURE_KEYS)
    C = np.corrcoef(X.T)           # (34, 34)
    np.fill_diagonal(C, 0.0)
    out = {}
    for i, key in enumerate(FEATURE_KEYS):
        abs_row = np.abs(C[i])
        top3_idx = np.argsort(abs_row)[::-1][:3]
        out[key] = {
            "max_cross_r": float(abs_row[top3_idx[0]]),
            "top3_correlated_peers": [
                {"feature": FEATURE_KEYS[j], "cross_r": float(C[i, j])}
                for j in top3_idx
            ],
        }
    return out


def _compute_mode_consistency(X: np.ndarray, y: np.ndarray,
                               modes_arr: np.ndarray) -> dict:
    unique_modes = sorted(set(modes_arr))
    if len(unique_modes) < 2:
        return {key: {"mode_consistency": None, "mode_ranks": None}
                for key in FEATURE_KEYS}

    mode_ranks: dict[str, list[int | None]] = {key: {} for key in FEATURE_KEYS}
    for m in unique_modes:
        mask = modes_arr == m
        Xm, ym = X[mask], y[mask]
        if mask.sum() < 10:
            for key in FEATURE_KEYS:
                mode_ranks[key][m] = None
            continue
        abs_rs = [abs(float(stats.pearsonr(Xm[:, i], ym)[0])) for i in range(len(FEATURE_KEYS))]
        # rank 1 = highest |r|
        order = np.argsort(abs_rs)[::-1]
        ranks = {FEATURE_KEYS[order[rank]]: rank + 1 for rank in range(len(FEATURE_KEYS))}
        for key in FEATURE_KEYS:
            mode_ranks[key][m] = ranks[key]

    half = len(FEATURE_KEYS) // 2
    out = {}
    for key in FEATURE_KEYS:
        ranks_for_key = [mode_ranks[key].get(m) for m in unique_modes]
        valid = [r for r in ranks_for_key if r is not None]
        top_half_count = sum(1 for r in valid if r <= half)
        consistency = top_half_count / len(valid) if valid else None
        out[key] = {"mode_consistency": consistency, "mode_ranks": mode_ranks[key]}
    return out


# ---------------------------------------------------------------------------
# Tier + verdict assignment
# ---------------------------------------------------------------------------

def _assign_tier(fs: dict, multi_mode: bool) -> str:
    """Assign a tier based on the six measures. fs is the merged feature dict."""
    abs_pr  = abs(fs.get("pearson_r", 0.0))
    abs_sr  = abs(fs.get("spearman_r", 0.0))
    abs_d   = abs(fs.get("cohens_d", 0.0))
    mi      = fs.get("mutual_info_bits", 0.0)
    mc      = fs.get("mode_consistency")
    p       = fs.get("pearson_p", 1.0)
    bonf_ok = fs.get("pearson_significant_bonferroni", False)

    # STRONG: all five conditions (mode_consistency only checked in multi-mode run)
    mc_ok_strong = (mc is None) or (mc >= STRONG_THRESHOLDS["mode_consistency"])
    if (abs_pr >= STRONG_THRESHOLDS["abs_pearson"] and bonf_ok and
            abs_sr >= STRONG_THRESHOLDS["abs_spearman"] and
            abs_d  >= STRONG_THRESHOLDS["abs_cohens_d"] and
            mi     >= STRONG_THRESHOLDS["mi_bits"] and
            (not multi_mode or mc_ok_strong)):
        return "STRONG"

    # MODERATE
    mc_ok_mod = (mc is None) or (mc >= MODERATE_THRESHOLDS["mode_consistency"])
    if (abs_pr >= MODERATE_THRESHOLDS["abs_pearson"] and bonf_ok and
            abs_sr >= MODERATE_THRESHOLDS["abs_spearman"] and
            abs_d  >= MODERATE_THRESHOLDS["abs_cohens_d"] and
            mi     >= MODERATE_THRESHOLDS["mi_bits"] and
            (not multi_mode or mc_ok_mod)):
        return "MODERATE"

    # WEAK: p < 0.05 uncorrected (not Bonferroni), other thresholds lower
    if (abs_pr >= WEAK_THRESHOLDS["abs_pearson"] and
            p < WEAK_THRESHOLDS["p_uncorrected"] and
            abs_d >= WEAK_THRESHOLDS["abs_cohens_d"] and
            mi    >= WEAK_THRESHOLDS["mi_bits"]):
        return "WEAK"

    # MARGINAL: any one of three weak conditions
    if (abs_pr >= MARGINAL_THRESHOLDS["abs_pearson"] or
            abs_d >= MARGINAL_THRESHOLDS["abs_cohens_d"] or
            mi    >= MARGINAL_THRESHOLDS["mi_bits"]):
        return "MARGINAL"

    return "NOISE"


def _assign_verdict(tier: str, fs: dict) -> tuple[str, str]:
    """Return (verdict, reason) string pair."""
    redundant  = fs.get("max_cross_r", 0.0) >= REDUNDANCY_THRESHOLD
    top_peer   = (fs.get("top3_correlated_peers") or [{}])[0].get("feature", "unknown")
    top_peer_r = (fs.get("top3_correlated_peers") or [{}])[0].get("cross_r", 0.0)

    pr  = fs.get("pearson_r", 0.0)
    p   = fs.get("pearson_p", 1.0)
    sr  = fs.get("spearman_r", 0.0)
    d   = fs.get("cohens_d", 0.0)
    mi  = fs.get("mutual_info_bits", 0.0)
    mc  = fs.get("mode_consistency")
    bonf = "OK" if fs.get("pearson_significant_bonferroni") else "FAIL"

    base = (f"Pearson r={pr:+.4f} (p={p:.2e}, Bonferroni:{bonf}), "
            f"Spearman r={sr:+.4f}, Cohen's d={d:.4f}, MI={mi:.4f} bits")
    if mc is not None:
        base += f", mode_consistency={mc:.2f}"
    if redundant:
        base += f". REDUNDANT: max_cross_r={fs['max_cross_r']:.4f} with '{top_peer}' (cross_r={top_peer_r:+.4f})"

    if tier == "NOISE":
        return "DROP", f"No statistically significant or practically meaningful signal on any measure. {base}"

    if tier == "MARGINAL":
        return "REVIEW", (f"Detectable association but does not survive Bonferroni correction. "
                          f"Retain only if domain knowledge justifies inclusion. {base}")

    if tier == "WEAK":
        if redundant:
            return "REVIEW", (f"Small signal but near-duplicate (max_cross_r>={REDUNDANCY_THRESHOLD}) "
                              f"of higher-tier feature '{top_peer}'. Ablation test before dropping. {base}")
        return "KEEP", f"Small but validated independent signal. {base}"

    if tier == "MODERATE":
        if redundant:
            return "REVIEW", (f"Moderate signal but highly correlated with '{top_peer}'. "
                              f"Ablation test recommended before dropping. {base}")
        return "KEEP", f"Moderate validated signal with independent information. {base}"

    # STRONG
    return "KEEP", f"Strong multi-measure signal. Core feature. {base}"


# ---------------------------------------------------------------------------
# PCA + K-means helpers
# ---------------------------------------------------------------------------

def _run_pca(X_scaled: np.ndarray, n_components: int = 10) -> dict:
    pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
    pca.fit(X_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_).tolist()
    top3_per_pc = {}
    for i in range(pca.n_components_):
        loading = pca.components_[i]
        top3_idx = np.argsort(np.abs(loading))[::-1][:3]
        top3_per_pc[f"PC{i+1:02d}"] = [
            {"feature": FEATURE_KEYS[j], "loading": float(loading[j])}
            for j in top3_idx
        ]
    return {
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": cumvar,
        "top3_loadings_per_pc": top3_per_pc,
    }


def _run_kmeans(X_scaled: np.ndarray, y_bin: np.ndarray) -> dict:
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)
    ari = adjusted_rand_score(y_bin, cluster_labels)
    cluster_stats = []
    for c in [0, 1]:
        mask = cluster_labels == c
        cluster_stats.append({"cluster": c, "n": int(mask.sum()),
                               "label1_rate": float(y_bin[mask].mean())})
    centroid_diff = np.abs(km.cluster_centers_[0] - km.cluster_centers_[1])
    top10_idx = np.argsort(centroid_diff)[::-1][:10]
    return {
        "ari": float(ari),
        "cluster_stats": cluster_stats,
        "top10_discriminative": [
            {"feature": FEATURE_KEYS[j], "centroid_distance": float(centroid_diff[j])}
            for j in top10_idx
        ],
    }


def _run_per_mode_correlations(X: np.ndarray, y: np.ndarray,
                                modes_arr: np.ndarray) -> dict:
    out = {}
    for m in sorted(set(modes_arr)):
        mask = modes_arr == m
        Xm, ym = X[mask], y[mask]
        feats = {}
        for i, key in enumerate(FEATURE_KEYS):
            r, p = stats.pearsonr(Xm[:, i], ym)
            feats[key] = {"pearson_r": float(r), "pearson_p": float(p)}
        out[m] = {"n": int(mask.sum()), "features": feats}
    return out


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------

def _build_report(meta: dict, feature_stats_all: dict,
                  global_stats: dict, per_mode: dict) -> dict:
    tier_summary: dict[str, list] = {t: [] for t in ["STRONG","MODERATE","WEAK","MARGINAL","NOISE"]}
    verdict_summary: dict[str, list] = {"KEEP": [], "REVIEW": [], "DROP": []}

    for key, fs in feature_stats_all.items():
        tier_summary[fs["tier"]].append(key)
        verdict_summary[fs["verdict"]].append(key)

    return {
        "meta": meta,
        "global_stats": global_stats,
        "features": feature_stats_all,
        "tier_summary": tier_summary,
        "verdict_summary": verdict_summary,
        "per_mode_correlations": per_mode,
    }


def _generate_markdown(report: dict) -> str:
    meta = report["meta"]
    gs   = report["global_stats"]
    vs   = report["verdict_summary"]
    ts   = report["tier_summary"]
    feats = report["features"]

    lines = [
        "# Ablation Experiment Report",
        f"Generated: {meta['timestamp']}  |  Tag: {meta['tag']}  |  "
        f"Pairs: {meta['n_pairs']}  |  Missing: {meta['n_missing']}",
        "",
        "## Run Parameters",
        f"- Mode filter: {meta['mode_filter'] or 'all'}",
        f"- Split filter: {meta['split_filter'] or 'all'}",
        f"- Framework version: {meta['framework_version']}",
        f"- Bonferroni threshold: p < {meta['bonferroni_threshold']:.4e}",
        f"- Redundancy threshold: max_cross_r >= {REDUNDANCY_THRESHOLD}",
        "",
        "## Global Clustering Metrics",
        f"- K-means ARI (k=2): {gs['kmeans']['ari']:.4f}",
        f"- PCA PC01 explained variance: {gs['pca']['explained_variance'][0]:.3f}  "
        f"(cumulative PC05: {gs['pca']['cumulative_variance'][4]:.3f})",
        "",
        "## Verdict Summary",
        "| Verdict | Count | Features |",
        "|---------|-------|---------|",
    ]
    for v in ["KEEP", "REVIEW", "DROP"]:
        feats_v = vs[v]
        lines.append(f"| {v} | {len(feats_v)} | {', '.join(feats_v) or '—'} |")

    lines += [
        "",
        "## Tier Summary",
        "| Tier | Count | Features |",
        "|------|-------|---------|",
    ]
    for t in ["STRONG", "MODERATE", "WEAK", "MARGINAL", "NOISE"]:
        feats_t = ts[t]
        lines.append(f"| {t} | {len(feats_t)} | {', '.join(feats_t) or '—'} |")

    lines += ["", "## Per-Feature Detail", "(sorted by |Pearson r| descending)", ""]
    sorted_keys = sorted(feats.keys(), key=lambda k: abs(feats[k].get("pearson_r", 0)), reverse=True)
    for key in sorted_keys:
        fs = feats[key]
        mr = fs.get("max_cross_r", 0.0)
        peer = (fs.get("top3_correlated_peers") or [{}])[0].get("feature", "n/a")
        mc = fs.get("mode_consistency")
        mc_str = f"{mc:.2f}" if mc is not None else "n/a"
        bonf_str = "OK" if fs.get("pearson_significant_bonferroni") else "FAIL"
        p_str = f"{fs['pearson_p']:.2e}" if fs.get("pearson_p", 1) > 1e-10 else "< 1e-10"
        lines += [
            f"### {key} -- {fs['tier']} -- {fs['verdict']}",
            f"- Pearson r: {fs.get('pearson_r',0):+.4f}  (p={p_str}, Bonferroni: {bonf_str})",
            f"- Spearman r: {fs.get('spearman_r',0):+.4f}",
            f"- Cohen's d: {fs.get('cohens_d',0):.4f}",
            f"- Mutual Info: {fs.get('mutual_info_bits',0):.4f} bits",
            f"- Max cross-r: {mr:.4f} with '{peer}' (redundant: {'yes' if mr >= REDUNDANCY_THRESHOLD else 'no'})",
            f"- Mode consistency: {mc_str}",
            f"- **Reason:** {fs.get('reason', '')}",
            "",
        ]

    lines += [
        "## PCA Top Loadings (PC01-PC05)",
        "| PC | Feature 1 | Loading | Feature 2 | Loading | Feature 3 | Loading |",
        "|----|-----------|---------|-----------|---------|-----------|---------|",
    ]
    for pc_label, top3 in list(gs["pca"]["top3_loadings_per_pc"].items())[:5]:
        row = f"| {pc_label} |"
        for entry in top3:
            row += f" {entry['feature']} | {entry['loading']:+.4f} |"
        lines.append(row)

    lines += [
        "",
        "## Top-10 K-means Discriminative Features",
        "| Feature | Centroid Distance |",
        "|---------|-----------------|",
    ]
    for entry in gs["kmeans"]["top10_discriminative"]:
        lines.append(f"| {entry['feature']} | {entry['centroid_distance']:.4f} |")

    # Per-mode table
    lines += ["", "## Per-Mode Pearson r", "| Feature | " +
              " | ".join(report["per_mode_correlations"].keys()) + " |",
              "|---------|" + "---------|" * len(report["per_mode_correlations"])]
    for key in sorted_keys:
        row = f"| {key} |"
        for m, mdata in report["per_mode_correlations"].items():
            r = mdata["features"].get(key, {}).get("pearson_r", 0.0)
            row += f" {r:+.4f} |"
        lines.append(row)

    return "\n".join(lines)


def _save_experiment(report: dict, out_dir: str, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = tag.replace(" ", "_").replace("/", "-")
    exp_dir = Path(out_dir) / "experiments" / f"{ts}_{safe_tag}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    (exp_dir / "experiment.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (exp_dir / "experiment.md").write_text(
        _generate_markdown(report), encoding="utf-8"
    )
    return exp_dir


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_correlation_table(feature_stats_all: dict, bonferroni_thresh: float) -> None:
    print(f"\n-- Feature correlation with label (Bonferroni p < {bonferroni_thresh:.2e}) --")
    print(f"{'Feature':<45} {'r':>7}  {'p-value':>10}  {'sig':>5}  {'tier':<8}  {'verdict'}")
    print("-" * 95)
    sorted_keys = sorted(feature_stats_all.keys(),
                         key=lambda k: abs(feature_stats_all[k].get("pearson_r", 0)), reverse=True)
    for key in sorted_keys:
        fs = feature_stats_all[key]
        r  = fs.get("pearson_r", 0.0)
        p  = fs.get("pearson_p", 1.0)
        bonf = "***" if fs.get("pearson_significant_bonferroni") else ("*" if p < 0.05 else "ns")
        p_str = f"{p:.2e}" if p > 1e-10 else "<1e-10"
        bar = "#" * int(abs(r) * 30)
        print(f"{key:<45} {r:+.4f}  {p_str:>10}  {bonf:>5}  {fs['tier']:<8}  {fs['verdict']:<6}  {bar}")


def _print_verdict_summary(verdict_summary: dict) -> None:
    print("\n-- Verdict Summary --")
    for v in ["KEEP", "REVIEW", "DROP"]:
        items = verdict_summary[v]
        print(f"  {v:<6} ({len(items):2d}):  {', '.join(items)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(mode: str | None, split: str | None, out_dir: str, tag: str) -> None:
    cache = FeatureCache()
    records = _load_pairs(mode, split)
    print(f"Loaded {len(records)} labelled pairs")

    vectors, labels, modes_list = [], [], []
    missing = 0
    for rec in records:
        entry = cache.get_features(rec["text1"], rec["text2"])
        if entry is None:
            missing += 1; continue
        vec = _feature_vector(entry)
        if vec is None:
            missing += 1; continue
        vectors.append(vec)
        labels.append(rec["label"])
        modes_list.append(rec["mode"])

    print(f"  Cache hits : {len(vectors)}")
    print(f"  Cache miss : {missing}")

    if len(vectors) < 10:
        print("Not enough cached pairs -- run precompute_features.py first.")
        return

    X          = np.stack(vectors)
    y          = np.array(labels)
    y_bin      = (y >= 0.5).astype(int)
    modes_arr  = np.array(modes_list)
    multi_mode = len(set(modes_list)) > 1

    bonferroni_thresh = BONFERRONI_ALPHA / len(FEATURE_KEYS)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Six measures ---
    print("\nComputing measures...")
    pearson_stats = _compute_pearson(X, y, bonferroni_thresh)
    spearman_stats = _compute_spearman(X, y)
    cohens_stats   = _compute_cohens_d(X, y_bin)
    mi_stats       = _compute_mutual_info(X, y_bin)
    cross_stats    = _compute_cross_correlations(X)
    mode_stats     = _compute_mode_consistency(X, y, modes_arr)

    # --- Merge per-feature ---
    feature_stats_all = {}
    for key in FEATURE_KEYS:
        fs = {}
        for d in [pearson_stats[key], spearman_stats[key], cohens_stats[key],
                  mi_stats[key], cross_stats[key], mode_stats[key]]:
            fs.update(d)
        fs["tier"]   = _assign_tier(fs, multi_mode)
        verdict, reason = _assign_verdict(fs["tier"], fs)
        fs["verdict"] = verdict
        fs["reason"]  = reason
        feature_stats_all[key] = fs

    # --- Global analyses ---
    pca_stats   = _run_pca(X_scaled)
    kmeans_stats = _run_kmeans(X_scaled, y_bin)
    per_mode     = _run_per_mode_correlations(X, y, modes_arr)

    # --- Print ---
    _print_correlation_table(feature_stats_all, bonferroni_thresh)

    print("\n-- PCA explained variance --")
    for i, (ev, cv) in enumerate(zip(pca_stats["explained_variance"],
                                     pca_stats["cumulative_variance"])):
        bar = "#" * int(ev * 100)
        print(f"PC{i+1:02d}  {ev:.3f}  (cum {cv:.3f})  {bar}")
    print("\nTop-3 feature loadings per PC:")
    for pc_label, top3 in list(pca_stats["top3_loadings_per_pc"].items())[:5]:
        desc = "  ".join(f"{e['feature']}({e['loading']:+.3f})" for e in top3)
        print(f"  {pc_label}: {desc}")

    print(f"\n-- K-means (k=2) ARI: {kmeans_stats['ari']:.4f} --")
    for cs in kmeans_stats["cluster_stats"]:
        print(f"  Cluster {cs['cluster']}: {cs['n']} pairs, label=1 rate={cs['label1_rate']:.2%}")

    if multi_mode:
        print("\n-- Per-mode top-5 correlations --")
        for m, mdata in per_mode.items():
            sorted_feats = sorted(mdata["features"].items(),
                                  key=lambda x: abs(x[1]["pearson_r"]), reverse=True)[:5]
            print(f"\n  {m}  (n={mdata['n']})")
            for k, v in sorted_feats:
                p_str = f"{v['pearson_p']:.2e}" if v['pearson_p'] > 1e-10 else "<1e-10"
                print(f"    {k:<45} {v['pearson_r']:+.4f}  {p_str}")

    _print_verdict_summary(
        {"KEEP":   [k for k,v in feature_stats_all.items() if v["verdict"]=="KEEP"],
         "REVIEW": [k for k,v in feature_stats_all.items() if v["verdict"]=="REVIEW"],
         "DROP":   [k for k,v in feature_stats_all.items() if v["verdict"]=="DROP"]}
    )

    # --- Build + save ---
    meta = {
        "timestamp": datetime.now().isoformat(),
        "tag": tag,
        "mode_filter": mode,
        "split_filter": split,
        "n_pairs": len(vectors),
        "n_missing": missing,
        "n_features": len(FEATURE_KEYS),
        "feature_keys": FEATURE_KEYS,
        "framework_version": FRAMEWORK_VERSION,
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "bonferroni_threshold": bonferroni_thresh,
        "redundancy_threshold": REDUNDANCY_THRESHOLD,
    }
    global_stats = {"pca": pca_stats, "kmeans": kmeans_stats}
    report = _build_report(meta, feature_stats_all, global_stats, per_mode)
    exp_dir = _save_experiment(report, out_dir, tag)

    # Legacy flat report for backward compat
    legacy = {
        "n_pairs": len(vectors),
        "n_missing": missing,
        "feature_correlations": {k: {"r": v["pearson_r"], "p": v["pearson_p"]}
                                  for k, v in feature_stats_all.items()},
        "pca_explained_variance": pca_stats["explained_variance"],
        "kmeans_ari": kmeans_stats["ari"],
        "kmeans_top_discriminative": {e["feature"]: e["centroid_distance"]
                                      for e in kmeans_stats["top10_discriminative"]},
    }
    Path(out_dir, "ablation_cluster_report.json").write_text(
        json.dumps(legacy, indent=2), encoding="utf-8"
    )

    print(f"\nExperiment saved -> {exp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=None,
                        choices=["context-vs-generated",
                                 "reference-vs-generated",
                                 "model-vs-model"])
    parser.add_argument("--split", default=None,
                        choices=["train", "validate", "test"])
    parser.add_argument("--out", default="ablation_reports")
    parser.add_argument("--tag", default="run",
                        help="Short label for the experiment directory")
    args = parser.parse_args()
    run(args.mode, args.split, args.out, args.tag)
